"""Tableau de bord web embarqué (Flask, thread démon).

Tout est sur la page principale `/` : flux vidéo annoté, détections et journal. Les
données sont **poussées** au navigateur via SSE (Server-Sent Events) — une seule
connexion persistante, pas de requêtes répétées.

Routes :
  - GET /        -> page unique
  - GET /video   -> flux MJPEG (source de la balise <img>)
  - GET /events  -> flux SSE des données { fps, timestamp, objects[] }
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, request
from loguru import logger

# Page unique : vidéo à gauche, détections + journal à droite. Le JS écoute /events
# (SSE) et tient lui-même un journal d'événements (apparition / disparition d'un ID).
_PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Counter-UAS</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0d1117; color: #c9d1d9;
         font-family: ui-monospace, Menlo, Consolas, monospace; }
  .wrap { display: flex; gap: 12px; padding: 12px; flex-wrap: wrap; }
  .video { flex: 1 1 640px; min-width: 320px; }
  .video img { width: 100%; border: 1px solid #30363d; border-radius: 6px; display: block; }
  .side { flex: 1 1 320px; min-width: 280px; display: flex; flex-direction: column; gap: 12px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 12px; }
  .card h2 { margin: 0 0 8px; font-size: 12px; color: #8b949e; text-transform: uppercase;
             letter-spacing: 1px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #21262d; }
  th { color: #8b949e; font-weight: normal; }
  td.empty { color: #6e7681; text-align: center; padding: 14px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
  #log { height: 220px; overflow-y: auto; font-size: 12px; line-height: 1.55; }
  #log div { white-space: nowrap; }
  .new { color: #3fb950; } .lost { color: #f85149; }
  .ts { color: #6e7681; }
  button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px;
           padding: 6px 12px; font: inherit; cursor: pointer; }
  button:hover { background: #30363d; }
  .focus-btns { display: flex; gap: 8px; margin-bottom: 8px; }
  .focus-state { font-size: 13px; color: #8b949e; }
  .focus-state b { color: #c9d1d9; font-weight: normal; }
  #scene3d { width: 100%; height: 340px; border-radius: 4px; overflow: hidden;
             background: #0d1117; cursor: grab; }
  #scene3d:active { cursor: grabbing; }
  #scene3d canvas { display: block; }
  .hint { font-size: 11px; color: #6e7681; margin-top: 6px; }
</style></head>
<body>
  <div class="wrap">
    <div class="video"><img src="/video" alt="flux"></div>
    <div class="side">
      <div class="card"><h2>Vue 3D</h2>
        <div id="scene3d"></div>
        <div class="hint">caméra au centre (cône) · cibles placées par distance + FOV · souris : pivoter / molette : zoom</div>
      </div>
      <!--FOCUS_CARD-->
      <div class="card"><h2>Cibles suivies</h2>
        <table><thead><tr><th>ID</th><th>Classe</th><th>Conf.</th><th>Dist.</th></tr></thead>
        <tbody id="rows"></tbody></table>
      </div>
      <div class="card"><h2>Journal</h2><div id="log"></div></div>
    </div>
  </div>
<script src="/static/three.min.js"></script>
<script src="/static/OrbitControls.js"></script>
<script>
const colorFor = cls => cls.toLowerCase() === "drone" ? "#ff0000" : "#00ff00";  // drone = priorité
let known = new Map();           // id -> classe, pour détecter apparitions/disparitions
const logEl = document.getElementById("log");

function logEvent(kind, id, cls) {
  const t = new Date().toLocaleTimeString();
  const div = document.createElement("div");
  div.innerHTML = `<span class="ts">${t}</span> ` +
    `<span class="${kind}">${kind === "new" ? "+ ACQUIS" : "- PERDU"} #${id} ${cls}</span>`;
  logEl.appendChild(div);                       // plus récent en bas
  while (logEl.childNodes.length > 200) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;         // auto-scroll vers le plus récent
}

function render(d) {
  const seen = new Set();
  const rows = d.objects.map(o => {
    seen.add(o.id);
    if (!known.has(o.id)) { known.set(o.id, o.class); logEvent("new", o.id, o.class); }
    const dist = o.distance == null ? "—" : o.distance.toFixed(1) + " m";
    return `<tr><td><span class="dot" style="background:${colorFor(o.class)}"></span>${o.id}</td>` +
           `<td>${o.class}</td><td>${o.conf.toFixed(2)}</td><td>${dist}</td></tr>`;
  });
  document.getElementById("rows").innerHTML =
    rows.length ? rows.join("") : '<tr><td class="empty" colspan="4">aucune cible</td></tr>';
  for (const [id, cls] of [...known]) {
    if (!seen.has(id)) { logEvent("lost", id, cls); known.delete(id); }
  }
}

// --- Optique (caméra C3 18X) : autofocus / focus manuel / zoom manuel à la demande.
const FOCUS_STEP = __FOCUS_STEP__;
const ZOOM_STEP = __ZOOM_STEP__;
function af() { fetch("/focus/auto", { method: "POST" }); }
function nudge(d) { fetch("/focus/nudge?delta=" + d, { method: "POST" }); }
function zoom(d) { fetch("/zoom/nudge?delta=" + d, { method: "POST" }); }
function dezoomMax() { fetch("/zoom/wide", { method: "POST" }); }
const FOCUS_LABELS = { idle: "prêt", focusing: "mise au point…", done: "net", error: "erreur" };
function renderFocus(f) {
  const el = document.getElementById("focus-state");
  if (!el || !f) return;
  const st = FOCUS_LABELS[f.status] || f.status;
  const pos = f.position == null ? "—" : f.position;
  const net = (f.sharpness || 0).toFixed(0);
  el.innerHTML = `état <b>${st}</b> · pos <b>${pos}</b> · net <b>${net}</b>`;
}

// --- Vue 3D (Three.js) : caméra système au centre (cône) + cône de champ (FOV),
// cibles placées par projection sténopé (centre du bbox + distance + focale).
// Repère monde : +X droite, +Y haut, +Z = axe optique (devant la caméra). 1 unité = 1 m.
let scene3d, cam3d, renderer3d, controls3d, fovGroup, camRig;
let camKey = "";                       // (W,H,HFOV) courant, pour ne reconstruire le FOV qu'au besoin
let camEuler = null;                   // orientation pan/tilt courante (appliquée aussi aux cibles)
const meshes = new Map();              // id -> THREE.Mesh (sphère cible)

function init3D() {
  const host = document.getElementById("scene3d");
  if (!host || typeof THREE === "undefined") return;     // three.js absent -> on n'active pas la 3D
  try {
    scene3d = new THREE.Scene();
    scene3d.background = new THREE.Color(0x0d1117);

    cam3d = new THREE.PerspectiveCamera(55, host.clientWidth / host.clientHeight, 0.1, 4000);
    cam3d.position.set(0, 16, -28);                       // recule + au-dessus, regarde vers +Z

    renderer3d = new THREE.WebGLRenderer({ antialias: true });
    renderer3d.setPixelRatio(window.devicePixelRatio || 1);
    renderer3d.setSize(host.clientWidth, host.clientHeight);
    host.appendChild(renderer3d.domElement);

    controls3d = new THREE.OrbitControls(cam3d, renderer3d.domElement);
    controls3d.target.set(0, 0, 12);
    controls3d.update();

    scene3d.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dl = new THREE.DirectionalLight(0xffffff, 0.6); dl.position.set(10, 20, 6); scene3d.add(dl);
    const grid = new THREE.GridHelper(160, 32, 0x30363d, 0x1c2128);
    scene3d.add(grid);                                    // sol = plan XZ
    scene3d.add(new THREE.AxesHelper(4));

    // Rig caméra : cône + frustum regroupés pour pivoter ensemble (pan/tilt des servos).
    camRig = new THREE.Group(); scene3d.add(camRig);

    // Caméra système : cône bleu pointant vers +Z (axe optique du rig).
    const coneGeo = new THREE.ConeGeometry(2, 5, 28);
    coneGeo.rotateX(Math.PI / 2);                         // apex vers +Z
    coneGeo.translate(0, 0, 2.5);
    const cone = new THREE.Mesh(coneGeo,
      new THREE.MeshStandardMaterial({ color: 0x58a6ff }));
    camRig.add(cone);

    fovGroup = new THREE.Group(); camRig.add(fovGroup);   // frustum de champ, construit au 1er paquet
    window.addEventListener("resize", resize3D);
    animate3D();
  } catch (err) { scene3d = null; console.warn("Vue 3D indisponible :", err); }
}

// Frustum filaire représentant le champ de vision, reconstruit seulement si W/H/HFOV change.
function buildFrustum(W, H, hfovDeg) {
  const key = W + "x" + H + "@" + hfovDeg;
  if (key === camKey || !fovGroup) return;
  camKey = key;
  while (fovGroup.children.length) {
    const c = fovGroup.children[0]; fovGroup.remove(c); c.geometry.dispose(); c.material.dispose();
  }
  const focal = (W / 2) / Math.tan(hfovDeg * Math.PI / 180 / 2);
  const D = 18;                                           // profondeur d'affichage du plan image (m)
  const hw = D * (W / 2) / focal, hh = D * (H / 2) / focal;
  const O = new THREE.Vector3(0, 0, 0);
  const c = [new THREE.Vector3(-hw, hh, D), new THREE.Vector3(hw, hh, D),
             new THREE.Vector3(hw, -hh, D), new THREE.Vector3(-hw, -hh, D)];
  const pts = [O, c[0], O, c[1], O, c[2], O, c[3],
               c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]];
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  fovGroup.add(new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({ color: 0x58a6ff, transparent: true, opacity: 0.45 })));
}

// Étiquette texte (sprite) accrochée à une cible ; régénérée seulement si le texte change.
function makeLabel(text) {
  const cv = document.createElement("canvas"), ctx = cv.getContext("2d");
  ctx.font = "26px monospace";
  cv.width = Math.ceil(ctx.measureText(text).width) + 16; cv.height = 34;
  ctx.font = "26px monospace";
  ctx.fillStyle = "rgba(13,17,23,0.78)"; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.fillText(text, 8, cv.height / 2);
  const spr = new THREE.Sprite(new THREE.SpriteMaterial(
    { map: new THREE.CanvasTexture(cv), depthTest: false }));
  spr.scale.set(cv.width / 34 * 1.6, 1.6, 1); spr.position.set(0, 1.8, 0);
  return spr;
}

function render3D(d) {
  if (!scene3d || !d.cam) return;
  const W = d.cam.w, H = d.cam.h, hfov = d.cam.hfov_deg;
  buildFrustum(W, H, hfov);
  const focal = (W / 2) / Math.tan(hfov * Math.PI / 180 / 2);

  // Orientation pan/tilt (servos / simu) : on pivote le rig caméra ET on applique la même
  // rotation aux cibles, qui sont calculées dans le repère caméra. Ainsi, quand la caméra
  // tourne pour suivre la cible verrouillée, le cône reste pointé sur elle (cohérence monde).
  const pt = d.pantilt || { pan_cam: 0, tilt_cam: 0 };
  const panRad = pt.pan_cam * Math.PI / 180, tiltRad = pt.tilt_cam * Math.PI / 180;
  // +pan_cam = caméra vers la droite (yaw +Y) ; +tilt_cam = vers le haut (pitch -X).
  camEuler = new THREE.Euler(-tiltRad, panRad, 0, "YXZ");
  if (camRig) camRig.rotation.copy(camEuler);

  const seen = new Set();
  for (const o of d.objects) {
    if (o.distance == null || !o.bbox) continue;          // sans distance/bbox : pas de position 3D
    seen.add(o.id);
    const cx = (o.bbox[0] + o.bbox[2]) / 2, cy = (o.bbox[1] + o.bbox[3]) / 2;
    const u = (cx - W / 2) / focal, v = (cy - H / 2) / focal;
    // La caméra de visualisation regarde +Z depuis l'arrière du cône : son axe
    // "droite écran" est -X monde. On prend donc -u (sinon un objet à droite de
    // l'image s'afficherait à gauche). -v : haut de l'image -> +Y (haut écran), OK.
    const pos = new THREE.Vector3(-u, -v, 1).normalize().multiplyScalar(o.distance);
    pos.applyEuler(camEuler);                             // dans le repère monde (caméra orientée)
    let m = meshes.get(o.id);
    if (!m) {
      m = new THREE.Mesh(new THREE.SphereGeometry(0.9, 16, 16),
        new THREE.MeshStandardMaterial({ color: colorFor(o.class) }));
      scene3d.add(m); meshes.set(o.id, m);
    }
    m.material.color.set(colorFor(o.class));
    // Cible verrouillée : sphère qui "brille" (emissive) et légèrement agrandie.
    const locked = d.target_id != null && o.id === d.target_id;
    m.material.emissive.set(locked ? colorFor(o.class) : 0x000000);
    m.material.emissiveIntensity = locked ? 0.9 : 0.0;
    const s = locked ? 1.35 : 1.0; m.scale.set(s, s, s);
    m.position.copy(pos);
    const txt = "#" + o.id + " " + o.class + " " + o.distance.toFixed(0) + "m";
    if (m.userData.txt !== txt) {
      if (m.userData.label) {
        m.remove(m.userData.label);
        m.userData.label.material.map.dispose(); m.userData.label.material.dispose();
      }
      const lbl = makeLabel(txt); m.add(lbl); m.userData.label = lbl; m.userData.txt = txt;
    }
  }
  for (const [id, m] of [...meshes]) {                     // retire les cibles disparues
    if (!seen.has(id)) {
      scene3d.remove(m); m.geometry.dispose(); m.material.dispose();
      if (m.userData.label) { m.userData.label.material.map.dispose(); m.userData.label.material.dispose(); }
      meshes.delete(id);
    }
  }
}

function animate3D() {
  requestAnimationFrame(animate3D);
  controls3d.update();
  renderer3d.render(scene3d, cam3d);
}
function resize3D() {
  const host = document.getElementById("scene3d");
  if (!scene3d || !host.clientWidth) return;
  cam3d.aspect = host.clientWidth / host.clientHeight; cam3d.updateProjectionMatrix();
  renderer3d.setSize(host.clientWidth, host.clientHeight);
}

init3D();

const es = new EventSource("/events");
es.onmessage = e => { const d = JSON.parse(e.data); render(d); renderFocus(d.focus); render3D(d); };
</script>
</body></html>"""

# Carte de contrôle du focus, insérée seulement quand une lentille motorisée
# (C3 18X) est présente. Bouton autofocus one-shot + réglage manuel fin +/-.
_FOCUS_CARD = """<div class="card"><h2>Optique (C3 18X)</h2>
        <div class="focus-btns">
          <button onclick="af()">Autofocus</button>
          <button onclick="nudge(-FOCUS_STEP)">− net</button>
          <button onclick="nudge(FOCUS_STEP)">+ net</button>
        </div>
        <div class="focus-btns">
          <button onclick="zoom(ZOOM_STEP)">− dézoom</button>
          <button onclick="zoom(-ZOOM_STEP)">+ zoom</button>
          <button onclick="dezoomMax()">grand-angle max</button>
        </div>
        <div id="focus-state" class="focus-state">—</div>
      </div>"""


class Dashboard:
    """Sert le tableau de bord unique + le flux MJPEG + le flux SSE de données."""

    def __init__(self, config: dict, on_autofocus=None, on_nudge=None,
                 on_zoom=None, on_dezoom_max=None,
                 focus_enabled: bool = False, focus_step: int = 50, zoom_step: int = 2000) -> None:
        self.host = config.get("mjpeg_host", "0.0.0.0")
        self.port = config.get("mjpeg_port", 5000)
        self._frame: bytes | None = None      # dernier JPEG encodé
        self._seq = 0                          # incrémenté à chaque nouvelle frame
        self._data: dict = {"fps": 0.0, "timestamp": "", "objects": [], "cam": None}
        self._lock = threading.Lock()
        self._placeholder = self._make_placeholder()

        # Contrôle de l'optique (caméra C3 18X). Callbacks branchés sur la caméra.
        self._on_autofocus = on_autofocus
        self._on_nudge = on_nudge
        self._on_zoom = on_zoom
        self._on_dezoom_max = on_dezoom_max
        self._focus_enabled = focus_enabled
        self._focus_step = int(focus_step)
        self._zoom_step = int(zoom_step)

        # static_folder absolu (output/static/) : sert three.min.js + OrbitControls.js
        # vendorés → la vue 3D marche hors-ligne, quel que soit le dossier de lancement.
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        self.app = Flask(__name__, static_folder=static_dir, static_url_path="/static")
        self.app.add_url_rule("/", "index", self._index)
        self.app.add_url_rule("/video", "video", self._video)
        self.app.add_url_rule("/events", "events", self._events)
        self.app.add_url_rule("/focus/auto", "focus_auto", self._focus_auto, methods=["POST"])
        self.app.add_url_rule("/focus/nudge", "focus_nudge", self._focus_nudge, methods=["POST"])
        self.app.add_url_rule("/zoom/nudge", "zoom_nudge", self._zoom_nudge, methods=["POST"])
        self.app.add_url_rule("/zoom/wide", "zoom_wide", self._zoom_wide, methods=["POST"])
        self._thread: threading.Thread | None = None

    # -- API publique ------------------------------------------------------

    def start(self) -> None:
        """Lance le serveur Flask dans un thread démon (logs d'accès silencieux)."""
        logging.getLogger("werkzeug").setLevel(logging.ERROR)  # coupe le spam par requête
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        logger.info("Tableau de bord : http://{}:{}/", self.host, self.port)

    def update(self, frame: np.ndarray, data: dict) -> None:
        """Publie la frame annotée + les données (stats/objets), thread-safe."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._lock:
            self._frame = buf.tobytes()
            self._data = data
            self._seq += 1

    # -- routes ------------------------------------------------------------

    def _serve(self) -> None:
        self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

    def _index(self) -> str:
        """Page principale ; insère la carte Optique seulement si la C3 est pilotable."""
        card = _FOCUS_CARD if self._focus_enabled else ""
        return (_PAGE.replace("<!--FOCUS_CARD-->", card)
                .replace("__FOCUS_STEP__", str(self._focus_step))
                .replace("__ZOOM_STEP__", str(self._zoom_step)))

    def _focus_auto(self):
        """Déclenche un autofocus one-shot (non bloquant)."""
        ok = bool(self._on_autofocus and self._on_autofocus())
        return {"ok": ok}

    def _focus_nudge(self):
        """Ajuste le focus de ±delta pas (réglage manuel fin)."""
        try:
            delta = int(request.args.get("delta", self._focus_step))
        except (TypeError, ValueError):
            return {"ok": False, "error": "delta invalide"}, 400
        ok = bool(self._on_nudge and self._on_nudge(delta))
        return {"ok": ok}

    def _zoom_nudge(self):
        """Ajuste le zoom de ±delta pas (delta<0 = dézoome / grand-angle)."""
        try:
            delta = int(request.args.get("delta", self._zoom_step))
        except (TypeError, ValueError):
            return {"ok": False, "error": "delta invalide"}, 400
        ok = bool(self._on_zoom and self._on_zoom(delta))
        return {"ok": ok}

    def _zoom_wide(self):
        """Va au grand-angle maximum (bouton de secours)."""
        ok = bool(self._on_dezoom_max and self._on_dezoom_max())
        return {"ok": ok}

    def _video(self) -> Response:
        return Response(self._mjpeg_generator(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    def _mjpeg_generator(self):
        # Tourne dans le thread Flask : DOIT dormir à chaque tour, sinon le busy-loop
        # accapare le GIL et affame le pipeline. On n'émet que les frames nouvelles.
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last_seq = None
        while True:
            with self._lock:
                frame, seq = self._frame, self._seq
            if frame is None:
                frame, seq = self._placeholder, "placeholder"
            if seq != last_seq:
                last_seq = seq
                yield boundary + frame + b"\r\n"
            time.sleep(0.03)

    def _events(self) -> Response:
        """Flux SSE : pousse les données quand elles changent (≈4 Hz), sans polling."""
        def gen():
            last_seq = None
            while True:
                with self._lock:
                    data, seq = self._data, self._seq
                if seq != last_seq:
                    last_seq = seq
                    yield f"data: {json.dumps(data)}\n\n"
                time.sleep(0.25)
        return Response(gen(), mimetype="text/event-stream")

    @staticmethod
    def _make_placeholder() -> bytes:
        """JPEG gris affiché tant qu'aucune frame n'est arrivée."""
        img = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.putText(img, "En attente de la camera...", (70, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()


if __name__ == "__main__":
    # Test standalone : mire animée + données factices. Ouvrir http://localhost:5000/
    server = Dashboard({"mjpeg_host": "0.0.0.0", "mjpeg_port": 5000})
    server.start()
    print("Tableau de bord sur http://localhost:5000/ (Ctrl+C pour arrêter)")
    import math
    x = 0
    while True:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (x % 640, 240), 30, (0, 255, 0), -1)
        # Pan/tilt factice qui oscille : le cône caméra doit pivoter dans la vue 3D.
        pan = 30.0 * math.sin(x / 60.0)
        server.update(img, {"fps": 30.0, "timestamp": "demo",
                            "cam": {"w": 640, "h": 480, "hfov_deg": 66.0},
                            "pantilt": {"driver": "sim", "pan_cam": round(pan, 1), "tilt_cam": 8.0},
                            "target_id": 1,
                            "objects": [{"id": 1, "class": "drone", "conf": 0.9, "distance": 12.0,
                                         "bbox": [x % 640 - 30, 210, x % 640 + 30, 270]}]})
        x += 8
        time.sleep(0.05)
