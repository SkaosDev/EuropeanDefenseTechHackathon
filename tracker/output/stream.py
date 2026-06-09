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
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response
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
</style></head>
<body>
  <div class="wrap">
    <div class="video"><img src="/video" alt="flux"></div>
    <div class="side">
      <div class="card"><h2>Cibles suivies</h2>
        <table><thead><tr><th>ID</th><th>Classe</th><th>Conf.</th><th>Dist.</th></tr></thead>
        <tbody id="rows"></tbody></table>
      </div>
      <div class="card"><h2>Journal</h2><div id="log"></div></div>
    </div>
  </div>
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

const es = new EventSource("/events");
es.onmessage = e => render(JSON.parse(e.data));
</script>
</body></html>"""


class Dashboard:
    """Sert le tableau de bord unique + le flux MJPEG + le flux SSE de données."""

    def __init__(self, config: dict) -> None:
        self.host = config.get("mjpeg_host", "0.0.0.0")
        self.port = config.get("mjpeg_port", 5000)
        self._frame: bytes | None = None      # dernier JPEG encodé
        self._seq = 0                          # incrémenté à chaque nouvelle frame
        self._data: dict = {"fps": 0.0, "timestamp": "", "objects": []}
        self._lock = threading.Lock()
        self._placeholder = self._make_placeholder()

        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", lambda: _PAGE)
        self.app.add_url_rule("/video", "video", self._video)
        self.app.add_url_rule("/events", "events", self._events)
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
    x = 0
    while True:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (x % 640, 240), 30, (0, 255, 0), -1)
        server.update(img, {"fps": 30.0, "timestamp": "demo",
                            "objects": [{"id": 1, "class": "drone", "conf": 0.9, "distance": 12.0,
                                         "bbox": [x % 640 - 30, 210, x % 640 + 30, 270]}]})
        x += 8
        time.sleep(0.05)
