"""Caméra Kurokesu C3 18X — affichage vidéo net + focus (script unique).

Architecture (celle qui marchait, prouvée par l'ancien focus_scan) :
- UN thread de capture qui lit la caméra EN CONTINU -> la dernière frame est
  toujours à jour, le buffer ne se remplit jamais de frames périmées ;
- tout le reste (autofocus, affichage, clavier) est SYNCHRONE dans le thread
  principal.

La netteté (variance du Laplacien sur la zone centrale) est mesurée en prenant
le MAX sur une courte fenêtre après l'arrêt du moteur : les frames de transition
sont plus floues, donc le max correspond à l'image stabilisée — robuste à la
latence du pipeline caméra.

Zoom fixe (dézoomé au max) : on ne pilote que le focus (axe B du SCF4).

Usage :
    python main.py --list-cams
    python main.py --cam 1
    python main.py --cam 1 --scan        # diagnostic : courbe netteté vs position

Touches : a/espace = autofocus, +/- (ou ./,) = focus manuel, h = homing, q = quitter.
"""

import argparse
import platform
import threading
import time

import cv2

import scf4

# --- Réglages (zone de focus mesurée avec --scan) ---
FOCUS_MIN, FOCUS_MAX = 2000, 15000
COARSE_STEP = 300
FINE_STEP = 30
FINE_WIN = 400
MANUAL_STEP = 50
ROI_RATIO = 0.40
FOCUS_SPEED = 600
SETTLE_SEC = 0.15        # fenêtre de mesure après l'arrêt (max de netteté ; ~ celle du focus_scan qui marchait)
SCALE = 0.5

# Jeu mécanique : on arrive toujours sur la position finale par le même sens
# (le bas, +1) pour absorber le backlash et rendre la mise au point répétable.
BACKLASH_STEPS = 20
APPROACH_DIR = 1
# Arrêt anticipé de la passe grossière : si la netteté retombe à moins de
# (1 - ratio) du pic vu, le pic est passé -> inutile de balayer toute la plage.
EARLY_STOP_RATIO = 0.30


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _backend():
    return cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


def list_cameras(max_index=8):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _backend())
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((i, w, h))
        cap.release()
    return found


class Camera:
    """Capture vidéo : un thread lit la caméra en continu (buffer toujours vidé)."""

    def __init__(self, index, width=1920, height=1080, fps=30):
        self.cap = cv2.VideoCapture(index, _backend())
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra index {index}.")
        self._frame = None
        self._lock = threading.Lock()
        self._run = False
        self._t = None

    def start(self):
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if ok and f is not None:
                with self._lock:
                    self._frame = f

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_ready(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest() is not None:
                return True
            time.sleep(0.05)
        raise RuntimeError("Pas d'image de la caméra (essaie un autre --cam).")

    def stop(self):
        self._run = False
        if self._t:
            self._t.join(timeout=2)
        self.cap.release()


def roi_rect(shape):
    h, w = shape[:2]
    rw, rh = int(w * ROI_RATIO), int(h * ROI_RATIO)
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    return x0, y0, x0 + rw, y0 + rh


def sharpness(frame):
    x0, y0, x1, y1 = roi_rect(frame.shape)
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return float(cv2.Laplacian(blurred, cv2.CV_64F).var())


def move_and_measure(cam, dev, pos, window=None, settle=SETTLE_SEC, backlash=False):
    """Va à `pos`, attend l'arrêt, puis renvoie (frame, MAX netteté) sur une
    courte fenêtre (les frames de transition sont plus floues -> le max = image nette).

    backlash=True : arrive sur `pos` via move_to_backlash (placement final répétable).
    Les balayages, eux, montent en pas réguliers donc abordent déjà chaque point
    par le même sens — pas besoin de compensation."""
    pos = clamp(int(pos), FOCUS_MIN, FOCUS_MAX)
    if backlash:
        dev.move_to_backlash("B", pos, BACKLASH_STEPS, APPROACH_DIR, wait=True)
    else:
        dev.move_abs("B", pos, wait=True)      # wait_stop exige 2 statuts 'arrêté' consécutifs

    best, frame = 0.0, None
    t0 = time.time()
    while time.time() - t0 < settle:
        f = cam.latest()
        if f is not None:
            s = sharpness(f)
            if s > best:
                best, frame = s, f
        time.sleep(0.02)

    if window is not None and frame is not None:
        _show(window, frame, f"Autofocus... pos={pos} net={best:.0f}")
        cv2.waitKey(1)
    return frame, best


def _scan_range(cam, dev, lo, hi, step, window=None, early_stop_ratio=None):
    """Balayage discret lo->hi. Si early_stop_ratio est fourni, on s'arrête dès
    que la netteté retombe sous (1 - ratio) du pic vu (le pic est derrière nous)."""
    table = []
    peak = 0.0
    for pos in range(int(lo), int(hi) + 1, int(step)):
        _, s = move_and_measure(cam, dev, pos, window)
        act = dev.position("B")                 # position RÉELLE lue sur le moteur
        print(f"  scan cmd={pos:>6} act={act:>6} net={s:6.1f}")
        table.append((pos, s))
        if early_stop_ratio is not None:
            peak = max(peak, s)
            if peak > 0 and len(table) >= 3 and s < peak * (1 - early_stop_ratio):
                print(f"  arrêt anticipé : net {s:.0f} < {(1 - early_stop_ratio) * 100:.0f}% du pic {peak:.0f}")
                break
    return table


def _parabolic_peak(table):
    i = max(range(len(table)), key=lambda k: table[k][1])
    pos = table[i][0]
    if 0 < i < len(table) - 1:
        (x0, y0), (x1, y1), (x2, y2) = table[i - 1], table[i], table[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            pos = x1 + 0.5 * (y0 - y2) / denom * (x1 - x0)
    return clamp(int(pos), FOCUS_MIN, FOCUS_MAX)


def autofocus(cam, dev, window=None):
    """Autofocus one-shot synchrone : passe grossière puis fine, va au pic."""
    t0 = time.time()
    dev.set_speed("B", FOCUS_SPEED)

    coarse = _scan_range(cam, dev, FOCUS_MIN, FOCUS_MAX, COARSE_STEP, window,
                         early_stop_ratio=EARLY_STOP_RATIO)
    peak_c, _ = max(coarse, key=lambda t: t[1])

    lo, hi = clamp(peak_c - FINE_WIN, FOCUS_MIN, FOCUS_MAX), clamp(peak_c + FINE_WIN, FOCUS_MIN, FOCUS_MAX)
    fine = _scan_range(cam, dev, lo, hi, FINE_STEP, window)

    best = _parabolic_peak(fine)
    _, final = move_and_measure(cam, dev, best, window, backlash=True)
    print(f"Autofocus : pos={best} net={final:.0f} en {time.time() - t0:.1f}s")
    return best, final


def _show(window, frame, text):
    x0, y0, x1, y1 = roi_rect(frame.shape)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(frame, text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow(window, cv2.resize(frame, None, fx=SCALE, fy=SCALE))


def run_scan(cam, dev, lo, hi, step):
    """Diagnostic : courbe netteté vs position (utilise la même mesure que l'AF)."""
    dev.set_speed("B", FOCUS_SPEED)
    table = _scan_range(cam, dev, lo, hi, step)
    smax = max(s for _, s in table) or 1.0
    print(f"\n{'pos':>7}  {'net':>8}  courbe")
    for pos, s in table:
        print(f"{pos:>7}  {s:>8.1f}  {'#' * int(50 * s / smax)}")
    best_pos, best_s = max(table, key=lambda t: t[1])
    print(f"\nPIC : pos={best_pos} net={best_s:.1f}")
    if best_s < 20:
        print("⚠ pic faible (<20) : élargis --min/--max ou vise une cible texturée/éclairée.")


def main():
    p = argparse.ArgumentParser(description="Kurokesu C3 18X : vidéo + focus")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--list-cams", action="store_true")
    p.add_argument("--no-home", action="store_true")
    p.add_argument("--scan", action="store_true", help="diagnostic : courbe netteté vs position")
    p.add_argument("--min", type=int, default=FOCUS_MIN)
    p.add_argument("--max", type=int, default=FOCUS_MAX)
    p.add_argument("--step", type=int, default=100)
    args = p.parse_args()

    if args.list_cams:
        print("Caméras détectées (index : résolution) :")
        for idx, w, h in list_cameras() or []:
            print(f"  --cam {idx}   ({w}x{h})")
        print("\nLa C3 18X sort en 1920x1080 / 1280x720 ; la webcam Mac est souvent l'index 0.")
        return

    print(f"Ouverture caméra (index {args.cam})...")
    cam = Camera(args.cam)
    cam.start()
    cam.wait_ready()
    print("Connexion au contrôleur SCF4...")
    dev = scf4.SCF4(port="auto")
    print("  version :", dev.init_controller())

    if not args.no_home:
        print("Homing focus...")
        dev.home_axis("B")

    try:
        if args.scan:
            run_scan(cam, dev, args.min, args.max, args.step)
            return

        window = "Kurokesu C3 18X"
        cv2.namedWindow(window)
        print("Autofocus initial...")
        focus_pos, _ = autofocus(cam, dev, window)

        last = time.time()
        while True:
            frame = cam.latest()
            if frame is not None:
                now = time.time()
                fps = 1.0 / (now - last) if now > last else 0.0
                last = now
                _show(window, frame,
                      f"focus={focus_pos}  net={sharpness(frame):.0f}  fps={fps:.0f}  "
                      f"[a]AF [j/l]+-500 [+/-]+-50 [h]home [q]quitter")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("a"), ord(" ")):
                focus_pos, _ = autofocus(cam, dev, window)
            elif key in (ord("+"), ord(".")):
                focus_pos = clamp(focus_pos + MANUAL_STEP, FOCUS_MIN, FOCUS_MAX)
                dev.move_abs("B", focus_pos, wait=True)
            elif key in (ord("-"), ord(",")):
                focus_pos = clamp(focus_pos - MANUAL_STEP, FOCUS_MIN, FOCUS_MAX)
                dev.move_abs("B", focus_pos, wait=True)
            elif key == ord("l"):                 # gros pas + (balayage rapide à l'œil)
                focus_pos = clamp(focus_pos + 500, FOCUS_MIN, FOCUS_MAX)
                dev.move_abs("B", focus_pos, wait=True)
            elif key == ord("j"):                 # gros pas -
                focus_pos = clamp(focus_pos - 500, FOCUS_MIN, FOCUS_MAX)
                dev.move_abs("B", focus_pos, wait=True)
            elif key == ord("h"):
                dev.home_axis("B")
                focus_pos, _ = autofocus(cam, dev, window)
    finally:
        print("Arrêt...")
        dev.close()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
