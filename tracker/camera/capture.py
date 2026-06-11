"""Capture vidéo : PiCam (picamera2), webcam USB, ou Kurokesu C3 18X.

`picamera2` est importé en différé (lazy) : le module reste importable sur une
machine de dev (Windows / sans libcamera), où l'on bascule sur OpenCV.
Toutes les frames renvoyées sont en BGR (`np.ndarray`), prêtes pour OpenCV/YOLO.

Le type de source se choisit via `camera.type` (picamera | webcam | c3). En
l'absence de `type`, on retombe sur l'ancien `camera.use_picamera` (rétro-compat).

La **C3 18X** est une caméra UVC (lue comme une webcam) PLUS une lentille
motorisée pilotée par le contrôleur SCF4 (port série). Pour ce backend, la vidéo
est lue par un thread dédié dans un cache thread-safe (un seul lecteur de la
`VideoCapture`), ce qui permet à l'autofocus de mesurer la netteté sur les mêmes
frames sans entrer en concurrence avec la boucle de détection. Si le SCF4 est
absent, la vidéo continue : seul le focus est désactivé.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np
from loguru import logger

from camera import scf4
from camera.focus import LensController


class CameraError(RuntimeError):
    """Levée quand aucune source vidéo n'a pu être ouverte."""


class CameraCapture:
    """Source vidéo unifiée. Essaie la C3 18X / PiCam, sinon une webcam USB."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.width, self.height = config["resolution"]
        self.fps = config.get("fps", 30)
        self._picam = None            # instance Picamera2 si active
        self._cap: cv2.VideoCapture | None = None  # VideoCapture si webcam/c3
        self._backend = "none"

        # Backend "c3" : thread de capture + cache (un seul lecteur de la VideoCapture).
        self._frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._run = False

        # Focus + zoom motorisés (uniquement backend c3, si le SCF4 répond).
        self._focus_cfg = config.get("focus", {}) or {}
        self._zoom_cfg = config.get("zoom", {}) or {}
        self.manual_step = int(self._focus_cfg.get("manual_step", 50))
        self.zoom_step = int(self._zoom_cfg.get("manual_step", 2000))
        self._lens: LensController | None = None

    # -- cycle de vie ------------------------------------------------------

    def start(self) -> None:
        """Ouvre la source vidéo selon `camera.type` (avec repli sur webcam)."""
        cam_type = self._resolve_type()

        if cam_type == "c3" and self._try_start_c3():
            self._backend = "c3"
            logger.info("Caméra : Kurokesu C3 18X (UVC) active en {}x{}", self.width, self.height)
            self._start_lens()  # best-effort : la vidéo marche même sans SCF4
            return
        if cam_type == "c3":
            logger.warning("C3 18X introuvable (index {}), bascule sur webcam.", self.cfg.get("source", 0))

        if cam_type == "picamera" and self._try_start_picamera():
            self._backend = "picamera2"
            logger.info("Caméra : PiCam (picamera2) active en {}x{}", self.width, self.height)
            return

        if self._try_start_webcam():
            self._backend = "webcam"
            logger.info("Caméra : webcam OpenCV (index {}) active", self.cfg.get("source", 0))
            return

        raise CameraError("Aucune caméra disponible (ni C3 18X, ni PiCam, ni webcam).")

    def _resolve_type(self) -> str:
        """picamera | webcam | c3. Rétro-compat : sans `type`, `use_picamera` décide."""
        t = str(self.cfg.get("type", "") or "").strip().lower()
        if t in ("c3", "c3_18x", "kurokesu"):
            return "c3"
        if t in ("picamera", "picam"):
            return "picamera"
        if t in ("webcam", "usb", "opencv"):
            return "webcam"
        return "picamera" if self.cfg.get("use_picamera", True) else "webcam"

    def _try_start_picamera(self) -> bool:
        """Tente d'initialiser la PiCam ; renvoie False si indisponible."""
        try:
            from picamera2 import Picamera2  # import différé : peut être absent en dev
        except Exception as exc:  # noqa: BLE001 - on veut attraper tout échec d'import
            logger.warning("picamera2 indisponible ({}), bascule sur webcam.", exc)
            return False
        try:
            cam = Picamera2()
            cfg = cam.create_video_configuration(
                main={"size": (self.width, self.height), "format": self.cfg.get("format", "RGB888")},
                controls={"FrameRate": float(self.fps)},
            )
            cam.configure(cfg)
            cam.start()
            self._picam = cam
            return True
        except Exception as exc:  # noqa: BLE001 - caméra absente, occupée, permissions...
            logger.warning("Échec d'initialisation PiCam ({}), bascule sur webcam.", exc)
            return False

    def _try_start_webcam(self) -> bool:
        """Tente d'ouvrir une webcam USB via OpenCV."""
        # Sous Windows, DSHOW s'ouvre bien plus vite que le backend MSMF par défaut.
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cfg.get("source", 0), backend)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # pas de file d'anciennes frames -> moins de latence
        self._cap = cap
        return True

    def _try_start_c3(self) -> bool:
        """Ouvre le flux UVC de la C3 18X + lance le thread de capture (cache)."""
        if sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION
        elif sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cfg.get("source", 0), backend)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Buffer à 1 : le pilote ne garde pas une file d'anciennes frames. Sans ça, quand le
        # décodage MJPG est plus lent que la caméra, la latence s'accumule (vidéo en retard +
        # à-coups). Le thread de capture vide en continu, on veut donc toujours la + récente.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._run = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        return True

    def _capture_loop(self) -> None:
        """Lit la VideoCapture en continu dans le cache (unique lecteur)."""
        while self._run and self._cap is not None:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._frame_lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

    def _start_lens(self) -> None:
        """Ouvre le contrôleur SCF4 et arme l'autofocus. Best-effort : si absent,
        on log et le focus reste désactivé (la vidéo continue de tourner)."""
        try:
            dev = scf4.SCF4(port=self._focus_cfg.get("port", "auto"))
            lens = LensController(dev, self._focus_cfg, self.get_frame, self._zoom_cfg)
            lens.start()
            self._lens = lens
            logger.info("Focus C3 18X activé (contrôleur SCF4 sur {}).", dev.port)
        except Exception as exc:  # noqa: BLE001 - SCF4 absent / occupé / homing KO
            logger.warning("Contrôleur SCF4 indisponible ({}) : focus désactivé.", exc)
            self._lens = None

    # -- acquisition -------------------------------------------------------

    def get_frame(self) -> np.ndarray | None:
        """Renvoie la frame courante en BGR, ou None si lecture impossible."""
        if self._backend == "c3":
            with self._frame_lock:
                return None if self._frame is None else self._frame.copy()
        if self._backend == "picamera2" and self._picam is not None:
            # picamera2 "RGB888" renvoie déjà un buffer ordonné BGR (quirk libcamera).
            return self._picam.capture_array()
        if self._backend == "webcam" and self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None
        return None

    def stop(self) -> None:
        """Libère proprement la source vidéo (et le contrôleur de focus)."""
        self._run = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2)
            self._capture_thread = None
        try:
            if self._lens is not None:
                self._lens.close()
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
            if self._cap is not None:
                self._cap.release()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Erreur à la fermeture de la caméra : {}", exc)
        finally:
            self._lens = None
            self._picam = None
            self._cap = None
            self._backend = "none"

    # -- focus (no-op si pas de lentille motorisée) ------------------------

    @property
    def focus_enabled(self) -> bool:
        return self._lens is not None

    def autofocus(self) -> bool:
        """Déclenche un autofocus one-shot (non bloquant). False si indisponible."""
        return self._lens.autofocus() if self._lens is not None else False

    def adjust_focus(self, delta: int) -> bool:
        """Décale le focus de `delta` pas. False si indisponible."""
        return self._lens.adjust_focus(delta) if self._lens is not None else False

    def adjust_zoom(self, delta: int) -> bool:
        """Décale le zoom de `delta` pas (delta<0 = dézoome). False si indisponible."""
        return self._lens.adjust_zoom(delta) if self._lens is not None else False

    def dezoom_max(self) -> bool:
        """Va au grand-angle max (bouton de secours). False si indisponible."""
        return self._lens.dezoom_max() if self._lens is not None else False

    def focus_state(self) -> dict:
        if self._lens is None:
            return {"enabled": False, "status": "idle", "position": None, "sharpness": 0.0}
        return self._lens.state()

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()


if __name__ == "__main__":
    # Test standalone : affiche 60 frames depuis la source disponible.
    cam = CameraCapture({"use_picamera": True, "source": 0, "resolution": [640, 480], "fps": 30})
    cam.start()
    try:
        for i in range(60):
            frame = cam.get_frame()
            if frame is None:
                print(f"frame {i}: lecture échouée")
                continue
            print(f"frame {i}: {frame.shape} dtype={frame.dtype}")
    finally:
        cam.stop()
