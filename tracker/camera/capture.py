"""Capture vidéo : PiCam (picamera2) avec repli automatique sur webcam USB.

`picamera2` est importé en différé (lazy) : le module reste importable sur une
machine de dev (Windows / sans libcamera), où l'on bascule sur OpenCV.
Toutes les frames renvoyées sont en BGR (`np.ndarray`), prêtes pour OpenCV/YOLO.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
from loguru import logger


class CameraError(RuntimeError):
    """Levée quand aucune source vidéo n'a pu être ouverte."""


class CameraCapture:
    """Source vidéo unifiée. Essaie la PiCam, sinon une webcam USB."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.width, self.height = config["resolution"]
        self.fps = config.get("fps", 30)
        self._picam = None            # instance Picamera2 si active
        self._cap: cv2.VideoCapture | None = None  # VideoCapture si fallback
        self._backend = "none"

    # -- cycle de vie ------------------------------------------------------

    def start(self) -> None:
        """Ouvre la source vidéo (PiCam en priorité, puis webcam)."""
        if self.cfg.get("use_picamera", True) and self._try_start_picamera():
            self._backend = "picamera2"
            logger.info("Caméra : PiCam (picamera2) active en {}x{}", self.width, self.height)
            return

        if self._try_start_webcam():
            self._backend = "webcam"
            logger.info("Caméra : webcam OpenCV (index {}) active", self.cfg.get("source", 0))
            return

        raise CameraError("Aucune caméra disponible (ni PiCam, ni webcam).")

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
        self._cap = cap
        return True

    # -- acquisition -------------------------------------------------------

    def get_frame(self) -> np.ndarray | None:
        """Renvoie la frame courante en BGR, ou None si lecture impossible."""
        if self._backend == "picamera2" and self._picam is not None:
            # picamera2 "RGB888" renvoie déjà un buffer ordonné BGR (quirk libcamera).
            return self._picam.capture_array()
        if self._backend == "webcam" and self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None
        return None

    def stop(self) -> None:
        """Libère proprement la source vidéo."""
        try:
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
            if self._cap is not None:
                self._cap.release()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Erreur à la fermeture de la caméra : {}", exc)
        finally:
            self._picam = None
            self._cap = None
            self._backend = "none"

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
