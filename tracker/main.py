"""Point d'entrée : capture → detect+track → distance → overlay → dashboard + log.

Usage : python main.py [--config config.yaml]

Arrêt propre sur Ctrl+C / SIGTERM. Le redémarrage automatique en cas de crash est délégué
à systemd (`Restart=on-failure`, cf. setup.sh) pour garder ce code simple.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

# Permet `python main.py` depuis n'importe quel dossier.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera.capture import CameraCapture
from detection.detector import ObjectDetector
from detection.distance import DistanceEstimator
from output.logger import DetectionLogger
from output.overlay import Overlay
from output.stream import Dashboard
from tracking.tracker import ObjectTracker
from utils.fps_counter import FPSCounter

_stop = threading.Event()


def _handle_signal(signum, _frame) -> None:
    logger.info("Signal {} reçu : arrêt...", signum)
    _stop.set()


def run(config: dict, model_name: str | None = None) -> None:
    """Assemble les modules et traite le flux vidéo jusqu'à l'arrêt."""
    out = config["output"]
    det_cfg = config["detection"]

    # Choix du modèle : --model (CLI) > detection.model (config).
    name = model_name or det_cfg.get("model", "drone")
    if name not in det_cfg["models"]:
        raise ValueError(f"Modèle '{name}' inconnu (dispo : {list(det_cfg['models'])}).")
    mcfg = det_cfg["models"][name]
    priority = mcfg.get("priority_class", "drone")
    # Config "à plat" attendue par ObjectDetector.
    flat_cfg = {
        "model_path": mcfg["path"],
        "confidence_threshold": mcfg.get("confidence", 0.30),
        "iou_threshold": det_cfg.get("iou_threshold", 0.5),
        "target_classes": mcfg.get("classes", []),
        "class_names": mcfg.get("class_names"),     # renommage par index (optionnel)
        "inference_size": mcfg.get("inference_size", 640),
        "device": det_cfg.get("device", "cpu"),
    }
    logger.info("Modèle sélectionné : '{}' ({})", name, mcfg["path"])

    # Ouvre la caméra EN PARALLÈLE du chargement du modèle (le plus lent) -> démarrage + rapide.
    cam = CameraCapture(config["camera"])
    cam_err: dict = {}

    def _start_cam() -> None:
        try:
            cam.start()
        except Exception as exc:  # noqa: BLE001 - remontée après le join
            cam_err["exc"] = exc

    cam_thread = threading.Thread(target=_start_cam)
    cam_thread.start()
    detector = ObjectDetector(flat_cfg)  # charge le modèle pendant l'ouverture caméra
    cam_thread.join()
    if "exc" in cam_err:
        raise cam_err["exc"]

    config["tracking"]["trajectory_length"] = out.get("trajectory_length", 50)
    tracker = ObjectTracker(config["tracking"])
    distance = DistanceEstimator(config.get("distance", {}), config["camera"]["resolution"][0])
    smooth = config.get("distance", {}).get("smoothing", 0.3)
    overlay = Overlay(out)
    det_logger = DetectionLogger(out)
    dashboard = Dashboard(out)
    dashboard.start()
    fps = FPSCounter()

    logger.info("Pipeline démarré (priorité={}).", priority)
    empty = 0
    try:
        while not _stop.is_set():
            frame = cam.get_frame()
            if frame is None:
                empty += 1
                if empty >= 150:
                    raise RuntimeError("Caméra : aucune frame reçue (câble/permission ?).")
                time.sleep(0.01)
                continue
            empty = 0

            objects = tracker.update(detector.detect(frame))

            # Distance estimée, lissée dans le temps par track (EMA) ; l'objet persiste
            # entre les frames donc obj.distance_m sert d'état.
            for obj in objects:
                raw = distance.estimate(obj.detection)
                if raw is None or obj.distance_m is None:
                    obj.distance_m = raw
                else:
                    obj.distance_m = round(smooth * raw + (1 - smooth) * obj.distance_m, 2)

            # Priorité d'affichage : les drones d'abord (dans la table comme à l'écran).
            objects.sort(key=lambda o: o.detection.class_name.lower() != priority.lower())

            timestamp = datetime.now().isoformat(timespec="seconds")
            overlay.draw(frame, objects, fps.fps, timestamp, priority)
            dashboard.update(frame, {
                "fps": round(fps.fps, 1),
                "timestamp": timestamp.split("T")[-1],
                "objects": [{"id": o.track_id, "class": o.detection.class_name,
                             "conf": round(o.detection.confidence, 3),
                             "distance": o.distance_m} for o in objects],
            })
            det_logger.log(objects, timestamp)
            fps.tick()
    finally:
        cam.stop()
        det_logger.close()
        logger.info("Arrêté proprement.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Détection & suivi de drones embarqué.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"),
                        help="Chemin du fichier de configuration YAML.")
    parser.add_argument("--model", choices=["flying", "drone", "base"], default=None,
                        help="Modèle à utiliser (surcharge detection.model du config).")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    with open(args.config, encoding="utf-8") as f:
        run(yaml.safe_load(f), model_name=args.model)


if __name__ == "__main__":
    main()
