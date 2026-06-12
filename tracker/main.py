"""Point d'entrée : capture → detect+track → distance → overlay → dashboard + log.

Usage : python main.py [--config config.yaml]

Arrêt propre sur Ctrl+C / SIGTERM. Le redémarrage automatique en cas de crash est délégué
à systemd (`Restart=on-failure`, cf. setup.sh) pour garder ce code simple.
"""

from __future__ import annotations

import argparse
import os
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
from camera.pantilt import PanTiltController
from detection.detector import ObjectDetector
from detection.distance import DistanceEstimator
from output.logger import DetectionLogger
from output.overlay import Overlay
from output.stream import Dashboard
from tracking.target import TargetSelector
from tracking.tracker import ObjectTracker
from utils.fps_counter import FPSCounter

_stop = threading.Event()


def _handle_signal(signum, _frame) -> None:
    logger.info("Signal {} reçu : arrêt...", signum)
    _stop.set()


def _tune_cpu_threads() -> None:
    """Utilise tous les cœurs du Pi pour l'inférence CPU (chemin .pt PyTorch).

    Sous NCNN c'est déjà automatique ; sans effet néfaste si torch est absent."""
    n = os.cpu_count() or 4
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:  # noqa: BLE001 - torch absent / API différente
        pass


def run(config: dict, model_name: str | None = None) -> None:
    """Assemble les modules et traite le flux vidéo jusqu'à l'arrêt."""
    _tune_cpu_threads()
    out = config["output"]
    det_cfg = config["detection"]

    # Choix du modèle : --model (CLI) > detection.model (config).
    name = model_name or det_cfg.get("model", "drone")
    if name not in det_cfg["models"]:
        raise ValueError(f"Modèle '{name}' inconnu (dispo : {list(det_cfg['models'])}).")
    mcfg = det_cfg["models"][name]
    # Poids absents : message actionnable plutôt qu'un stack trace ultralytics.
    # ("base" est exclu : yolov8n.pt est auto-téléchargé par ultralytics.)
    weights = Path(mcfg["path"])
    ncnn_dir = weights.with_name(weights.stem + "_ncnn_model")
    if name != "base" and not weights.exists() and not ncnn_dir.is_dir():
        raise SystemExit(
            f"Poids absents pour le modèle '{name}' : {weights}\n"
            "  1. Entraînez-le sur une machine GPU : voir training/README.md\n"
            f"  2. Copiez le fichier obtenu vers {weights}\n"
            "  En attendant : python main.py --model base"
        )
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
    # Suivi de cible : on verrouille la classe prioritaire (person en base, drone en drone)
    # et on oriente la caméra (servos sur Pi, simulé en dev) pour la garder au centre.
    selector = TargetSelector(priority)
    pantilt = PanTiltController(config.get("pantilt", {}))
    pantilt.start()
    overlay = Overlay(out)
    det_logger = DetectionLogger(out)
    # Contrôles de focus (caméra C3 18X) : boutons autofocus + réglage manuel,
    # branchés sur la caméra. No-op / boutons masqués pour les autres sources.
    focus_cfg = config["camera"].get("focus", {}) or {}
    dashboard = Dashboard(out, on_autofocus=cam.autofocus, on_nudge=cam.adjust_focus,
                          on_zoom=cam.adjust_zoom, on_dezoom_max=cam.dezoom_max,
                          on_set_focus=cam.set_focus,
                          focus_enabled=cam.focus_enabled, focus_step=cam.manual_step,
                          zoom_step=cam.zoom_step,
                          focus_min=int(focus_cfg.get("focus_min", 0)),
                          focus_max=int(focus_cfg.get("focus_max", 0)))
    dashboard.start()
    fps = FPSCounter()

    # Mise au point initiale (one-shot, non bloquante). Ensuite on n'y revient
    # plus, sauf clic sur le bouton du dashboard.
    if cam.focus_enabled and config["camera"].get("focus", {}).get("autofocus_on_start", True):
        cam.autofocus()

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

            h, w = frame.shape[:2]
            # Cible verrouillée -> on oriente la caméra pour la centrer (servos prod / simu dev).
            # Aucune cible visible : on laisse la caméra sur sa dernière orientation.
            target = selector.select(objects)
            if target is not None:
                pantilt.track(target.detection.bbox, w, h, distance.hfov_deg)

            timestamp = datetime.now().isoformat(timespec="seconds")
            overlay.draw(frame, objects, fps.fps, timestamp, priority)
            dashboard.update(frame, {
                "fps": round(fps.fps, 1),
                "timestamp": timestamp.split("T")[-1],
                # bbox + intrinsèques caméra : la vue 3D du dashboard projette chaque cible
                # à sa distance estimée, à l'azimut/élévation déduits du centre du bbox et du FOV.
                "cam": {"w": w, "h": h, "hfov_deg": distance.hfov_deg},
                "objects": [{"id": o.track_id, "class": o.detection.class_name,
                             "conf": round(o.detection.confidence, 3),
                             "distance": o.distance_m,
                             "bbox": list(o.detection.bbox)} for o in objects],
                "focus": cam.focus_state(),
                # orientation caméra (pan/tilt) + id de la cible suivie : la vue 3D fait
                # pivoter le cône caméra et met la cible en évidence.
                "pantilt": pantilt.state(),
                "target_id": selector.current_id,
            })
            det_logger.log(objects, timestamp)
            fps.tick()
    finally:
        cam.stop()
        pantilt.close()
        det_logger.close()
        logger.info("Arrêté proprement.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Détection & suivi de drones embarqué.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"),
                        help="Chemin du fichier de configuration YAML.")
    parser.add_argument("--model", choices=["drone", "base"], default=None,
                        help="Modèle à utiliser (surcharge detection.model du config).")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    with open(args.config, encoding="utf-8") as f:
        run(yaml.safe_load(f), model_name=args.model)


if __name__ == "__main__":
    main()
