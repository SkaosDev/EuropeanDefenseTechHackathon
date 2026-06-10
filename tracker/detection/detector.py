"""Wrapper YOLO : chargement + suivi ByteTrack intégré.

Le modèle est chargé une fois puis réchauffé. `track()` renvoie les détections filtrées
(classes cibles) avec identifiants persistants — ByteTrack est géré par ultralytics.

Si un export NCNN existe à côté du .pt (dossier `<nom>_ncnn_model`, généré par setup.sh
sur le Pi), il est chargé en priorité : ~2x plus rapide qu'ONNX sur ARM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from ultralytics import YOLO

from common.types import Detection


def _resolve_model_path(path: str) -> str:
    """Préfère l'export NCNN frère (`<nom>_ncnn_model/`) au .pt s'il existe."""
    pt = Path(path)
    ncnn_dir = pt.with_name(pt.stem + "_ncnn_model")
    if pt.suffix == ".pt" and ncnn_dir.is_dir():
        logger.info("Export NCNN détecté : {} (backend ARM optimisé)", ncnn_dir)
        return str(ncnn_dir)
    return path


class ObjectDetector:
    """Détecteur + tracker YOLOv8n, filtré par classes cibles."""

    def __init__(self, config: dict) -> None:
        self.conf = config["confidence_threshold"]
        self.iou = config["iou_threshold"]
        self.device = config.get("device", "cpu")
        self.imgsz = config.get("inference_size") or 640
        # Filtre de classes ; vide => on accepte toutes les classes du modèle.
        self.target_classes = set(config.get("target_classes") or [])
        # Renommage d'affichage optionnel (ex. "Drone" -> "drone").
        self.class_aliases: dict[str, str] = config.get("class_aliases", {})

        model_path = _resolve_model_path(config["model_path"])
        logger.info("Chargement du modèle YOLO : {}", model_path)
        self.model = YOLO(model_path)
        # Renommage des classes par INDEX (utile si le modèle a des labels exotiques,
        # ex. classes en cyrillique) : class_names = ["drone","avion",...].
        override = config.get("class_names")
        self.names: dict[int, str] = (
            {i: n for i, n in enumerate(override)} if override else self.model.names)
        # Warm-up : première inférence pour absorber le coût d'initialisation.
        self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
                           imgsz=self.imgsz, device=self.device, verbose=False)
        logger.info("Modèle prêt.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Détecte et suit sur une frame BGR -> `Detection` avec `track_id` (ByteTrack)."""
        result = self.model.track(
            frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None:
            return []

        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        detections: list[Detection] = []
        for xyxy, conf, cls_id, track_id in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.int().tolist(), ids
        ):
            name = self.names.get(cls_id, str(cls_id))
            if self.target_classes and name not in self.target_classes:
                continue
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=self.class_aliases.get(name, name),
                track_id=track_id,
            ))
        return detections


if __name__ == "__main__":
    # Test standalone : détecte sur une frame webcam (sinon frame noire).
    import cv2

    cfg = {
        "model_path": "models/yolov8n.pt", "confidence_threshold": 0.35, "iou_threshold": 0.5,
        "target_classes": ["person", "airplane", "kite", "bird"],
        "class_aliases": {"airplane": "drone", "kite": "drone"},
        "device": "cpu", "inference_size": 320,
    }
    detector = ObjectDetector(cfg)
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for d in detector.detect(frame):
        print(f"  {d.class_name:8s} conf={d.confidence:.2f} id={d.track_id} bbox={d.bbox}")
