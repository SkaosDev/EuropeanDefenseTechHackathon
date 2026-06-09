"""Logger de détections non-bloquant (JSON-lines + rotation par taille).

L'écriture disque se fait dans un thread démon alimenté par une file : le
pipeline n'attend jamais l'I/O. Chaque ligne du fichier est un objet JSON.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger as log

from common.types import TrackedObject

# Sentinelle signalant au thread d'écriture de s'arrêter.
_STOP = object()


class DetectionLogger:
    """Écrit chaque détection dans un fichier JSON-lines, sans bloquer le pipeline."""

    def __init__(self, config: dict) -> None:
        self.path = Path(config["log_path"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(config.get("log_rotation_mb", 10)) * 1024 * 1024

        self._queue: queue.Queue = queue.Queue(maxsize=10_000)
        self._file = self.path.open("a", encoding="utf-8")
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        log.info("Logger détections -> {}", self.path)

    def log(self, objects: list[TrackedObject], timestamp: str | None = None) -> None:
        """Met en file les détections d'une frame (non bloquant ; ignore si saturé)."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        for obj in objects:
            record = {
                "timestamp": ts,
                "track_id": obj.track_id,
                "class_name": obj.detection.class_name,
                "confidence": round(obj.detection.confidence, 3),
                "distance_m": obj.distance_m,
                "bbox": list(obj.detection.bbox),
            }
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                pass  # on préfère perdre un log que bloquer le temps réel

    def _writer_loop(self) -> None:
        """Thread d'écriture : draine la file et gère la rotation."""
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            try:
                self._file.write(json.dumps(item) + "\n")
                self._file.flush()
                self._maybe_rotate()
            except Exception as exc:  # noqa: BLE001 - ne jamais tuer le thread de log
                log.warning("Écriture log échouée : {}", exc)

    def _maybe_rotate(self) -> None:
        """Renomme le fichier courant s'il dépasse la taille max, puis rouvre."""
        if self._file.tell() < self.max_bytes:
            return
        self._file.close()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rotated = self.path.with_name(f"{self.path.stem}_{stamp}{self.path.suffix}")
        os.replace(self.path, rotated)
        self._file = self.path.open("a", encoding="utf-8")
        log.info("Rotation log -> {}", rotated.name)

    def close(self) -> None:
        """Vide la file et ferme le fichier proprement."""
        self._queue.put(_STOP)
        self._thread.join(timeout=2.0)
        if not self._file.closed:
            self._file.close()


if __name__ == "__main__":
    # Test standalone : écrit quelques lignes puis relit le fichier.
    from common.types import Detection

    dl = DetectionLogger({"log_path": "logs/test_detections.json", "log_rotation_mb": 10})
    objs = [
        TrackedObject(1, Detection((10, 10, 60, 120), 0.88, 0, "person"), distance_m=3.1),
        TrackedObject(2, Detection((300, 200, 360, 240), 0.72, 2, "car"), distance_m=12.4),
    ]
    dl.log(objs)
    dl.close()
    print("Contenu de logs/test_detections.json :")
    print(Path("logs/test_detections.json").read_text(encoding="utf-8").strip())
