"""Rendu visuel : bboxes, labels (classe + distance), trajectoires et HUD.

Travaille sur des frames BGR OpenCV. La classe prioritaire (drone) est dessinée en
**rouge** pour la mettre en évidence ; le reste en vert.
"""

from __future__ import annotations

import cv2
import numpy as np

from common.types import TrackedObject

_RED = (0, 0, 255)      # classe prioritaire (drone)
_GREEN = (0, 255, 0)    # autres classes
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Overlay:
    """Dessine bboxes, labels, trajectoires et HUD (FPS + heure)."""

    def __init__(self, config: dict) -> None:
        self.draw_trajectory = config.get("draw_trajectory", True)

    def draw(self, frame: np.ndarray, objects: list[TrackedObject],
             fps: float, timestamp: str, priority: str = "drone") -> np.ndarray:
        """Annoter la frame avec les objets suivis et le HUD. Modifie/renvoie la frame."""
        for obj in objects:
            color = _RED if obj.detection.class_name.lower() == priority.lower() else _GREEN
            self._draw_object(frame, obj, color)
        self._draw_hud(frame, fps, timestamp)
        return frame

    def _draw_object(self, frame: np.ndarray, obj: TrackedObject, color) -> None:
        """Dessine bbox + label (classe, id, confiance, distance) + trajectoire."""
        x1, y1, x2, y2 = obj.detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{obj.detection.class_name} #{obj.track_id} {obj.detection.confidence:.2f}"
        if obj.distance_m is not None:
            label += f" ~{obj.distance_m:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)

        if self.draw_trajectory and len(obj.trajectory) > 1:
            pts = np.array(obj.trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

    def _draw_hud(self, frame: np.ndarray, fps: float, timestamp: str) -> None:
        """Affiche FPS et heure en haut à droite."""
        w = frame.shape[1]
        clock = timestamp.split("T")[-1] if "T" in timestamp else timestamp
        for i, text in enumerate([f"FPS {fps:4.1f}", clock]):
            (tw, _), _ = cv2.getTextSize(text, _FONT, 0.6, 1)
            x, y = w - tw - 10, 22 + i * 22
            cv2.putText(frame, text, (x, y), _FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), _FONT, 0.6, _GREEN, 1, cv2.LINE_AA)


if __name__ == "__main__":
    # Test standalone : une image annotée enregistrée.
    from collections import deque

    from common.types import Detection

    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    obj = TrackedObject(
        track_id=1,
        detection=Detection((200, 150, 280, 350), 0.91, 0, "drone"),
        trajectory=deque([(240, 250), (245, 255), (250, 260)], maxlen=50),
        hits=5, distance_m=12.3,
    )
    Overlay({"draw_trajectory": True}).draw(img, [obj], 18.3, "2026-06-09T12:00:00")
    cv2.imwrite("overlay_test.png", img)
    print("Image annotée écrite dans overlay_test.png")
