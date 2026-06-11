"""Types partagés par le pipeline : `Detection` et `TrackedObject`.

Volontairement minimal : on détecte, on suit, on estime la classe et la distance.
Pas de notion de "menace" — l'estimation de l'objet est simplement sa classe YOLO.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# Boîte englobante : (x1, y1, x2, y2) en pixels.
BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class Detection:
    """Résultat brut d'une détection YOLO sur une frame."""

    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None

    @property
    def center(self) -> tuple[int, int]:
        """Centre de la bbox (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def width(self) -> int:
        """Largeur de la bbox en pixels (sert à l'estimation de distance :
        plus stable que la hauteur quand le sujet se penche)."""
        return self.bbox[2] - self.bbox[0]


@dataclass(slots=True)
class TrackedObject:
    """Objet suivi : détection courante + identité + historique + distance estimée."""

    track_id: int
    detection: Detection
    trajectory: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=50))
    hits: int = 0           # nombre de détections cumulées
    lost_frames: int = 0    # frames consécutives sans détection
    distance_m: float | None = None  # distance estimée caméra→objet (mètres)
    first_seq: int = 0      # ordre d'acquisition (croissant) : sert à choisir la cible "la plus ancienne"
