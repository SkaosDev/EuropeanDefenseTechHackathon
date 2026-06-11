"""Estimation monoculaire de la distance caméra → objet.

Principe (sténopé) : connaissant la **largeur** réelle approximative d'un objet selon sa
classe et la focale de la caméra en pixels (déduite du champ de vision horizontal), la
distance vaut :

    distance = largeur_reelle_m * focale_px / largeur_bbox_px

On utilise la largeur plutôt que la hauteur car elle est bien plus stable : se pencher ou
s'accroupir écrase la hauteur de la bbox (et ferait "bondir" la distance) mais change peu
la largeur. Reste une estimation grossière (objet vu de face, taille typique) — un lissage
temporel par track est appliqué en amont (cf. main.py) pour amortir le bruit.
"""

from __future__ import annotations

import math

from common.types import Detection


class DistanceEstimator:
    """Estime une distance en mètres à partir de la largeur d'une bbox."""

    def __init__(self, config: dict, frame_width: int) -> None:
        self.enable = config.get("enable", True)
        self.hfov_deg = config.get("hfov_deg", 66.0)  # réutilisé par la vue 3D du dashboard
        hfov = math.radians(self.hfov_deg)
        # Focale en pixels déduite du champ de vision horizontal et de la largeur image.
        self.focal_px = (frame_width / 2.0) / math.tan(hfov / 2.0)
        self.known_widths = config.get("known_widths_m", {})
        self.default_width = config.get("default_width_m", 0.5)

    def estimate(self, detection: Detection) -> float | None:
        """Distance brute estimée (m) pour une détection, ou None si désactivé/indéterminé."""
        if not self.enable:
            return None
        w_px = detection.width
        if w_px <= 0:
            return None
        real_w = self.known_widths.get(detection.class_name, self.default_width)
        return round(real_w * self.focal_px / w_px, 2)


if __name__ == "__main__":
    # Test standalone : un drone de 0.4 m de large, caméra 640 px / 66° HFOV.
    cfg = {"enable": True, "hfov_deg": 66.0,
           "known_widths_m": {"drone": 0.4}, "default_width_m": 0.5}
    est = DistanceEstimator(cfg, frame_width=640)
    print(f"focale ~ {est.focal_px:.0f} px")
    for w in (160, 80, 40, 20):
        d = Detection((0, 0, w, 60), 0.9, 0, "drone")
        print(f"bbox w={w:3d}px -> distance ~ {est.estimate(d)} m")
