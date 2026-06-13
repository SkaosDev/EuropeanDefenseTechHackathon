"""
drone_classes.py — Profils des classes de drones.

Charge depuis config.yaml l'enveloppe cinématique (vitesse/altitude/portée) et les
forces d'émission par modalité (qui pilotent la portée effective des capteurs).
Fournit aussi l'échantillonnage de la classe et des paramètres de vol d'un drone.
"""
from __future__ import annotations

from dataclasses import dataclass

# Ordre canonique des classes (= colonnes des matrices de confusion, indices de label).
CLASS_ORDER = ["shahed136", "gerbera", "fpv_fiber", "lancet"]
# 5e "classe" réservée au clutter (faux positifs) côté événements.
EST_CLASS_ORDER = CLASS_ORDER + ["clutter"]


@dataclass
class DroneClass:
    name: str                 # clé interne (ex. "shahed136")
    label: str                # libellé affichable
    spawn_weight: float
    speed_kmh: tuple          # (min, max)
    alt_m: tuple              # (min, max)
    range_km: float
    origin_mode: str          # "hub" | "forward"
    emission: dict            # {modality: force 0..1}


def load_drone_classes(cfg: dict) -> dict:
    """Construit {name -> DroneClass} depuis la section `drone_classes` du config."""
    out = {}
    for name in CLASS_ORDER:
        c = cfg["drone_classes"][name]
        out[name] = DroneClass(
            name=name,
            label=c["label"],
            spawn_weight=float(c["spawn_weight"]),
            speed_kmh=tuple(c["speed_kmh"]),
            alt_m=tuple(c["alt_m"]),
            range_km=float(c["range_km"]),
            origin_mode=c["origin_mode"],
            emission=dict(c["emission"]),
        )
    return out


def sample_class(rng, classes: dict) -> DroneClass:
    """Tire une classe selon les poids `spawn_weight`."""
    names = list(classes.keys())
    weights = [classes[n].spawn_weight for n in names]
    total = sum(weights)
    probs = [w / total for w in weights]
    return classes[rng.choice(names, p=probs)]


def sample_speed_mps(rng, dc: DroneClass) -> float:
    """Vitesse de croisière de base (m/s) tirée dans l'enveloppe de la classe."""
    lo, hi = dc.speed_kmh
    return rng.uniform(lo, hi) / 3.6


def sample_alt_m(rng, dc: DroneClass) -> float:
    """Altitude de croisière de base (m) tirée dans l'enveloppe de la classe."""
    lo, hi = dc.alt_m
    return rng.uniform(lo, hi)
