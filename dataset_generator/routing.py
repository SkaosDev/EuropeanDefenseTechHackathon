"""
routing.py — Macro-routing (matrice Origine-Destination) + micro-routing (waypoints).

Macro : tire une origine et une cible plausibles selon la classe du drone.
  * Classes `hub`     : partent d'un site de lancement réel (§1.A), cible pondérée par
    la valeur-cible (zone_type), la distance et les préférences directionnelles OSINT.
  * Classes `forward` : courte portée -> point de lancement avancé près du front, calculé
    à partir d'une cible située dans un oblast de front.

Micro : produit des waypoints d'ancrage (origine -> midpoints latéraux -> cible) ;
l'évitement fin des villes (DCA) et le suivi de corridor sont gérés par le steering
de kinematics.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geo


@dataclass
class Target:
    dest_id: int
    name: str
    oblast: str
    zone_type: str
    lat: float
    lon: float
    pop: int
    objective: str


@dataclass
class DcaZone:
    lat: float
    lon: float
    radius_m: float
    influence_m: float


def build_targets(cfg: dict) -> list:
    """Construit la liste des cibles avec dest_id séquentiel + objectif dérivé."""
    obj_map = cfg["objective_by_zone_type"]
    targets = []
    for i, t in enumerate(cfg["targets"]):
        targets.append(
            Target(
                dest_id=i,
                name=t["name"],
                oblast=t["oblast"],
                zone_type=t["zone_type"],
                lat=float(t["lat"]),
                lon=float(t["lon"]),
                pop=int(t.get("pop", 0)),
                objective=obj_map[t["zone_type"]],
            )
        )
    return targets


def build_dca_zones(cfg: dict, targets: list) -> list:
    """Zones DCA = villes dont la population dépasse le seuil (à contourner)."""
    d = cfg["routing"]["dca"]
    radius_m = d["radius_km"] * 1000.0
    influence_m = d["influence_km"] * 1000.0
    zones = []
    for t in targets:
        if t.zone_type == "city" and t.pop >= d["pop_threshold"]:
            zones.append(DcaZone(t.lat, t.lon, radius_m, influence_m))
    return zones


# --------------------------------------------------------------------------- #
#  Macro-routing                                                              #
# --------------------------------------------------------------------------- #
def _sample_origin(rng, origins: list):
    weights = np.array([o["weight"] for o in origins], dtype=float)
    idx = rng.choice(len(origins), p=weights / weights.sum())
    return origins[idx]


def _target_weights(origin, targets, dc, cfg):
    """Poids O-D d'une origine HUB vers chaque cible (0 si hors de portée)."""
    r = cfg["routing"]
    scale_km = r["distance_scale_km"]
    max_km = dc.range_km * r["range_margin"]
    prefers = set(origin.get("prefers", []))
    ztw = r["zone_type_weight"]
    w = np.zeros(len(targets))
    for i, t in enumerate(targets):
        dist_km = geo.distance_m(origin["lat"], origin["lon"], t.lat, t.lon) / 1000.0
        if dist_km > max_km:
            continue
        boost = r["prefer_boost"] if t.oblast in prefers else 1.0
        w[i] = ztw[t.zone_type] * np.exp(-dist_km / scale_km) * boost
    return w


def choose_od(rng, dc, origins, targets, cfg):
    """
    Retourne (origin_lat, origin_lon, origin_name, target:Target) pour un drone.
    Gère les modes `hub` (départ réel) et `forward` (point avancé calculé).
    """
    if dc.origin_mode == "hub":
        for _ in range(6):  # quelques essais si l'origine tirée n'a aucune cible à portée
            origin = _sample_origin(rng, origins)
            w = _target_weights(origin, targets, dc, cfg)
            if w.sum() > 0:
                tgt = targets[rng.choice(len(targets), p=w / w.sum())]
                return origin["lat"], origin["lon"], origin["name"], tgt
        # repli : cible la plus proche de l'origine
        origin = _sample_origin(rng, origins)
        dists = [geo.distance_m(origin["lat"], origin["lon"], t.lat, t.lon) for t in targets]
        tgt = targets[int(np.argmin(dists))]
        return origin["lat"], origin["lon"], origin["name"], tgt

    # mode forward : cible dans un oblast de front, origine = point avancé côté ennemi
    front = set(cfg["front_oblasts"])
    ztw = cfg["routing"]["zone_type_weight"]
    cand = [t for t in targets if t.oblast in front]
    w = np.array([ztw[t.zone_type] for t in cand], dtype=float)
    tgt = cand[rng.choice(len(cand), p=w / w.sum())]
    # cap depuis la cible vers le hub réel le plus proche -> direction "front"
    hub = min(origins, key=lambda o: geo.distance_m(tgt.lat, tgt.lon, o["lat"], o["lon"]))
    bearing = geo.initial_bearing(tgt.lat, tgt.lon, hub["lat"], hub["lon"])
    launch_km = dc.range_km * rng.uniform(0.4, 0.85)
    olat, olon = geo.destination_point(tgt.lat, tgt.lon, bearing, launch_km * 1000.0)
    return olat, olon, f"forward:{tgt.oblast}", tgt


# --------------------------------------------------------------------------- #
#  Micro-routing                                                              #
# --------------------------------------------------------------------------- #
def build_waypoints(rng, olat, olon, tgt, cfg):
    """Waypoints d'ancrage : origine -> midpoints latéraux aléatoires -> cible."""
    r = cfg["routing"]
    lo, hi = r["n_midpoints"]
    n_mid = int(rng.integers(lo, hi + 1))
    total_m = geo.distance_m(olat, olon, tgt.lat, tgt.lon)
    route_bearing = geo.initial_bearing(olat, olon, tgt.lat, tgt.lon)
    wps = []
    for k in range(1, n_mid + 1):
        frac = k / (n_mid + 1)
        # point sur la ligne directe à la fraction `frac`
        plat, plon = geo.destination_point(olat, olon, route_bearing, total_m * frac)
        # décalage latéral gaussien (perpendiculaire à la route)
        offset_m = rng.normal(0.0, r["midpoint_offset_km"] * 1000.0)
        perp = (route_bearing + (90.0 if offset_m >= 0 else -90.0)) % 360.0
        plat, plon = geo.destination_point(plat, plon, perp, abs(offset_m))
        wps.append((plat, plon))
    wps.append((tgt.lat, tgt.lon))
    return wps


def maybe_pick_corridor(rng, olat, olon, tgt, cfg):
    """Choisit (probabiliste) un corridor d'attraction proche de la route, sinon None."""
    r = cfg["routing"]
    if rng.random() > r["corridor_follow_prob"]:
        return None
    mlat = (olat + tgt.lat) / 2.0
    mlon = (olon + tgt.lon) / 2.0
    best, best_d = None, float("inf")
    for c in r["corridors"]:
        poly = [(p[0], p[1]) for p in c["polyline"]]
        d = geo.point_to_polyline_m(mlat, mlon, poly)
        if d < best_d:
            best, best_d = poly, d
    return best
