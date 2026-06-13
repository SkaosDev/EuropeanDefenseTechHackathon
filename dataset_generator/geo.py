"""
geo.py — Primitives géodésiques (WGS-84).

Deux familles de fonctions :
  * Scalaires (haversine pur math) pour le routage / la cinématique pas-à-pas.
  * Vectorisées numpy (haversine) pour la détection capteurs (matrices T×N rapides).

Convention de cap (bearing) : degrés, 0 = Nord, sens horaire (0..360), comme en navigation.
"""
from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_371_000.0  # rayon moyen terrestre


# --------------------------------------------------------------------------- #
#  Scalaire — haversine rapide (pur math), appelé en boucle serrée            #
#  (steering pas-à-pas, matrice O-D). Précision sphérique < 0.5 % : largement #
#  suffisante ici, et ~100x plus rapide que la géodésique itérative de geopy. #
# --------------------------------------------------------------------------- #
def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance haversine (mètres) entre deux points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Cap initial (degrés, 0=N horaire) du point 1 vers le point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float):
    """Point d'arrivée (lat, lon) en partant de (lat,lon), cap `bearing_deg`, sur `distance_m`."""
    delta = distance_m / EARTH_RADIUS_M
    theta = math.radians(bearing_deg)
    phi1, lam1 = math.radians(lat), math.radians(lon)
    sin_phi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))
    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * sin_phi2
    lam2 = lam1 + math.atan2(y, x)
    return math.degrees(phi2), (math.degrees(lam2) + 540.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
#  Vecteurs de cap (steering behaviors)                                        #
# --------------------------------------------------------------------------- #
def bearing_to_unit(bearing_deg):
    """Cap -> vecteur unitaire (est, nord)."""
    theta = math.radians(bearing_deg)
    return math.sin(theta), math.cos(theta)


def unit_to_bearing(east: float, north: float) -> float:
    """Vecteur (est, nord) -> cap (degrés, 0=N horaire)."""
    return (math.degrees(math.atan2(east, north)) + 360.0) % 360.0


def angle_diff(a: float, b: float) -> float:
    """Écart angulaire signé minimal a->b dans [-180, 180]."""
    return (b - a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
#  Vectorisé numpy — pour la détection capteurs                               #
# --------------------------------------------------------------------------- #
def haversine_m_vec(lat1, lon1, lat2, lon2):
    """
    Distance haversine (mètres), broadcast numpy.
    Tous les arguments en degrés ; supporte le broadcasting (ex. [T,1] vs [N]).
    """
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_vec(lat1, lon1, lat2, lon2):
    """Cap (degrés, 0=N horaire) de (lat1,lon1) vers (lat2,lon2), vectorisé."""
    phi1 = np.radians(np.asarray(lat1, dtype=float))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    x = np.sin(dlon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# --------------------------------------------------------------------------- #
#  Distance point -> segment / polyligne (fibre DAS, corridors)               #
# --------------------------------------------------------------------------- #
def _local_xy(lat, lon, lat0, lon0):
    """Projection équirectangulaire locale (mètres) autour de (lat0, lon0)."""
    x = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def point_to_segment_m(lat, lon, lat_a, lon_a, lat_b, lon_b):
    """
    Distance (mètres) d'un point au segment [A,B] via projection locale.
    Suffisamment précis pour des segments de quelques dizaines de km (fibre DAS).
    """
    lat0, lon0 = lat_a, lon_a
    px, py = _local_xy(lat, lon, lat0, lon0)
    ax, ay = 0.0, 0.0
    bx, by = _local_xy(lat_b, lon_b, lat0, lon0)
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (px * abx + py * aby) / ab2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def point_to_polyline_m(lat, lon, polyline):
    """Distance minimale (mètres) d'un point à une polyligne [(lat,lon), ...]."""
    best = float("inf")
    for (la, lo), (lb, lob) in zip(polyline[:-1], polyline[1:]):
        best = min(best, point_to_segment_m(lat, lon, la, lo, lb, lob))
    return best


def _xy_to_latlon(x, y, lat0, lon0):
    """Inverse de _local_xy : repasse d'un point local (m) en (lat, lon)."""
    lat = lat0 + math.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    return lat, lon


def nearest_point_on_polyline(lat, lon, polyline):
    """
    Point le plus proche (lat, lon) sur une polyligne + sa distance (mètres).
    Sert à l'attraction "corridor" du steering.
    """
    best = (polyline[0][0], polyline[0][1], float("inf"))
    for (la, lo), (lb, lob) in zip(polyline[:-1], polyline[1:]):
        lat0, lon0 = la, lo
        px, py = _local_xy(lat, lon, lat0, lon0)
        bx, by = _local_xy(lb, lob, lat0, lon0)
        ab2 = bx * bx + by * by
        t = 0.0 if ab2 == 0.0 else max(0.0, min(1.0, (px * bx + py * by) / ab2))
        cx, cy = t * bx, t * by
        d = math.hypot(px - cx, py - cy)
        if d < best[2]:
            clat, clon = _xy_to_latlon(cx, cy, lat0, lon0)
            best = (clat, clon, d)
    return best


def in_bbox(lat: float, lon: float, bbox: dict) -> bool:
    """Test d'appartenance à une bounding box {lat_min,lat_max,lon_min,lon_max}."""
    return (
        bbox["lat_min"] <= lat <= bbox["lat_max"]
        and bbox["lon_min"] <= lon <= bbox["lon_max"]
    )


# --------------------------------------------------------------------------- #
#  Point-in-polygon (clip des capteurs au territoire)                          #
# --------------------------------------------------------------------------- #
def _point_in_ring(x, y, ring):
    """Ray-casting : (x=lon, y=lat) dans l'anneau [(lon,lat), ...] ?"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_in_geojson(lat, lon, geom) -> bool:
    """Point (lat, lon) dans une géométrie GeoJSON Polygon/MultiPolygon (coords [lon,lat])."""
    t = geom["type"]
    polys = geom["coordinates"] if t == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        ext = poly[0]
        if _point_in_ring(lon, lat, ext):
            if not any(_point_in_ring(lon, lat, hole) for hole in poly[1:]):
                return True
    return False


def load_geojson_geometry(path):
    """Charge une géométrie GeoJSON {type, coordinates} depuis un fichier."""
    import json
    with open(path) as f:
        return json.load(f)
