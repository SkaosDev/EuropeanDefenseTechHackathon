"""
sensors.py — Réseau de capteurs (grille nationale) + génération d'événements.

Deux étapes :
  1. build_network() : place les capteurs ponctuels (optique/acoustique/vibration/RF)
     sur une grille jitterée + les lignes fibre DAS.
  2. simulate_events() : pour une trajectoire vérité-terrain donnée, chaque capteur à
     portée transforme la physique locale en ÉVÉNEMENTS bruités
     {t, sensor, modalité, classe estimée, confiance, bearing, range} + faux positifs.

Le flux d'événements est la SEULE entrée du modèle (la trajectoire reste cachée).

Réalisme clé : la portée effective d'un capteur = portée_base × force_d'émission de la
classe pour cette modalité. Une émission nulle (FPV fibre en RF) => AUCUN événement.
La distance utilisée est la distance OBLIQUE (sol + altitude) : un drone haut échappe
naturellement aux capteurs courte portée (vibration/DAS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geo
from .drone_classes import CLASS_ORDER

MODALITY_ORDER = ["optical", "acoustic", "vibration", "das", "rf"]


@dataclass
class PointSensors:
    modality: str
    ids: list
    lats: np.ndarray
    lons: np.ndarray
    base_range_m: float
    base_pd: float
    falloff: float
    gives_bearing: bool
    gives_range: bool


@dataclass
class DasLine:
    sensor_id: str
    polyline: list  # [(lat, lon), ...]


@dataclass
class SensorNetwork:
    point: dict           # modality -> PointSensors
    das_lines: list       # [DasLine]
    das_params: dict
    cfg: dict
    all_point_records: list = field(default_factory=list)  # pour export CSV


def _grid_points(bbox, grid_km, jitter_km, rng):
    """Grille régulière lat/lon sur la bbox, avec jitter de position."""
    mean_lat = (bbox["lat_min"] + bbox["lat_max"]) / 2.0
    dlat = grid_km / 111.0
    dlon = grid_km / (111.0 * np.cos(np.radians(mean_lat)))
    lats = np.arange(bbox["lat_min"], bbox["lat_max"], dlat)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"], dlon)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    glat = grid_lat.ravel()
    glon = grid_lon.ravel()
    jit_lat = rng.uniform(-jitter_km, jitter_km, glat.shape) / 111.0
    jit_lon = rng.uniform(-jitter_km, jitter_km, glon.shape) / (111.0 * np.cos(np.radians(mean_lat)))
    return glat + jit_lat, glon + jit_lon


def build_network(cfg: dict, rng) -> SensorNetwork:
    """Construit le réseau de capteurs depuis la config."""
    bbox = cfg["ukraine_bbox"]
    s = cfg["sensors"]
    point = {}
    records = []
    for mod in ["optical", "acoustic", "vibration", "rf"]:
        m = s["modalities"][mod]
        lats, lons = _grid_points(bbox, m["grid_km"], s["jitter_km"], rng)
        ids = [f"{mod[:3]}_{i}" for i in range(len(lats))]
        point[mod] = PointSensors(
            modality=mod, ids=ids, lats=lats, lons=lons,
            base_range_m=m["base_range_km"] * 1000.0, base_pd=m["base_pd"],
            falloff=m["falloff"], gives_bearing=m["gives_bearing"], gives_range=m["gives_range"],
        )
        for i in range(len(lats)):
            records.append({"sensor_id": ids[i], "lat": float(lats[i]), "lon": float(lons[i]),
                            "modality": mod, "range_km": m["base_range_km"]})

    das_lines = []
    for j, line in enumerate(s["das"]["lines"]):
        das_lines.append(DasLine(sensor_id=f"das_{j}", polyline=[(p[0], p[1]) for p in line]))

    return SensorNetwork(point=point, das_lines=das_lines, das_params=s["das"], cfg=cfg,
                         all_point_records=records)


# --------------------------------------------------------------------------- #
#  Génération d'événements                                                    #
# --------------------------------------------------------------------------- #
def _thin_indices(times, min_interval_s):
    """Sous-échantillonne des temps croissants pour respecter un intervalle minimal."""
    kept = []
    last = -1e18
    for t in times:
        if t - last >= min_interval_s:
            kept.append(True)
            last = t
        else:
            kept.append(False)
    return np.array(kept, dtype=bool)


def _sample_est_class(rng, confusion_row):
    return CLASS_ORDER[rng.choice(len(CLASS_ORDER), p=confusion_row)]


def simulate_events(rng, traj, dc, net: SensorNetwork, cfg: dict, drone_id: int):
    """Retourne la liste des événements (dicts) produits par un drone."""
    s = cfg["sensors"]
    interval = s["event_min_interval_s"]
    bearing_noise = s["bearing_noise_deg"]
    range_noise = s["range_noise_frac"]
    conf_noise = s["confidence_noise"]
    confusion = s["confusion"]

    # Sous-échantillonnage temporel de la détection (les événements sont de toute façon
    # éclaircis à event_min_interval_s) -> accélère sans perte de qualité.
    sl = slice(None, None, s.get("detection_stride", 1))
    tlat, tlon, talt, tt = traj["lat"][sl], traj["lon"][sl], traj["alt"][sl], traj["t"][sl]
    T = len(tt)
    events = []

    # --- capteurs ponctuels ---
    for mod, ps in net.point.items():
        emission = dc.emission[mod]
        if emission <= 0.0:      # ex. FPV fibre en RF -> jamais d'événement
            continue
        range_eff = ps.base_range_m * emission
        if range_eff <= 0.0:
            continue

        # pré-filtre spatial : ne garder que les capteurs proches de la trajectoire
        deg_margin = range_eff / 111000.0 + 0.1
        in_box = (
            (ps.lats >= tlat.min() - deg_margin) & (ps.lats <= tlat.max() + deg_margin)
            & (ps.lons >= tlon.min() - deg_margin) & (ps.lons <= tlon.max() + deg_margin)
        )
        sub = np.where(in_box)[0]
        if sub.size == 0:
            continue
        slat, slon = ps.lats[sub], ps.lons[sub]

        # distance oblique [T, Nsub]
        ground = geo.haversine_m_vec(tlat[:, None], tlon[:, None], slat[None, :], slon[None, :])
        slant = np.sqrt(ground ** 2 + talt[:, None] ** 2)
        within = slant <= range_eff
        if not within.any():
            continue
        norm = np.clip(1.0 - slant / range_eff, 0.0, 1.0)
        pd = ps.base_pd * norm ** ps.falloff
        draw = (rng.random(pd.shape) < pd) & within

        det_cols = np.where(draw.any(axis=0))[0]
        crow = confusion[mod]
        for c in det_cols:
            rows = np.where(draw[:, c])[0]
            keep = _thin_indices(tt[rows], interval)
            for ti in rows[keep]:
                true_b = geo.initial_bearing(slat[c], slon[c], tlat[ti], tlon[ti])
                ev = {
                    "t": float(tt[ti]),
                    "drone_id": drone_id,
                    "sensor_id": ps.ids[sub[c]],
                    "sensor_lat": float(slat[c]),
                    "sensor_lon": float(slon[c]),
                    "modality": mod,
                    "est_class": _sample_est_class(rng, crow[dc.name]),
                    "confidence": float(np.clip(pd[ti, c] + rng.normal(0, conf_noise), 0.05, 0.99)),
                    "bearing_est": float((true_b + rng.normal(0, bearing_noise[mod])) % 360.0),
                    "range_est": (float(slant[ti, c] * (1.0 + rng.normal(0, range_noise)))
                                  if ps.gives_range else np.nan),
                }
                events.append(ev)

    # --- fibre DAS (capteurs linéaires) ---
    emission = dc.emission["das"]
    if emission > 0.0:
        dp = net.das_params
        range_eff = dp["base_range_km"] * 1000.0 * emission
        crow = confusion["das"]
        das_margin = range_eff / 111000.0 + 0.1
        for line in net.das_lines:
            # pré-filtre : ignorer la ligne si la trajectoire ne l'approche jamais
            llat = [p[0] for p in line.polyline]
            llon = [p[1] for p in line.polyline]
            if (tlat.min() > max(llat) + das_margin or tlat.max() < min(llat) - das_margin
                    or tlon.min() > max(llon) + das_margin or tlon.max() < min(llon) - das_margin):
                continue
            dists = np.empty(T)
            nlat = np.empty(T)
            nlon = np.empty(T)
            for ti in range(T):
                cl, co, d = geo.nearest_point_on_polyline(tlat[ti], tlon[ti], line.polyline)
                dists[ti], nlat[ti], nlon[ti] = d, cl, co
            slant = np.sqrt(dists ** 2 + talt ** 2)
            within = slant <= range_eff
            if not within.any():
                continue
            norm = np.clip(1.0 - slant / range_eff, 0.0, 1.0)
            pd = dp["base_pd"] * norm ** dp["falloff"]
            draw = (rng.random(T) < pd) & within
            rows = np.where(draw)[0]
            keep = _thin_indices(tt[rows], interval)
            for ti in rows[keep]:
                events.append({
                    "t": float(tt[ti]),
                    "drone_id": drone_id,
                    "sensor_id": line.sensor_id,
                    "sensor_lat": float(nlat[ti]),
                    "sensor_lon": float(nlon[ti]),
                    "modality": "das",
                    "est_class": _sample_est_class(rng, crow[dc.name]),
                    "confidence": float(np.clip(pd[ti] + rng.normal(0, conf_noise), 0.05, 0.99)),
                    "bearing_est": float(rng.uniform(0, 360)),  # DAS : pas de cap fiable
                    "range_est": np.nan,
                })

    # --- faux positifs (clutter) : non liés au drone (drone_id=null) ---
    # Volume proportionnel au nb d'événements réels -> ratio FP contrôlé et réaliste.
    n_clutter = int(rng.poisson(s["fp_ratio"] * len(events)))
    if n_clutter > 0 and len(net.all_point_records) > 0:
        recs = net.all_point_records
        for _ in range(n_clutter):
            rec = recs[rng.integers(len(recs))]
            mod = rec["modality"]
            events.append({
                "t": float(rng.uniform(tt.min(), tt.max())),
                "drone_id": None,
                "sensor_id": rec["sensor_id"],
                "sensor_lat": rec["lat"],
                "sensor_lon": rec["lon"],
                "modality": mod,
                "est_class": CLASS_ORDER[rng.integers(len(CLASS_ORDER))],
                "confidence": float(rng.uniform(0.1, 0.5)),
                "bearing_est": float(rng.uniform(0, 360)),
                "range_est": np.nan,
            })

    return events
