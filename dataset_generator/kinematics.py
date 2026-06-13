"""
kinematics.py — Intégrateur de la "vraie" trajectoire (vérité-terrain).

Modèle de steering behaviors pas-à-pas (dt fixe) :
  desired = seek(waypoint) + avoid(zones DCA) + (optionnel) follow(corridor)
Le cap réel rejoint `desired` en respectant un cap de virage maximal (rayon de
virage réaliste, pas de virage à 90° instantané). Vitesse et altitude bruitées
dans l'enveloppe de la classe.

Retourne un dict de tableaux numpy (t, lat, lon, alt, speed, bearing) — base de la
vérité-terrain ET source des signatures physiques exploitées par les capteurs.
"""
from __future__ import annotations

import math

import numpy as np

from . import geo
from .drone_classes import sample_alt_m, sample_speed_mps


def simulate_trajectory(rng, olat, olon, waypoints, tgt, dc, cfg, dca_zones, corridor):
    """Simule une trajectoire origine -> cible. `corridor` = polyligne ou None."""
    k = cfg["kinematics"]
    r = cfg["routing"]
    sim = cfg["sim"]

    dt = float(sim["dt_s"])
    max_steps = int(sim["max_steps"])
    reach_m = float(sim["reach_radius_m"])
    wp_reach_m = float(r["waypoint_reach_m"])
    max_turn = k["max_turn_rate_deg_s"] * dt
    seek_w = r["seek_weight"]
    avoid_w = r["avoid_weight"]
    corr_w = r["corridor_weight"]
    corr_infl_m = r["corridor_influence_km"] * 1000.0

    base_speed = sample_speed_mps(rng, dc)
    base_alt = sample_alt_m(rng, dc)
    alt_lo, alt_hi = dc.alt_m

    # Zones DCA à éviter SAUF celle de la cible (on doit pouvoir l'atteindre).
    avoid = [
        z for z in dca_zones
        if geo.distance_m(z.lat, z.lon, tgt.lat, tgt.lon) > z.radius_m
    ]

    lat, lon = olat, olon
    heading = geo.initial_bearing(lat, lon, waypoints[0][0], waypoints[0][1])
    wp_idx = 0

    t_arr, lat_arr, lon_arr, alt_arr, spd_arr, brg_arr = [], [], [], [], [], []

    for step in range(max_steps):
        # --- vecteur seek vers le waypoint courant ---
        wlat, wlon = waypoints[wp_idx]
        b_seek = geo.initial_bearing(lat, lon, wlat, wlon)
        se, sn = geo.bearing_to_unit(b_seek)
        de, dn = seek_w * se, seek_w * sn

        # --- répulsion des zones DCA dans le rayon d'influence ---
        for z in avoid:
            dist = geo.distance_m(z.lat, z.lon, lat, lon)
            if dist < z.influence_m:
                mag = ((z.influence_m - dist) / z.influence_m) ** 2  # 0..1, fort de près
                b_away = geo.initial_bearing(z.lat, z.lon, lat, lon)  # pousse loin du centre
                ae, an = geo.bearing_to_unit(b_away)
                de += avoid_w * mag * ae
                dn += avoid_w * mag * an

        # --- attraction corridor optionnelle ---
        if corridor is not None:
            clat, clon, cdist = geo.nearest_point_on_polyline(lat, lon, corridor)
            if cdist < corr_infl_m:
                mag = min(1.0, cdist / corr_infl_m)  # nul si déjà sur le corridor
                b_corr = geo.initial_bearing(lat, lon, clat, clon)
                ce, cn = geo.bearing_to_unit(b_corr)
                de += corr_w * mag * ce
                dn += corr_w * mag * cn

        desired = geo.unit_to_bearing(de, dn)

        # --- slew du cap (cap de virage borné) + micro-jitter ---
        diff = geo.angle_diff(heading, desired)
        diff = max(-max_turn, min(max_turn, diff))
        heading = (heading + diff + rng.normal(0.0, k["heading_jitter_deg"])) % 360.0

        # --- vitesse / altitude bruitées ---
        speed = max(5.0, base_speed * (1.0 + rng.normal(0.0, k["speed_noise_frac"])))
        alt = float(np.clip(base_alt + rng.normal(0.0, k["alt_noise_m"]), alt_lo, alt_hi))

        # --- enregistrement de l'état courant (avant déplacement) ---
        t_arr.append(step * dt)
        lat_arr.append(lat)
        lon_arr.append(lon)
        alt_arr.append(alt)
        spd_arr.append(speed)
        brg_arr.append(heading)

        # --- avance ---
        lat, lon = geo.destination_point(lat, lon, heading, speed * dt)

        # --- gestion des waypoints / arrivée ---
        if geo.distance_m(lat, lon, tgt.lat, tgt.lon) <= reach_m:
            break
        if wp_idx < len(waypoints) - 1:
            if geo.distance_m(lat, lon, wlat, wlon) <= wp_reach_m:
                wp_idx += 1

    return {
        "t": np.array(t_arr),
        "lat": np.array(lat_arr),
        "lon": np.array(lon_arr),
        "alt": np.array(alt_arr),
        "speed": np.array(spd_arr),
        "bearing": np.array(brg_arr),
    }
