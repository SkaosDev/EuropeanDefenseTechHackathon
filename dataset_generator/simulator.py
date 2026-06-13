"""
simulator.py — Orchestration : spawn -> vraie trajectoire -> événements -> labels.

`generate_all()` boucle sur N drones et renvoie :
  * targets   : liste des cibles (espace de labels)
  * net       : réseau de capteurs (pour export/visualisation)
  * gt_df     : trajectoires vérité-terrain (PAS l'entrée du modèle)
  * events_df : flux d'événements de détection (ENTRÉE du modèle)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kinematics, routing, sensors
from .drone_classes import load_drone_classes, sample_class


def simulate_drone(rng, drone_id, classes, origins, targets, dca_zones, net, cfg):
    """Simule un drone : renvoie (lignes vérité-terrain, événements)."""
    dc = sample_class(rng, classes)
    olat, olon, oname, tgt = routing.choose_od(rng, dc, origins, targets, cfg)
    waypoints = routing.build_waypoints(rng, olat, olon, tgt, cfg)
    corridor = routing.maybe_pick_corridor(rng, olat, olon, tgt, cfg)
    traj = kinematics.simulate_trajectory(rng, olat, olon, waypoints, tgt, dc, cfg, dca_zones, corridor)

    if len(traj["t"]) == 0:
        return [], []

    gt_rows = []
    n = len(traj["t"])
    for i in range(n):
        gt_rows.append({
            "t": float(traj["t"][i]),
            "drone_id": drone_id,
            "lat": float(traj["lat"][i]),
            "lon": float(traj["lon"][i]),
            "alt": float(traj["alt"][i]),
            "speed": float(traj["speed"][i]),
            "bearing": float(traj["bearing"][i]),
            "drone_class": dc.name,
            "origin": oname,
            "dest_id": tgt.dest_id,
            "dest_name": tgt.name,
            "dest_oblast": tgt.oblast,
            "dest_zone_type": tgt.zone_type,
            "objective": tgt.objective,
        })

    events = sensors.simulate_events(rng, traj, dc, net, cfg, drone_id)
    return gt_rows, events


def generate_all(cfg, rng, n_drones, log=print):
    """Génère le dataset complet. Renvoie (targets, net, gt_df, events_df)."""
    classes = load_drone_classes(cfg)
    origins = cfg["origins"]
    targets = routing.build_targets(cfg)
    dca_zones = routing.build_dca_zones(cfg, targets)
    net = sensors.build_network(cfg, rng)

    gt_rows, events = [], []
    event_id = 0
    for did in range(n_drones):
        g, e = simulate_drone(rng, did, classes, origins, targets, dca_zones, net, cfg)
        gt_rows.extend(g)
        for ev in e:
            ev["event_id"] = event_id
            ev["scenario_id"] = did   # lie l'événement au scénario (réel ET clutter)
            event_id += 1
            events.append(ev)
        if (did + 1) % max(1, n_drones // 20) == 0:
            log(f"  ... {did + 1}/{n_drones} drones, {len(events)} événements")

    gt_df = pd.DataFrame(gt_rows)
    cols = ["event_id", "scenario_id", "t", "drone_id", "sensor_id", "sensor_lat", "sensor_lon",
            "modality", "est_class", "confidence", "bearing_est", "range_est"]
    events_df = pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
    return targets, net, gt_df, events_df
