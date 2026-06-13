"""
export.py — Écriture des sorties (CSV + GeoJSON).

  * targets.csv                  : les 65 cibles (espace de labels)
  * ground_truth_trajectories.csv: vérité-terrain (PAS l'entrée du modèle)
  * detection_events.csv         : flux d'événements (ENTRÉE du modèle)
  * sensors.csv                  : capteurs ponctuels
  * trajectories.geojson         : LineStrings (carte pitch)
  * sensors.geojson              : capteurs ponctuels + lignes fibre DAS
"""
from __future__ import annotations

import json
import os

import pandas as pd


def write_targets(targets, path):
    rows = [{"dest_id": t.dest_id, "name": t.name, "oblast": t.oblast,
             "zone_type": t.zone_type, "objective": t.objective,
             "lat": t.lat, "lon": t.lon, "pop": t.pop} for t in targets]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_ground_truth(gt_df, path):
    gt_df.to_csv(path, index=False)


def write_events(events_df, path):
    events_df.to_csv(path, index=False)


def write_sensors_csv(net, path):
    pd.DataFrame(net.all_point_records).to_csv(path, index=False)


def write_trajectories_geojson(gt_df, path):
    """Une LineString par drone (couleur par cible côté front)."""
    features = []
    for did, g in gt_df.groupby("drone_id"):
        g = g.sort_values("t")
        coords = [[float(lo), float(la)] for la, lo in zip(g["lat"], g["lon"])]  # GeoJSON = [lon, lat]
        if len(coords) < 2:
            continue
        first = g.iloc[0]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "drone_id": int(did),
                "drone_class": first["drone_class"],
                "origin": first["origin"],
                "dest_id": int(first["dest_id"]),
                "dest_name": first["dest_name"],
                "dest_oblast": first["dest_oblast"],
                "objective": first["objective"],
            },
        })
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def write_sensors_geojson(net, path):
    """Capteurs ponctuels (Point) + fibres DAS (LineString)."""
    features = []
    for rec in net.all_point_records:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [rec["lon"], rec["lat"]]},
            "properties": {"sensor_id": rec["sensor_id"], "modality": rec["modality"],
                           "range_km": rec["range_km"]},
        })
    for line in net.das_lines:
        coords = [[lo, la] for la, lo in line.polyline]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"sensor_id": line.sensor_id, "modality": "das"},
        })
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def write_all(out_dir, targets, net, gt_df, events_df):
    """Écrit l'ensemble des fichiers dans `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, name)
    write_targets(targets, p("targets.csv"))
    write_ground_truth(gt_df, p("ground_truth_trajectories.csv"))
    write_events(events_df, p("detection_events.csv"))
    write_sensors_csv(net, p("sensors.csv"))
    write_trajectories_geojson(gt_df, p("trajectories.geojson"))
    write_sensors_geojson(net, p("sensors.geojson"))
