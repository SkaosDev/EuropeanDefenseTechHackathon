"""
visualize.py — Carte folium pour le pitch.

Affiche : cibles (par type), capteurs (échantillonnés, par modalité), lignes fibre DAS,
trajectoires vérité-terrain (par classe) et un échantillon d'événements détectés.
Génère un fichier HTML autonome.

Usage : python -m dataset_generator.visualize --out out/
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

CLASS_COLORS = {"shahed136": "#d62728", "gerbera": "#ff7f0e",
                "fpv_fiber": "#9467bd", "lancet": "#1f77b4"}
MOD_COLORS = {"optical": "#2ca02c", "acoustic": "#1f77b4", "vibration": "#8c564b",
              "das": "#e377c2", "rf": "#ff7f0e"}
ZONE_COLORS = {"city": "gray", "power_tpp": "red", "power_hpp": "darkred",
               "power_npp": "black", "airbase": "blue", "port": "cadetblue",
               "defense_industry": "purple"}


def build_map(out_dir, max_traj=60, max_sensors=350, max_events=1500):
    import folium

    bbox_center = [48.5, 31.5]
    m = folium.Map(location=bbox_center, zoom_start=6, tiles="CartoDB positron")

    # --- cibles ---
    targets = pd.read_csv(os.path.join(out_dir, "targets.csv"))
    fg_t = folium.FeatureGroup(name="Cibles", show=True)
    for _, t in targets.iterrows():
        folium.CircleMarker(
            [t["lat"], t["lon"]], radius=4,
            color=ZONE_COLORS.get(t["zone_type"], "gray"), fill=True, fill_opacity=0.9,
            popup=f"{t['name']} ({t['zone_type']}, {t['oblast']})",
        ).add_to(fg_t)
    fg_t.add_to(m)

    # --- capteurs (échantillon) ---
    sensors = pd.read_csv(os.path.join(out_dir, "sensors.csv"))
    if len(sensors) > max_sensors:
        sensors = sensors.sample(max_sensors, random_state=0)
    fg_s = folium.FeatureGroup(name="Capteurs (échantillon)", show=False)
    for _, s in sensors.iterrows():
        folium.CircleMarker(
            [s["lat"], s["lon"]], radius=1.5,
            color=MOD_COLORS.get(s["modality"], "black"), fill=True, fill_opacity=0.5,
        ).add_to(fg_s)
    fg_s.add_to(m)

    # --- lignes DAS + reste des capteurs via geojson ---
    sj_path = os.path.join(out_dir, "sensors.geojson")
    if os.path.exists(sj_path):
        with open(sj_path) as f:
            sj = json.load(f)
        fg_das = folium.FeatureGroup(name="Fibre DAS", show=True)
        for feat in sj["features"]:
            if feat["geometry"]["type"] == "LineString":
                coords = [[c[1], c[0]] for c in feat["geometry"]["coordinates"]]
                folium.PolyLine(coords, color=MOD_COLORS["das"], weight=2, opacity=0.6).add_to(fg_das)
        fg_das.add_to(m)

    # --- trajectoires (échantillon) ---
    with open(os.path.join(out_dir, "trajectories.geojson")) as f:
        traj = json.load(f)
    fg_tr = folium.FeatureGroup(name="Trajectoires (vérité-terrain)", show=True)
    for feat in traj["features"][:max_traj]:
        coords = [[c[1], c[0]] for c in feat["geometry"]["coordinates"]]
        cls = feat["properties"]["drone_class"]
        folium.PolyLine(
            coords, color=CLASS_COLORS.get(cls, "black"), weight=2, opacity=0.7,
            popup=f"{cls} -> {feat['properties']['dest_name']}",
        ).add_to(fg_tr)
    fg_tr.add_to(m)

    # --- événements détectés (échantillon) ---
    events = pd.read_csv(os.path.join(out_dir, "detection_events.csv"))
    if len(events) > max_events:
        events = events.sample(max_events, random_state=0)
    fg_e = folium.FeatureGroup(name="Événements détectés (échantillon)", show=False)
    for _, e in events.iterrows():
        folium.CircleMarker(
            [e["sensor_lat"], e["sensor_lon"]], radius=2,
            color=MOD_COLORS.get(e["modality"], "black"), fill=True, fill_opacity=0.7,
            popup=f"{e['modality']} / est={e['est_class']} / conf={e['confidence']:.2f}",
        ).add_to(fg_e)
    fg_e.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_path = os.path.join(out_dir, "map.html")
    m.save(out_path)
    return out_path


def _cli():
    ap = argparse.ArgumentParser(description="Carte folium des sorties du simulateur.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = build_map(args.out)
    print(f"[visualize] carte -> {path}")


if __name__ == "__main__":
    _cli()
