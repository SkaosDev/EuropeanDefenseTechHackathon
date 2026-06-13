"""
main.py — CLI du simulateur de dataset counter-UAS.

Exemple :
  python -m dataset_generator.main --n-drones 2000 --seed 42 --out out/

Produit dans --out : targets.csv, ground_truth_trajectories.csv, detection_events.csv,
sensors.csv, trajectories.geojson, sensors.geojson, sequences.npz, dataset_meta.json,
map.html.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

from . import export, sequence_prep, simulator

HERE = os.path.dirname(__file__)


def _summary(gt_df, events_df):
    """Petit récap + contrôles de cohérence rapides."""
    n_drones = gt_df["drone_id"].nunique()
    n_ev = len(events_df)
    real = events_df["drone_id"].notna().sum()
    clutter = n_ev - real
    # invariant : un FPV fibre ne doit JAMAIS produire d'événement RF (drone_id réel).
    cls = gt_df.drop_duplicates("drone_id").set_index("drone_id")["drone_class"]
    real_ev = events_df[events_df["drone_id"].notna()].copy()
    real_ev["true_class"] = real_ev["drone_id"].map(cls)
    fpv_rf = ((real_ev["true_class"] == "fpv_fiber") & (real_ev["modality"] == "rf")).sum()
    print("\n===== RÉCAP =====")
    print(f"  drones                : {n_drones}")
    print(f"  événements            : {n_ev}  (réels {real}, clutter {clutter}, "
          f"{100 * clutter / max(1, n_ev):.1f}% FP)")
    print(f"  modalités             : {dict(events_df['modality'].value_counts())}")
    print(f"  classes (vérité)      : {dict(cls.value_counts())}")
    print(f"  CHECK FPV-fibre en RF : {fpv_rf}  (doit être 0)")


def main():
    ap = argparse.ArgumentParser(description="Simulateur de dataset counter-UAS.")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-drones", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dt", type=float, default=None, help="pas de temps (s)")
    ap.add_argument("--grid-km", type=float, default=None,
                    help="surcharge le pas de grille de TOUTES les modalités ponctuelles")
    ap.add_argument("--no-prep", action="store_true", help="ne pas générer sequences.npz")
    ap.add_argument("--no-viz", action="store_true", help="ne pas générer map.html")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.n_drones is not None:
        cfg["sim"]["n_drones"] = args.n_drones
    if args.seed is not None:
        cfg["sim"]["seed"] = args.seed
    if args.dt is not None:
        cfg["sim"]["dt_s"] = args.dt
    if args.grid_km is not None:
        for m in cfg["sensors"]["modalities"].values():
            m["grid_km"] = args.grid_km

    seed = cfg["sim"]["seed"]
    n_drones = cfg["sim"]["n_drones"]
    rng = np.random.default_rng(seed)

    print(f"[main] génération de {n_drones} drones (seed={seed}, dt={cfg['sim']['dt_s']}s)")
    targets, net, gt_df, events_df = simulator.generate_all(cfg, rng, n_drones)

    os.makedirs(args.out, exist_ok=True)
    export.write_all(args.out, targets, net, gt_df, events_df)
    print(f"[main] CSV/GeoJSON écrits dans {args.out}/")

    if not args.no_prep:
        data = sequence_prep.build_sequences(gt_df, events_df, cfg, seed=seed)
        meta = sequence_prep.make_meta(cfg, targets)
        path = sequence_prep.save(args.out, data, meta)
        print(f"[main] séquences LSTM -> {path}  X={data['X'].shape}  "
              f"(train {int((data['is_val'] == 0).sum())}, val {int(data['is_val'].sum())})")

    if not args.no_viz:
        try:
            from . import visualize
            path = visualize.build_map(args.out)
            print(f"[main] carte pitch -> {path}")
        except Exception as exc:   # folium absent / souci d'affichage ne doit pas tout casser
            print(f"[main] viz ignorée ({exc})")

    _summary(gt_df, events_df)


if __name__ == "__main__":
    main()
