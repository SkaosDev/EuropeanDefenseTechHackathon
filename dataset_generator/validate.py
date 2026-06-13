"""
validate.py — Contrôles de cohérence du dataset généré (verification §7 du plan).

Usage : python -m dataset_generator.validate --out out/

Vérifie : bornes géo, enveloppes cinématiques (vitesse/altitude/cap), arrivée sur cible,
invariant FPV-fibre/RF, ratio de faux positifs, matrices de confusion (agrégées),
intégrité des séquences (.npz). Affiche un rapport PASS/FAIL.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from .drone_classes import CLASS_ORDER


def _check(report, name, ok, detail=""):
    report.append((name, bool(ok), detail))


def validate(out_dir, cfg):
    report = []
    bbox = cfg["ukraine_bbox"]
    gt = pd.read_csv(os.path.join(out_dir, "ground_truth_trajectories.csv"))
    ev = pd.read_csv(os.path.join(out_dir, "detection_events.csv"))
    tg = pd.read_csv(os.path.join(out_dir, "targets.csv"))

    # --- géo : cibles dans la bbox, origines dehors ---
    in_bb = ((tg.lat.between(bbox["lat_min"], bbox["lat_max"]))
             & (tg.lon.between(bbox["lon_min"], bbox["lon_max"]))).all()
    _check(report, "Cibles dans la bbox UA", in_bb)

    # --- enveloppes cinématiques par classe (avec marge de bruit) ---
    speed_ok, alt_ok = True, True
    for cls, g in gt.groupby("drone_class"):
        c = cfg["drone_classes"][cls]
        smin, smax = c["speed_kmh"][0] / 3.6, c["speed_kmh"][1] / 3.6
        if not (g.speed.min() >= smin * 0.7 and g.speed.max() <= smax * 1.3):
            speed_ok = False
        amin, amax = c["alt_m"]
        if not (g.alt.min() >= amin - 60 and g.alt.max() <= amax + 60):
            alt_ok = False
    _check(report, "Vitesse dans l'enveloppe de classe (±bruit)", speed_ok)
    _check(report, "Altitude dans l'enveloppe de classe (±bruit)", alt_ok)

    # --- cap de virage : |Δcap|/dt <= max_turn + jitter ---
    dt = cfg["sim"]["dt_s"]
    limit = cfg["kinematics"]["max_turn_rate_deg_s"] * dt + 6 * cfg["kinematics"]["heading_jitter_deg"]
    worst = 0.0
    for _, g in gt.groupby("drone_id"):
        b = g.sort_values("t").bearing.to_numpy()
        if len(b) < 2:
            continue
        d = np.abs((np.diff(b) + 180) % 360 - 180)
        worst = max(worst, d.max())
    _check(report, "Cap de virage borné (pas de virage instantané)", worst <= limit,
           f"max |Δcap|={worst:.1f}° (limite {limit:.1f}°)")

    # --- arrivée sur cible ---
    reach_km = cfg["sim"]["reach_radius_m"] / 1000.0
    from . import geo
    tg_idx = tg.set_index("dest_id")[["lat", "lon"]]
    arrived = 0
    total = 0
    for did, g in gt.groupby("drone_id"):
        g = g.sort_values("t")
        last = g.iloc[-1]
        dst = tg_idx.loc[int(last.dest_id)]
        d_km = geo.distance_m(last.lat, last.lon, dst.lat, dst.lon) / 1000.0
        total += 1
        if d_km <= reach_km + last.speed * dt / 1000.0 + 0.5:
            arrived += 1
    _check(report, "Trajectoires arrivant sur leur cible", arrived / max(1, total) >= 0.9,
           f"{arrived}/{total} arrivées")

    # --- trajectoires non triviales (le drone parcourt une vraie distance) ---
    lengths_ok = 0
    for did, g in gt.groupby("drone_id"):
        g = g.sort_values("t")
        if geo.distance_m(g.iloc[0].lat, g.iloc[0].lon, g.iloc[-1].lat, g.iloc[-1].lon) > 5000:
            lengths_ok += 1
    _check(report, "Trajectoires non triviales (>5 km parcourus)",
           lengths_ok / max(1, total) >= 0.9, f"{lengths_ok}/{total}")

    # --- invariant : aucun événement RF pour un FPV fibre (détection réelle) ---
    cls_map = gt.drop_duplicates("drone_id").set_index("drone_id")["drone_class"]
    real = ev[ev.drone_id.notna()].copy()
    real["true_class"] = real.drone_id.map(cls_map)
    fpv_rf = ((real.true_class == "fpv_fiber") & (real.modality == "rf")).sum()
    _check(report, "FPV fibre invisible au RF", fpv_rf == 0, f"{fpv_rf} événements RF")

    # --- ratio de faux positifs raisonnable ---
    fp_ratio = (ev.drone_id.isna()).mean()
    _check(report, "Ratio FP raisonnable (5–45%)", 0.05 <= fp_ratio <= 0.45,
           f"{100 * fp_ratio:.1f}% FP")

    # --- confusion optique Shahed : diagonale dominante ---
    opt_sh = real[(real.modality == "optical") & (real.true_class == "shahed136")]
    if len(opt_sh) > 30:
        frac = (opt_sh.est_class == "shahed136").mean()
        _check(report, "Confusion optique Shahed plausible (0.5–0.85)", 0.5 <= frac <= 0.85,
               f"P(est=shahed)={frac:.2f}")

    # --- séquences .npz ---
    npz_path = os.path.join(out_dir, "sequences.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X = d["X"]
        _check(report, "Pas de NaN dans X", not np.isnan(X).any())
        _check(report, "Labels cible dans [0, n_targets)",
               d["y_target"].min() >= 0 and d["y_target"].max() < len(tg))
        _check(report, "Labels classe dans [0,4)",
               d["y_class"].min() >= 0 and d["y_class"].max() < len(CLASS_ORDER))
        _check(report, "Masque cohérent (>=1 événement/échantillon)",
               (d["mask"].sum(axis=1) >= 1).all(),
               f"X={X.shape}")

    return report


def _cli():
    ap = argparse.ArgumentParser(description="Validation du dataset counter-UAS.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    report = validate(args.out, cfg)
    print("\n===== VALIDATION =====")
    n_ok = 0
    for name, ok, detail in report:
        tag = "PASS" if ok else "FAIL"
        n_ok += ok
        print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
    print(f"\n  {n_ok}/{len(report)} contrôles OK")
    raise SystemExit(0 if n_ok == len(report) else 1)


if __name__ == "__main__":
    _cli()
