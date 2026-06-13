"""
sequence_prep.py — Flux d'événements -> séquences prêtes pour un LSTM/GRU multi-têtes.

Le modèle ne voit QUE les événements. Pour chaque scénario (un drone réel + son clutter),
on construit plusieurs échantillons tronqués à différents taux d'observation
(`observation_fractions`) : c'est l'augmentation "alerte précoce" (prédire tôt, sur peu
d'événements). Chaque échantillon porte 3 labels :

  * y_target : id de la cible (parmi 65)        -> tête distribution sur cibles
  * y_class  : classe du drone (parmi 4)        -> tête classe
  * y_future : positions futures normalisées     -> tête trajectoire (+ y_future_mask)

Vecteur de features par événement (n_feat = 17) :
  [dt_norm, slat_norm, slon_norm, modality(5 one-hot), est_class(4 one-hot),
   confidence, bearing_sin, bearing_cos, range_norm, has_range]

Split train/val PAR SCÉNARIO (anti-fuite : aucun échantillon d'un même drone des deux côtés).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import yaml

from .drone_classes import CLASS_ORDER
from .sensors import MODALITY_ORDER

RANGE_SCALE_M = 10000.0  # normalisation de range_est (~10 km)

FEATURE_NAMES = (
    ["dt_norm", "slat_norm", "slon_norm"]
    + [f"mod_{m}" for m in MODALITY_ORDER]
    + [f"est_{c}" for c in CLASS_ORDER]
    + ["confidence", "bearing_sin", "bearing_cos", "range_norm", "has_range"]
)
N_FEAT = len(FEATURE_NAMES)


def _norm_lat(lat, bbox):
    return (lat - bbox["lat_min"]) / (bbox["lat_max"] - bbox["lat_min"])


def _norm_lon(lon, bbox):
    return (lon - bbox["lon_min"]) / (bbox["lon_max"] - bbox["lon_min"])


def _encode_window(win, t0, bbox, dt_norm_s):
    """win : DataFrame d'événements (triée). Retourne [L, N_FEAT]."""
    L = len(win)
    X = np.zeros((L, N_FEAT), dtype=np.float32)
    X[:, 0] = (win["t"].to_numpy() - t0) / dt_norm_s
    X[:, 1] = _norm_lat(win["sensor_lat"].to_numpy(), bbox)
    X[:, 2] = _norm_lon(win["sensor_lon"].to_numpy(), bbox)
    mod_base = 3
    for j, m in enumerate(MODALITY_ORDER):
        X[:, mod_base + j] = (win["modality"].to_numpy() == m).astype(np.float32)
    est_base = mod_base + len(MODALITY_ORDER)
    for j, c in enumerate(CLASS_ORDER):
        X[:, est_base + j] = (win["est_class"].to_numpy() == c).astype(np.float32)
    tail = est_base + len(CLASS_ORDER)
    X[:, tail] = win["confidence"].to_numpy()
    brg = np.radians(win["bearing_est"].to_numpy())
    X[:, tail + 1] = np.sin(brg)
    X[:, tail + 2] = np.cos(brg)
    rng_est = win["range_est"].to_numpy(dtype=float)
    has = ~np.isnan(rng_est)
    X[:, tail + 3] = np.where(has, np.nan_to_num(rng_est) / RANGE_SCALE_M, 0.0)
    X[:, tail + 4] = has.astype(np.float32)
    return X


def _future_targets(gt_t, gt_lat, gt_lon, last_t, n_future, bbox):
    """Échantillonne n_future positions futures (normalisées) après `last_t`."""
    fut = np.zeros((n_future, 2), dtype=np.float32)
    fmask = np.zeros(n_future, dtype=np.float32)
    sel = gt_t > last_t
    rlat, rlon = gt_lat[sel], gt_lon[sel]
    m = len(rlat)
    if m == 0:
        return fut, fmask
    idx = (np.linspace(0, m - 1, n_future).round().astype(int) if m >= n_future
           else np.arange(m))
    for k, ix in enumerate(idx):
        fut[k] = [_norm_lat(rlat[ix], bbox), _norm_lon(rlon[ix], bbox)]
        fmask[k] = 1.0
    return fut, fmask


def build_sequences(gt_df, events_df, cfg, seed=0):
    """Construit tous les tenseurs + métadonnées. Renvoie un dict."""
    bbox = cfg["ukraine_bbox"]
    sq = cfg["sequence"]
    max_len = sq["max_len"]
    n_future = sq["n_future"]
    fracs = sq["observation_fractions"]
    dt_norm_s = sq["dt_norm_s"]

    # Vérité-terrain par drone : label cible/classe + trajectoire pour la tête future.
    gt_meta = {}
    for did, g in gt_df.groupby("drone_id"):
        g = g.sort_values("t")
        gt_meta[int(did)] = {
            "dest_id": int(g.iloc[0]["dest_id"]),
            "class_idx": CLASS_ORDER.index(g.iloc[0]["drone_class"]),
            "t": g["t"].to_numpy(),
            "lat": g["lat"].to_numpy(),
            "lon": g["lon"].to_numpy(),
        }

    X_list, mask_list = [], []
    y_target, y_class = [], []
    y_future, y_future_mask = [], []
    scen_list, frac_list = [], []

    for sid, ev in events_df.groupby("scenario_id"):
        sid = int(sid)
        if sid not in gt_meta:
            continue
        ev = ev.sort_values("t")
        n = len(ev)
        meta = gt_meta[sid]
        for f in fracs:
            cut = max(1, math.ceil(f * n))
            win = ev.iloc[:cut]
            if len(win) > max_len:
                win = win.iloc[-max_len:]   # garde les événements les plus récents
            t0 = float(win.iloc[0]["t"])
            last_t = float(win.iloc[-1]["t"])
            Xw = _encode_window(win, t0, bbox, dt_norm_s)

            X = np.zeros((max_len, N_FEAT), dtype=np.float32)
            mask = np.zeros(max_len, dtype=np.float32)
            X[: len(Xw)] = Xw
            mask[: len(Xw)] = 1.0

            fut, fmask = _future_targets(meta["t"], meta["lat"], meta["lon"],
                                         last_t, n_future, bbox)
            X_list.append(X)
            mask_list.append(mask)
            y_target.append(meta["dest_id"])
            y_class.append(meta["class_idx"])
            y_future.append(fut)
            y_future_mask.append(fmask)
            scen_list.append(sid)
            frac_list.append(f)

    # Split train/val par scénario.
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(scen_list)))
    rng.shuffle(uniq)
    n_val = int(len(uniq) * sq["val_fraction"])
    val_set = set(uniq[:n_val].tolist())
    is_val = np.array([1 if s in val_set else 0 for s in scen_list], dtype=np.int8)

    return {
        "X": np.asarray(X_list, dtype=np.float32),
        "mask": np.asarray(mask_list, dtype=np.float32),
        "y_target": np.asarray(y_target, dtype=np.int64),
        "y_class": np.asarray(y_class, dtype=np.int64),
        "y_future": np.asarray(y_future, dtype=np.float32),
        "y_future_mask": np.asarray(y_future_mask, dtype=np.float32),
        "scenario_id": np.asarray(scen_list, dtype=np.int64),
        "obs_fraction": np.asarray(frac_list, dtype=np.float32),
        "is_val": is_val,
    }


def make_meta(cfg, targets):
    bbox = cfg["ukraine_bbox"]
    return {
        "feature_names": FEATURE_NAMES,
        "n_feat": N_FEAT,
        "modality_order": MODALITY_ORDER,
        "class_order": CLASS_ORDER,
        "max_len": cfg["sequence"]["max_len"],
        "n_future": cfg["sequence"]["n_future"],
        "range_scale_m": RANGE_SCALE_M,
        "dt_norm_s": cfg["sequence"]["dt_norm_s"],
        "bbox": bbox,
        "n_targets": len(targets),
        "target_names": {t.dest_id: t.name for t in targets},
        "target_oblasts": {t.dest_id: t.oblast for t in targets},
    }


def save(out_dir, data, meta):
    npz_path = os.path.join(out_dir, "sequences.npz")
    np.savez_compressed(npz_path, **data)
    with open(os.path.join(out_dir, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return npz_path


def _cli():
    ap = argparse.ArgumentParser(description="Construit sequences.npz depuis les CSV générés.")
    ap.add_argument("--out", required=True, help="dossier contenant les CSV (et où écrire le npz)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from .routing import build_targets
    targets = build_targets(cfg)
    gt_df = pd.read_csv(os.path.join(args.out, "ground_truth_trajectories.csv"))
    events_df = pd.read_csv(os.path.join(args.out, "detection_events.csv"))
    data = build_sequences(gt_df, events_df, cfg, seed=cfg["sim"]["seed"])
    meta = make_meta(cfg, targets)
    path = save(args.out, data, meta)
    print(f"[sequence_prep] {data['X'].shape[0]} échantillons -> {path}")
    print(f"  X={data['X'].shape}  y_target classes={len(set(data['y_target']))}  "
          f"val={int(data['is_val'].sum())}")


if __name__ == "__main__":
    _cli()
