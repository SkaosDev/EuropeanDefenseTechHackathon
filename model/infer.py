"""
infer.py — inférence avec PARITÉ d'encodage train/serve.

On RÉUTILISE `dataset_generator.sequence_prep._encode_window` (mêmes 17 features, mêmes
normalisations, même règle de troncature aux 64 derniers événements). Aucune
réimplémentation : c'est non négociable.

  from model.infer import Predictor
  pred = Predictor().predict(events, clock_t=T)
  # -> {target_topk: [{dest_id,name,p}...], pred_class, pred_class_p, pred_future:[[lat,lon]x12]}

CLI de validation P1 :
  python -m model.infer [--scenario-id N]   # rejoue un scénario de detection_events.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from dataset_generator.sequence_prep import N_FEAT, _encode_window
from .net import DroneNet

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)
WEIGHTS_DIR = os.path.join(HERE, "weights")
DEFAULT_OUT = os.path.join(REPO, "dataset_generator", "out")


class Predictor:
    def __init__(self, weights_dir=WEIGHTS_DIR, device="cpu"):
        with open(os.path.join(weights_dir, "dataset_meta.json")) as f:
            meta = json.load(f)
        self.meta = meta
        self.bbox = meta["bbox"]
        self.dt_norm_s = meta["dt_norm_s"]
        self.max_len = meta["max_len"]
        self.class_order = meta["class_order"]
        self.n_future = meta["n_future"]
        self.target_names = {int(k): v for k, v in meta["target_names"].items()}
        ckpt = torch.load(os.path.join(weights_dir, "model.pt"), map_location=device,
                          weights_only=True)
        self.model = DroneNet(**ckpt["arch"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.device = device

    def _encode(self, events, clock_t=None):
        """events: liste de dicts (clutter inclus). Renvoie (X[64,17], mask[64]) ou (None,None)."""
        if not len(events):
            return None, None
        df = pd.DataFrame(events)
        if clock_t is not None:
            df = df[df["t"] <= clock_t]
        if len(df) == 0:
            return None, None
        df = df.sort_values("t")
        if len(df) > self.max_len:
            df = df.iloc[-self.max_len:]           # garde les plus récents (parité train)
        t0 = float(df.iloc[0]["t"])
        Xw = _encode_window(df, t0, self.bbox, self.dt_norm_s)
        X = np.zeros((self.max_len, N_FEAT), dtype=np.float32)
        mask = np.zeros(self.max_len, dtype=np.float32)
        X[: len(Xw)] = Xw
        mask[: len(Xw)] = 1.0
        return X, mask

    def _denorm_future(self, fut_norm):
        b = self.bbox
        lat_r, lon_r = b["lat_max"] - b["lat_min"], b["lon_max"] - b["lon_min"]
        return [[float(la * lat_r + b["lat_min"]), float(lo * lon_r + b["lon_min"])]
                for la, lo in fut_norm]

    @torch.no_grad()
    def predict(self, events, clock_t=None, topk=5):
        X, mask = self._encode(events, clock_t)
        if X is None:
            return None
        xt = torch.from_numpy(X).unsqueeze(0).to(self.device)
        mt = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        tl, cl, fut = self.model(xt, mt)
        tprob = torch.softmax(tl, dim=1)[0].cpu().numpy()
        cprob = torch.softmax(cl, dim=1)[0].cpu().numpy()
        top_idx = tprob.argsort()[::-1][:topk]
        target_topk = [
            {"dest_id": int(i), "name": self.target_names.get(int(i), str(i)),
             "p": float(tprob[i])}
            for i in top_idx
        ]
        cls_idx = int(cprob.argmax())
        return {
            "target_topk": target_topk,
            "pred_class": self.class_order[cls_idx],
            "pred_class_p": float(cprob[cls_idx]),
            "pred_future": self._denorm_future(fut[0].cpu().numpy()),
        }


def _cli():
    ap = argparse.ArgumentParser(description="Rejoue un scénario et affiche la prédiction.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scenario-id", type=int, default=None)
    args = ap.parse_args()

    events_df = pd.read_csv(os.path.join(args.out, "detection_events.csv"))
    gt_df = pd.read_csv(os.path.join(args.out, "ground_truth_trajectories.csv"),
                        usecols=["drone_id", "dest_id", "dest_name", "drone_class"])
    sid = args.scenario_id
    if sid is None:
        sid = int(events_df["scenario_id"].sample(1).iloc[0])
    ev = events_df[events_df["scenario_id"] == sid].sort_values("t")
    events = ev.to_dict("records")
    truth = gt_df[gt_df["drone_id"] == sid].iloc[0] if (gt_df["drone_id"] == sid).any() else None

    pred = Predictor()
    res = pred.predict(events)
    print(f"=== scénario {sid} : {len(events)} événements ===")
    if truth is not None:
        print(f"  VÉRITÉ  : cible={truth['dest_name']} (id {truth['dest_id']}), "
              f"classe={truth['drone_class']}")
    print(f"  CLASSE  : {res['pred_class']} (p={res['pred_class_p']:.2f})")
    print("  TOP-5 CIBLES :")
    for r in res["target_topk"]:
        flag = "  <-- vraie" if truth is not None and r["dest_id"] == truth["dest_id"] else ""
        print(f"    {r['p']*100:5.1f}%  {r['name']} (id {r['dest_id']}){flag}")


if __name__ == "__main__":
    _cli()
