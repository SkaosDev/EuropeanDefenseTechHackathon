"""
train.py — entraînement offline du LSTM multi-têtes (CPU, quelques minutes).

  python -m model.train [--out dataset_generator/out] [--epochs 50]

Charge sequences.npz + dataset_meta.json, split STRICTEMENT via `is_val` (anti-fuite),
optimise CE(target, label_smoothing) + 0.3*CE(class, class_weight) + λ*MSE(future, masquée),
logue top-1/top-3 cible + accuracy classe + MSE future en mètres, et sauvegarde les poids
(+ une copie de dataset_meta.json) dans model/weights/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .net import DroneNet

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(REPO, "dataset_generator", "out")
WEIGHTS_DIR = os.path.join(HERE, "weights")


def _split_tensors(d, split):
    return TensorDataset(
        torch.from_numpy(d["X"][split]).float(),
        torch.from_numpy(d["mask"][split]).float(),
        torch.from_numpy(d["y_target"][split]).long(),
        torch.from_numpy(d["y_class"][split]).long(),
        torch.from_numpy(d["y_future"][split]).float(),
        torch.from_numpy(d["y_future_mask"][split]).float(),
    )


def _class_weights(y_class_train, n_classes):
    counts = np.bincount(y_class_train, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)   # inverse-fréquence normalisée
    return torch.from_numpy(w).float()


def evaluate(model, loader, bbox):
    model.eval()
    tot = top1 = top3 = ccorrect = 0
    fut_err_sum = 0.0
    fut_n = 0.0
    lat_r, lat_min = bbox["lat_max"] - bbox["lat_min"], bbox["lat_min"]
    lon_r, lon_min = bbox["lon_max"] - bbox["lon_min"], bbox["lon_min"]
    with torch.no_grad():
        for X, mask, yt, yc, yf, yfm in loader:
            tl, cl, fut = model(X, mask)
            top = tl.topk(3, dim=1).indices
            top1 += (top[:, 0] == yt).sum().item()
            top3 += (top == yt.unsqueeze(1)).any(dim=1).sum().item()
            ccorrect += (cl.argmax(1) == yc).sum().item()
            tot += yt.size(0)
            # erreur trajectoire future en mètres (sur les points valides)
            pred_lat = fut[..., 0] * lat_r + lat_min
            pred_lon = fut[..., 1] * lon_r + lon_min
            true_lat = yf[..., 0] * lat_r + lat_min
            true_lon = yf[..., 1] * lon_r + lon_min
            dlat = (pred_lat - true_lat) * 111320.0
            dlon = (pred_lon - true_lon) * 111320.0 * torch.cos(torch.deg2rad(true_lat))
            dist = torch.sqrt(dlat ** 2 + dlon ** 2)
            fut_err_sum += (dist * yfm).sum().item()
            fut_n += yfm.sum().item()
    return {
        "top1": top1 / tot, "top3": top3 / tot, "cls": ccorrect / tot,
        "fut_m": fut_err_sum / max(1.0, fut_n),
    }


def train(out_dir=DEFAULT_OUT, epochs=50, batch=128, lr=1e-3, lam=0.2,
          seed=0, log=print):
    torch.manual_seed(seed)
    d = np.load(os.path.join(out_dir, "sequences.npz"))
    meta = json.load(open(os.path.join(out_dir, "dataset_meta.json")))
    bbox = meta["bbox"]
    n_targets = meta["n_targets"]
    n_classes = len(meta["class_order"])
    n_future = meta["n_future"]
    n_feat = meta["n_feat"]

    tr = d["is_val"] == 0
    va = d["is_val"] == 1
    train_loader = DataLoader(_split_tensors(d, tr), batch_size=batch, shuffle=True)
    val_loader = DataLoader(_split_tensors(d, va), batch_size=256, shuffle=False)
    log(f"[train] {int(tr.sum())} train / {int(va.sum())} val  "
        f"| targets={n_targets} classes={n_classes} future={n_future}")

    arch = dict(n_feat=n_feat, proj=64, hidden=128, n_layers=2, bidir=False,
                n_targets=n_targets, n_classes=n_classes, n_future=n_future,
                dropout=0.3, lstm_dropout=0.2)
    model = DroneNet(**arch)

    class_w = _class_weights(d["y_class"][tr], n_classes)
    ce_target = nn.CrossEntropyLoss(label_smoothing=0.05)
    ce_class = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_top3 = -1.0
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for X, mask, yt, yc, yf, yfm in train_loader:
            opt.zero_grad()
            tl, cl, fut = model(X, mask)
            loss_t = ce_target(tl, yt)
            loss_c = ce_class(cl, yc)
            # MSE future masquée : moyenne sur les points valides
            sq = ((fut - yf) ** 2).sum(dim=-1)          # [B, 12]
            loss_f = (sq * yfm).sum() / yfm.sum().clamp(min=1.0)
            loss = loss_t + 0.3 * loss_c + lam * loss_f
            loss.backward()
            opt.step()
            run += loss.item() * yt.size(0)
        m = evaluate(model, val_loader, bbox)
        log(f"  ep {ep:02d} | loss {run / int(tr.sum()):.3f} "
            f"| target top1 {m['top1']:.3f} top3 {m['top3']:.3f} "
            f"| class {m['cls']:.3f} | future {m['fut_m'] / 1000:.1f} km")
        if m["top3"] > best_top3:
            best_top3 = m["top3"]
            torch.save({"state_dict": model.state_dict(), "arch": arch,
                        "val_metrics": m}, os.path.join(WEIGHTS_DIR, "model.pt"))
    shutil.copy(os.path.join(out_dir, "dataset_meta.json"),
                os.path.join(WEIGHTS_DIR, "dataset_meta.json"))
    log(f"[train] best val top3={best_top3:.3f} -> {os.path.join(WEIGHTS_DIR, 'model.pt')}")
    return best_top3


def _cli():
    ap = argparse.ArgumentParser(description="Entraîne le LSTM multi-têtes.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.2, help="poids MSE trajectoire future")
    args = ap.parse_args()
    train(out_dir=args.out, epochs=args.epochs, batch=args.batch, lr=args.lr, lam=args.lam)


if __name__ == "__main__":
    _cli()
