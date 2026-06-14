"""
train.py — entraînement offline du LSTM multi-têtes (CPU, quelques minutes).

  python -m model.train [--epochs 50] [--batch 128] [--lr 1e-3] [--lam 0.2] [--seed 0]
                        [--proj 64] [--hidden 128] [--layers 3] [--dropout 0.3]
                        [--lstm-dropout 0.2] [--out dataset_generator/out]
  (l'architecture est enregistrée dans le checkpoint -> l'inférence la reprend telle quelle.)

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
import time

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
          proj=64, hidden=128, n_layers=3, dropout=0.3, lstm_dropout=0.2,
          seed=0, log=None):
    # logger qui FLUSH à chaque ligne (sinon les logs sont bufferisés quand stdout est
    # redirigé vers un fichier -> on croit à tort que l'entraînement est figé).
    log = log or (lambda *a: print(*a, flush=True))
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

    arch = dict(n_feat=n_feat, proj=proj, hidden=hidden, n_layers=n_layers, bidir=False,
                n_targets=n_targets, n_classes=n_classes, n_future=n_future,
                dropout=dropout, lstm_dropout=lstm_dropout)
    model = DroneNet(**arch)

    class_w = _class_weights(d["y_class"][tr], n_classes)
    ce_target = nn.CrossEntropyLoss(label_smoothing=0.05)
    ce_class = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_top3 = -1.0
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    n_batches = len(train_loader)
    log(f"[train] {n_batches} batches/epoch × {epochs} epochs — démarrage…")
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        t0 = time.perf_counter()
        for bi, (X, mask, yt, yc, yf, yfm) in enumerate(train_loader):
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
            if n_batches >= 8 and (bi + 1) % max(1, n_batches // 4) == 0:
                log(f"    ep {ep:02d} · batch {bi + 1}/{n_batches}…")
        m = evaluate(model, val_loader, bbox)
        dt = time.perf_counter() - t0
        flag = "  ★ best" if m["top3"] > best_top3 else ""
        log(f"  ep {ep:02d} | {dt:4.1f}s | loss {run / int(tr.sum()):.3f} "
            f"| target top1 {m['top1']:.3f} top3 {m['top3']:.3f} "
            f"| class {m['cls']:.3f} | future {m['fut_m'] / 1000:.1f} km{flag}")
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
    ap.add_argument("--out", default=DEFAULT_OUT, help="dossier dataset (sequences.npz + meta)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128, help="taille de batch (train)")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate (AdamW)")
    ap.add_argument("--lam", type=float, default=0.2, help="poids MSE trajectoire future")
    ap.add_argument("--seed", type=int, default=0, help="graine torch (reproductibilité)")
    # --- architecture (sauvegardée dans le checkpoint -> reprise auto à l'inférence) ---
    ap.add_argument("--proj", type=int, default=64, help="dim. projection d'entrée")
    ap.add_argument("--hidden", type=int, default=128, help="dim. cachée LSTM")
    ap.add_argument("--layers", type=int, default=3, dest="n_layers", help="nb couches LSTM")
    ap.add_argument("--dropout", type=float, default=0.3, help="dropout des têtes")
    ap.add_argument("--lstm-dropout", type=float, default=0.2, dest="lstm_dropout",
                    help="dropout inter-couches LSTM")
    args = ap.parse_args()
    train(out_dir=args.out, epochs=args.epochs, batch=args.batch, lr=args.lr, lam=args.lam,
          seed=args.seed, proj=args.proj, hidden=args.hidden, n_layers=args.n_layers,
          dropout=args.dropout, lstm_dropout=args.lstm_dropout)


if __name__ == "__main__":
    _cli()
