"""
eval_fractions.py — accuracy cible/classe ventilée par taux d'observation.

C'est le cœur narratif de la démo : la prédiction doit se RESSERRER quand le drone
avance (plus d'événements observés). On évalue le modèle sauvegardé sur le split val,
groupé par `obs_fraction` (0.25 / 0.5 / 0.75 / 1.0).

  python -m model.eval_fractions
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .infer import DEFAULT_OUT, WEIGHTS_DIR
from .net import DroneNet


def main(out_dir=DEFAULT_OUT, weights_dir=WEIGHTS_DIR):
    d = np.load(os.path.join(out_dir, "sequences.npz"))
    ckpt = torch.load(os.path.join(weights_dir, "model.pt"), map_location="cpu",
                      weights_only=True)
    model = DroneNet(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    va = d["is_val"] == 1
    X = torch.from_numpy(d["X"][va]).float()
    mask = torch.from_numpy(d["mask"][va]).float()
    yt = torch.from_numpy(d["y_target"][va]).long()
    yc = torch.from_numpy(d["y_class"][va]).long()
    frac = d["obs_fraction"][va]

    with torch.no_grad():
        tl, cl, _ = model(X, mask)
        top = tl.topk(3, dim=1).indices
        t1 = (top[:, 0] == yt).numpy()
        t3 = (top == yt.unsqueeze(1)).any(dim=1).numpy()
        cc = (cl.argmax(1) == yc).numpy()

    print(f"  global   | top1 {t1.mean():.3f}  top3 {t3.mean():.3f}  class {cc.mean():.3f}")
    for f in sorted(set(frac.tolist())):
        s = frac == f
        print(f"  obs {f:.2f} | top1 {t1[s].mean():.3f}  top3 {t3[s].mean():.3f}  "
              f"class {cc[s].mean():.3f}  (n={int(s.sum())})")


if __name__ == "__main__":
    main()
