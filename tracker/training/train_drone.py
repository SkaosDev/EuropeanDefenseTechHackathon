#!/usr/bin/env python3
"""Entraînement du modèle de détection de drones — à lancer sur une machine GPU (ex. RTX 3090).

Pipeline :
  1. Télécharge le dataset Roboflow drone_bird_uav_aircraft (~30,7k images, CC BY 4.0,
     4 classes : drone / uav / bird / airplane). Clé API gratuite requise (env ROBOFLOW_API_KEY).
  2. Fusionne la classe redondante `uav` dans `drone` et réindexe les labels
     -> 3 classes finales : drone, bird, airplane (bird/airplane = anti-faux-positifs).
  3. Entraîne un YOLO11n à 640px (résolution minimale pour les petits drones).
  4. Copie le meilleur poids ici sous `drone_yolo11n.pt`.

Usage :
    pip install -r requirements-train.txt
    export ROBOFLOW_API_KEY=xxxx          # https://app.roboflow.com -> Settings -> API
    python train_drone.py                 # device auto : cuda > mps > cpu

Durées indicatives (100 epochs, 640px) : RTX 3090 ~1-2 h | Mac Apple Silicon (MPS)
~10-25 h (préférer --epochs 50) | CPU : déconseillé.

Puis copier `drone_yolo11n.pt` dans `tracker/models/` sur le Pi / la machine de dev.
Toutes les étapes sont idempotentes : relancer reprend là où ça s'est arrêté.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
OUTPUT_WEIGHTS = HERE / "drone_yolo11n.pt"

ROBOFLOW_WORKSPACE = "donia-mceky"
ROBOFLOW_PROJECT = "drone_bird_uav_aircraft-pz6zp"

# Classes finales, dans cet ordre (drone = id 0). `uav` est fusionné dans `drone` :
# la distinction quadri/aile-fixe n'apporte rien au tracker et divise les exemples.
FINAL_CLASSES = ["drone", "bird", "airplane"]
MERGE_INTO_DRONE = {"uav", "drone"}


def download_dataset(version: int | None) -> Path:
    """Télécharge l'export YOLOv8 du dataset Roboflow (skip si déjà présent)."""
    existing = list(DATASET_DIR.glob("*/data.yaml"))
    if existing:
        print(f"==> Dataset déjà présent : {existing[0].parent}")
        return existing[0].parent

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ROBOFLOW_API_KEY manquante.\n"
            "  1. Compte gratuit sur https://app.roboflow.com\n"
            "  2. Settings -> API -> Private API Key\n"
            "  3. export ROBOFLOW_API_KEY=xxxx puis relancer."
        )

    from roboflow import Roboflow  # import tardif : pas requis si dataset déjà là

    project = Roboflow(api_key=api_key).workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    if version is None:
        version = max(int(str(v.version).rsplit("/", 1)[-1]) for v in project.versions())
        print(f"==> Dernière version du dataset : {version}")
    DATASET_DIR.mkdir(exist_ok=True)
    print("==> Téléchargement (~quelques Go, une seule fois)...")
    ds = project.version(version).download("yolov8", location=str(DATASET_DIR / f"v{version}"))
    return Path(ds.location)


def _load_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):  # format {0: "x", 1: "y"} possible selon l'export
        names = [names[k] for k in sorted(names)]
    return [str(n).lower() for n in names]


def remap_classes(ds_root: Path) -> None:
    """Fusionne uav->drone et réindexe les labels vers FINAL_CLASSES (idempotent)."""
    data_yaml = ds_root / "data.yaml"
    names = _load_names(data_yaml)
    if names == FINAL_CLASSES:
        print("==> Labels déjà remappés (drone/bird/airplane), rien à faire.")
        return

    print(f"==> Remap des classes : {names} -> {FINAL_CLASSES}")
    id_map: dict[int, int] = {}
    for old_id, name in enumerate(names):
        target = "drone" if name in MERGE_INTO_DRONE else name
        if target not in FINAL_CLASSES:
            sys.exit(f"Classe inattendue dans le dataset : '{name}' — vérifier data.yaml.")
        id_map[old_id] = FINAL_CLASSES.index(target)

    before: Counter[int] = Counter()
    after: Counter[int] = Counter()
    label_files = list(ds_root.glob("*/labels/*.txt"))
    if not label_files:
        sys.exit(f"Aucun fichier label trouvé sous {ds_root}/*/labels/ — export corrompu ?")
    for lf in label_files:
        lines_out = []
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            old_id = int(parts[0])
            before[old_id] += 1
            parts[0] = str(id_map[old_id])
            after[id_map[old_id]] += 1
            lines_out.append(" ".join(parts))
        lf.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")

    assert sum(before.values()) == sum(after.values()), "Perte d'annotations pendant le remap !"
    print(f"    {len(label_files)} fichiers labels réécrits, {sum(after.values())} annotations :")
    for new_id, cls in enumerate(FINAL_CLASSES):
        print(f"      {cls:10s} : {after.get(new_id, 0)}")

    # data.yaml : 3 classes + chemins absolus (évite les soucis de cwd d'ultralytics).
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg["nc"] = len(FINAL_CLASSES)
    cfg["names"] = FINAL_CLASSES
    cfg["path"] = str(ds_root)
    for split in ("train", "val", "test"):
        if (ds_root / split / "images").is_dir():
            cfg[split] = f"{split}/images"
        elif split == "val" and (ds_root / "valid" / "images").is_dir():
            cfg["val"] = "valid/images"  # Roboflow nomme le split "valid"
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"    data.yaml réécrit : {data_yaml}")


def resolve_device(requested: str) -> str:
    """'auto' -> cuda (RTX) > mps (Apple Silicon) > cpu ; sinon valeur demandée."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        print(f"==> Device : CUDA ({torch.cuda.get_device_name(0)})")
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("==> Device : MPS (Apple Silicon) — comptez plusieurs heures.")
        print("    Astuce : --epochs 50 pour un premier modèle utilisable plus vite.")
        return "mps"
    print("==> Device : CPU — TRÈS lent (~plusieurs jours). GPU fortement conseillé.")
    return "cpu"


def train(ds_root: Path, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    device = resolve_device(args.device)
    batch = args.batch
    if batch == -1 and device in ("cpu", "mps"):
        batch = 16  # l'auto-batch (-1) mesure la VRAM : CUDA uniquement

    model = YOLO(args.model)
    results = model.train(
        data=str(ds_root / "data.yaml"),
        imgsz=args.imgsz,           # 640 mini : les drones lointains sont minuscules
        epochs=args.epochs,
        batch=batch,
        device=device,
        patience=20,                # early stopping si la val stagne
        cache="disk",
        project=str(HERE / "runs"),
        name="drone_yolo11n",
        exist_ok=True,              # reprend le même dossier en cas de relance
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, OUTPUT_WEIGHTS)
    metrics = model.val(data=str(ds_root / "data.yaml"))
    print("\n==> Entraînement terminé.")
    print(f"    mAP50 : {metrics.box.map50:.3f}   mAP50-95 : {metrics.box.map:.3f}")
    for i, cls in enumerate(FINAL_CLASSES):
        if i < len(metrics.box.maps):
            print(f"      {cls:10s} mAP50-95 : {metrics.box.maps[i]:.3f}")
    print(f"\n    Poids : {OUTPUT_WEIGHTS}")
    print("    -> à copier dans tracker/models/drone_yolo11n.pt")
    print("    (sur le Pi, setup.sh fera ensuite l'export NCNN automatiquement)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement YOLO drone (GPU recommandé).")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Poids de départ (yolo11n.pt ; essayer yolo26n.pt si dispo).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 = auto selon la VRAM (CUDA) ; 16 par défaut sur mps/cpu.")
    parser.add_argument("--device", default="auto",
                        help="auto (défaut : cuda > mps > cpu), '0' (GPU NVIDIA), 'mps', 'cpu'.")
    parser.add_argument("--version", type=int, default=None,
                        help="Version Roboflow du dataset (défaut : la plus récente).")
    args = parser.parse_args()

    ds_root = download_dataset(args.version)
    remap_classes(ds_root)
    train(ds_root, args)


if __name__ == "__main__":
    main()
