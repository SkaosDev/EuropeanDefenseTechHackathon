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


def configure_apple_silicon() -> bool:
    """Règle l'environnement MPS AVANT tout import de torch (sinon ignoré).

    Renvoie True si on tourne sur un Mac Apple Silicon. Ne touche à rien ailleurs.
    Rappel : la puce neuronale (ANE) n'est PAS utilisable pour l'entraînement
    (CoreML/inférence seulement). On maximise donc GPU (MPS) + CPU (dataloader).
    """
    if sys.platform != "darwin":
        return False
    # Les ops non implémentées en MPS retombent sur le CPU au lieu de crasher.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Threads CPU pour le décodage/redimensionnement des images (data-loading).
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
    return True

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
        print("==> Device : MPS (GPU Apple Silicon) — mode Mac optimisé activé.")
        return "mps"
    print("==> Device : CPU — TRÈS lent (~plusieurs jours). GPU fortement conseillé.")
    return "cpu"


def resolve_mac_settings(args: argparse.Namespace) -> dict:
    """Réglages saturant un Mac Apple Silicon (M-series) en MPS.

    Objectif : ~1 s/it en exploitant le GPU + les cœurs CPU + la mémoire unifiée.
      - batch : 32 par défaut (≈8-10 Go sur 24 Go) ; --fast pousse à 48.
      - workers : tous les cœurs CPU pour que le dataloader ne bride jamais le GPU.
      - cache : 'disk' par défaut (sûr) ; 'ram' interdit ici car ~30k images
        décodées > 24 Go -> gèlerait le Mac. Forçable via --cache ram à tes risques.
    """
    cpu = os.cpu_count() or 8
    workers = args.workers if args.workers and args.workers > 0 else cpu
    batch = args.batch
    if batch == -1:  # -1 = auto : l'auto-batch VRAM d'ultralytics est CUDA-only
        batch = 48 if args.fast else 32
    imgsz = args.imgsz
    if args.fast and imgsz == 640:
        # 512px : ~1.5x plus rapide. Compromis : drones très lointains plus durs
        # à voir. Garde 640 si la portée max compte plus que la vitesse.
        imgsz = 512
        print("    --fast : imgsz 640 -> 512 (plus rapide, un peu moins de portée).")
    return {"batch": batch, "workers": workers, "imgsz": imgsz}


def train(ds_root: Path, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    device = resolve_device(args.device)

    if device == "mps":
        cfg = resolve_mac_settings(args)
        batch, workers, imgsz = cfg["batch"], cfg["workers"], cfg["imgsz"]
    else:
        imgsz = args.imgsz
        batch = args.batch if args.batch != -1 else (16 if device == "cpu" else args.batch)
        workers = args.workers if args.workers and args.workers > 0 else 8

    if device == "mps":
        print(f"==> Mac MPS : batch={batch}, workers={workers}, imgsz={imgsz}, "
              f"cache={args.cache} (24 Go unifiés exploités).")

    model = YOLO(args.model)
    results = model.train(
        data=str(ds_root / "data.yaml"),
        imgsz=imgsz,                # 640 mini : les drones lointains sont minuscules
        epochs=args.epochs,
        batch=batch,
        device=device,
        workers=workers,            # CPU à fond pour le data-loading -> GPU jamais à l'arrêt
        cache=args.cache,           # 'disk' : décode une fois, relit sans re-décoder
        amp=True,                   # demi-précision : ~1.5-2x plus rapide en MPS
        fraction=args.fraction,     # <1.0 = sous-échantillonne (itération rapide)
        patience=20,                # early stopping si la val stagne
        project=args.project or str(HERE / "runs"),  # Drive sur Colab = survit aux coupures
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
    configure_apple_silicon()  # règle l'env MPS avant tout import de torch
    parser = argparse.ArgumentParser(description="Entraînement YOLO drone (GPU recommandé).")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Poids de départ (yolo11n.pt ; essayer yolo26n.pt si dispo).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 = auto (CUDA: VRAM ; Mac MPS: 32, ou 48 avec --fast). "
                             "Si OOM/gel sur Mac, baisse (ex. --batch 24).")
    parser.add_argument("--device", default="auto",
                        help="auto (défaut : cuda > mps > cpu), '0' (GPU NVIDIA), 'mps', 'cpu'.")
    parser.add_argument("--workers", type=int, default=-1,
                        help="Threads dataloader (-1 = tous les cœurs CPU). Mac M3 = 8.")
    parser.add_argument("--cache", default="disk", choices=["disk", "ram", "False"],
                        help="disk (défaut, sûr) ; ram = plus rapide mais risque OOM "
                             "sur Mac 24 Go avec ~30k images.")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction du train à utiliser (<1.0 pour itérer vite, ex. 0.3).")
    parser.add_argument("--fast", action="store_true",
                        help="Mac : preset vitesse (batch 48 + imgsz 512). "
                             "Plus rapide, portée drone un peu réduite.")
    parser.add_argument("--version", type=int, default=None,
                        help="Version Roboflow du dataset (défaut : la plus récente).")
    parser.add_argument("--project", default=None,
                        help="Dossier des runs (défaut : ./runs). Sur Colab gratuit, "
                             "pointe-le vers Drive pour que best.pt survive aux déconnexions.")
    args = parser.parse_args()
    if args.cache == "False":
        args.cache = False

    ds_root = download_dataset(args.version)
    remap_classes(ds_root)
    train(ds_root, args)


if __name__ == "__main__":
    main()
