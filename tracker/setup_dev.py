#!/usr/bin/env python3
"""Installation DEV (Windows / macOS) — webcam, pas de PiCam.

Crée un venv local, installe les dépendances Python et télécharge le modèle YOLO.
Sur PC il n'y a pas de picamera2 : le pipeline bascule automatiquement sur la webcam.

Usage :
    python setup_dev.py

Puis :
    Windows :  venv\\Scripts\\activate
    macOS   :  source venv/bin/activate
    python main.py --config config.dev.yaml      # tableau de bord : http://localhost:5000/
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / "venv"
MODELS = HERE / "models"
DRONE_MODEL = MODELS / "drone_yolo11n.pt"   # entraîné via training/train_drone.py (GPU)
BASE_MODEL = MODELS / "yolov8n.pt"
IS_WINDOWS = sys.platform.startswith("win")


def venv_python() -> Path:
    """Chemin de l'interpréteur Python à l'intérieur du venv."""
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(cmd: list[str]) -> None:
    """Exécute une commande en affichant ce qui tourne ; stoppe si échec."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    print(f"==> Projet : {HERE}")
    print(f"==> Plateforme : {'Windows' if IS_WINDOWS else sys.platform}")

    # 1. venv
    if VENV.exists():
        print("==> venv déjà présent, réutilisé.")
    else:
        print("==> Création du venv...")
        run([sys.executable, "-m", "venv", str(VENV)])

    py = venv_python()

    # 2. Dépendances Python (depuis requirements.txt, picamera2 exclu : inutile sur PC).
    print("==> Installation des dépendances...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(py), "-m", "pip", "install", "-r", str(HERE / "requirements.txt")])

    # 3. Dossiers de travail
    MODELS.mkdir(exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)

    # 4. Modèles (drone custom + base)
    if DRONE_MODEL.exists():
        print(f"==> Modèle drone présent : {DRONE_MODEL}")
    else:
        print(f"!! Modèle drone absent : {DRONE_MODEL}")
        print("   Entraînez-le sur une machine GPU (cf. training/README.md) puis copiez")
        print("   drone_yolo11n.pt dans models/.")
        print("   En attendant : python main.py --config config.dev.yaml --model base")
    if BASE_MODEL.exists():
        print(f"==> Modèle de base déjà présent : {BASE_MODEL}")
    else:
        print("==> Téléchargement du modèle de base yolov8n.pt (~6 Mo)...")
        try:
            run([str(py), "-c", "from ultralytics import YOLO; YOLO('yolov8n.pt')"])
            dl = HERE / "yolov8n.pt"  # ultralytics télécharge dans le cwd
            if dl.exists():
                dl.replace(BASE_MODEL)
        except subprocess.CalledProcessError:
            print("!! Téléchargement du modèle de base échoué.")

    # 5. Instructions finales
    activate = "venv\\Scripts\\activate" if IS_WINDOWS else "source venv/bin/activate"
    print("\n==> Installation DEV terminée.")
    print(f"    Activer l'env : {activate}")
    print("    Lancer        : python main.py --config config.dev.yaml")
    print("    Tableau de bord : http://localhost:5000/")
    print("    (config.dev.yaml -> camera.use_picamera: false : webcam)")


if __name__ == "__main__":
    main()