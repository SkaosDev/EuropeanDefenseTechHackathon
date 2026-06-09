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
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / "venv"
MODELS = HERE / "models"
MODEL = MODELS / "drone_yolov8x.pt"
MODEL_URL = "https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x/resolve/main/weight/best.pt"
FLYING_MODEL = MODELS / "flying_yolov8m.pt"
FLYING_URL = "https://huggingface.co/Javvanny/yolov8m_flying_objects_detection/resolve/main/yolov8m/weights/best.pt"
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

    # 4. Modèles (tous en local : flying + drone + base)
    def _download(label, dest, url, size):
        if dest.exists():
            print(f"==> Modèle {label} déjà présent : {dest}")
            return
        print(f"==> Téléchargement du modèle {label} ({size})...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"    OK -> {dest}")
        except Exception as exc:  # noqa: BLE001
            print(f"!! Téléchargement {label} échoué ({exc}). À récupérer manuellement :")
            print(f"   {url}  ->  {dest}")

    _download("flying (yolov8m, recommandé)", FLYING_MODEL, FLYING_URL, "~52 Mo")
    _download("drone (yolov8x)", MODEL, MODEL_URL, "~130 Mo")
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