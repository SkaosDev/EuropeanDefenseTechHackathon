#!/usr/bin/env bash
# ============================================================================
#  Installation complète du système de tracking sur Raspberry Pi OS Bookworm.
#  Usage : bash setup.sh [--service]
#    --service : génère et active en plus un service systemd (démarrage auto).
# ============================================================================

# Ce script utilise des bashismes ([[ ]], ${BASH_SOURCE}…). Si on l'a lancé via
# `sh setup.sh` (dash sur Pi OS), on se relance automatiquement sous bash.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/venv"
PY="$VENV/bin/python"
DRONE_MODEL="$HERE/models/drone_yolo11n.pt"     # entraîné via training/train_drone.py (GPU)
NCNN_DIR="$HERE/models/drone_yolo11n_ncnn_model"
BASE_MODEL="$HERE/models/yolov8n.pt"
MAKE_SERVICE=0
[[ "${1:-}" == "--service" ]] && MAKE_SERVICE=1

echo "==> Dossier projet : $HERE"

# --- 1. Vérification Raspberry Pi OS ---------------------------------------
if grep -qiE "raspbian|raspberry" /etc/os-release 2>/dev/null || [[ -f /sys/firmware/devicetree/base/model ]]; then
  echo "==> Raspberry Pi détecté."
else
  echo "!! Attention : ce système ne ressemble pas à un Raspberry Pi OS."
  echo "   L'installation continue (mode dev), mais picamera2 ne sera pas fonctionnel."
fi

# --- 2. Dépendances système (libcamera, picamera2 via apt) -----------------
# Installés UN PAR UN : les noms diffèrent entre Bookworm et trixie (Pi OS récent) et un
# paquet absent/renommé ne doit pas faire échouer toute l'installation.
#   libcamera-apps  -> rpicam-apps      (trixie)
#   libatlas-base-dev -> libopenblas-dev (trixie ; ATLAS retiré)
echo "==> Installation des paquets système (sudo requis)..."
sudo apt-get update || true
SYS_PKGS="python3-venv python3-pip python3-dev \
  python3-picamera2 libcamera-apps rpicam-apps \
  libopenblas-dev libatlas-base-dev libopenjp2-7 ffmpeg"
for pkg in $SYS_PKGS; do
  if sudo apt-get install -y "$pkg" >/dev/null 2>&1; then
    echo "   ok       : $pkg"
  else
    echo "   ignoré   : $pkg (indisponible sur cette distrib — normal pour un équivalent renommé)"
  fi
done

# --- 3. Environnement virtuel ----------------------------------------------
# --system-site-packages : indispensable pour voir picamera2 (installé par apt).
if [[ ! -d "$VENV" ]]; then
  echo "==> Création du venv (--system-site-packages)..."
  python3 -m venv --system-site-packages "$VENV"
fi

# --- 4. Dépendances Python --------------------------------------------------
echo "==> Installation des dépendances Python..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$HERE/requirements.txt"

# --- 5. Modèles (drone custom + base) ----------------------------------------
mkdir -p "$HERE/models" "$HERE/logs"
# 5a. Modèle drone YOLO11n custom (entraîné sur GPU, cf. training/README.md)
if [[ -f "$DRONE_MODEL" ]]; then
  echo "==> Modèle drone présent : $DRONE_MODEL"
  # Export NCNN (~2x plus rapide qu'ONNX sur ARM) — chargé en priorité par le détecteur.
  if [[ -d "$NCNN_DIR" ]]; then
    echo "==> Export NCNN déjà présent : $NCNN_DIR"
  else
    echo "==> Export NCNN du modèle drone (quelques minutes sur Pi)..."
    "$PY" -m pip install ncnn
    "$PY" -c "from ultralytics import YOLO; YOLO('$DRONE_MODEL').export(format='ncnn', imgsz=640)" || \
      echo "!! Export NCNN échoué — le tracker utilisera le .pt (plus lent)."
  fi
else
  echo "!! Modèle drone absent : $DRONE_MODEL"
  echo "   Entraînez-le sur une machine GPU (cf. training/README.md) puis copiez"
  echo "   drone_yolo11n.pt dans models/ et relancez setup.sh pour l'export NCNN."
  echo "   En attendant : python main.py --model base"
fi
# 5b. Modèle de base YOLOv8n COCO (~6 Mo, rapide)
if [[ -f "$BASE_MODEL" ]]; then
  echo "==> Modèle de base déjà présent : $BASE_MODEL"
else
  echo "==> Téléchargement du modèle de base yolov8n.pt..."
  "$PY" -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null
  [[ -f "$HERE/yolov8n.pt" ]] && mv "$HERE/yolov8n.pt" "$BASE_MODEL"
  [[ -f "$BASE_MODEL" ]] || wget -q -O "$BASE_MODEL" \
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" || \
    echo "!! Échec du téléchargement du modèle de base."
fi

# --- 6. Service systemd (optionnel) ----------------------------------------
if [[ "$MAKE_SERVICE" == "1" ]]; then
  echo "==> Génération du service systemd 'tracker'..."
  SERVICE=/etc/systemd/system/tracker.service
  sudo tee "$SERVICE" >/dev/null <<EOF
[Unit]
Description=Systeme de tracking embarque (YOLOv8 + ByteTrack)
After=network.target

[Service]
Type=simple
WorkingDirectory=$HERE
ExecStart=$PY $HERE/main.py --config $HERE/config.yaml
Restart=on-failure
RestartSec=3
User=$USER

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable tracker.service
  echo "   Service installé. Démarrer avec : sudo systemctl start tracker"
fi

echo ""
echo "==> Installation terminée."
echo "    Activer l'env  : source venv/bin/activate"
echo "    Lancer (démo)  : python main.py --headless"
echo "    Flux MJPEG     : http://<ip-du-pi>:5000/"