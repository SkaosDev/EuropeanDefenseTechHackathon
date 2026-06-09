#!/usr/bin/env bash
# ============================================================================
#  Installation complète du système de tracking sur Raspberry Pi OS Bookworm.
#  Usage : bash setup.sh [--service]
#    --service : génère et active en plus un service systemd (démarrage auto).
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/venv"
PY="$VENV/bin/python"
MODEL="$HERE/models/drone_yolov8x.pt"
MODEL_URL="https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x/resolve/main/weight/best.pt"
FLYING_MODEL="$HERE/models/flying_yolov8m.pt"
FLYING_URL="https://huggingface.co/Javvanny/yolov8m_flying_objects_detection/resolve/main/yolov8m/weights/best.pt"
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
echo "==> Installation des paquets système (sudo requis)..."
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-dev \
  python3-picamera2 libcamera-apps \
  libatlas-base-dev libopenjp2-7 ffmpeg || \
  echo "!! Certains paquets n'ont pas pu être installés (machine de dev ?). On continue."

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

# --- 5. Modèles (tous en local : flying + drone + base) ---------------------
mkdir -p "$HERE/models" "$HERE/logs"
# 5a0. Modèle flying YOLOv8m (~52 Mo, recommandé : rapide + détecte drones)
if [[ -f "$FLYING_MODEL" ]]; then
  echo "==> Modèle flying déjà présent : $FLYING_MODEL"
else
  echo "==> Téléchargement du modèle flying YOLOv8m (~52 Mo)..."
  wget -q --show-progress -O "$FLYING_MODEL" "$FLYING_URL" || \
    curl -fL -o "$FLYING_MODEL" "$FLYING_URL" || \
    echo "!! Échec du téléchargement flying : $FLYING_URL"
fi
# 5a. Modèle drone YOLOv8x (~130 Mo)
if [[ -f "$MODEL" ]]; then
  echo "==> Modèle drone déjà présent : $MODEL"
else
  echo "==> Téléchargement du modèle drone YOLOv8x (~130 Mo) depuis HuggingFace..."
  wget -q --show-progress -O "$MODEL" "$MODEL_URL" || \
    curl -fL -o "$MODEL" "$MODEL_URL" || {
      echo "!! Échec du téléchargement drone. Récupérez best.pt manuellement :"
      echo "   $MODEL_URL  ->  $MODEL"; }
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