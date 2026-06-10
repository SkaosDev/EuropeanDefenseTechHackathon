# Entraînement du modèle drone (YOLO11n)

Produit `drone_yolo11n.pt`, le modèle anti-drone du tracker : **3 classes**
(`drone`, `bird`, `airplane` — les deux dernières servent de distracteurs pour
éviter de confondre un oiseau ou un avion avec un drone).

- **Dataset** : [drone_bird_uav_aircraft (Roboflow, donia-mceky)](https://universe.roboflow.com/donia-mceky/drone_bird_uav_aircraft-pz6zp)
  — ~30 700 images, CC BY 4.0. La classe `uav` (redondante) est fusionnée dans `drone`.
- **Modèle** : YOLO11n (~5,4 Mo) — assez léger pour tourner à ~10-15 fps en NCNN sur Pi 5.
- **Machine** : un GPU est fortement recommandé. Sur RTX 3090 : **~1-2 h** (100 epochs, 640px).

## Étapes (sur la machine GPU)

```bash
# 1. Récupérer ce dossier (git clone ou copie de tracker/training/)


# 2. Dépendances (venv conseillé)
python -m venv venv && source venv/bin/activate
pip install -r requirements-train.txt

# 3. Clé API Roboflow (compte gratuit : https://app.roboflow.com -> Settings -> API)
export ROBOFLOW_API_KEY=xxxxxxxxxxxx

# 4. Lancer (download dataset ~qq Go la 1re fois, puis training)
python train_drone.py
```

Le device est **auto-détecté** : CUDA (RTX) > MPS (Mac Apple Silicon) > CPU.
Le même script fonctionne donc tel quel sur la 3090 (~1-2 h) comme sur un Mac
(~10-25 h en MPS — utilisable pour dépanner, mais préférez le GPU NVIDIA).

Options utiles : `--epochs 50` (plus rapide, un peu moins bon — recommandé sur Mac),
`--batch 64` (fixe le batch au lieu de l'auto), `--model yolo26n.pt` (encore plus
rapide sur Pi si votre version d'ultralytics le propose), `--device 0|mps|cpu`
(forcer le device au lieu de l'auto-détection).

Le script est **idempotent** : en cas d'interruption, relancer reprend le dataset
déjà téléchargé et le même dossier de runs.

## Après l'entraînement

1. Vérifier les métriques affichées (viser **mAP50 > 0.9** sur la classe drone).
2. Copier le poids vers le projet :
   ```bash
   cp drone_yolo11n.pt ../models/drone_yolo11n.pt
   ```
   (et pareil sur le Raspberry Pi : `scp drone_yolo11n.pt pi@<ip>:.../tracker/models/`)
3. Tester en dev : `python main.py --config config.dev.yaml --model drone`
4. Sur le Pi : relancer `bash setup.sh` — il exporte automatiquement le `.pt` en
   **NCNN** (`models/drone_yolo11n_ncnn_model/`), chargé en priorité par le détecteur
   (~2x plus rapide qu'ONNX sur ARM).

## Artefacts

| Fichier | Rôle |
|---------|------|
| `dataset/` | export YOLOv8 du dataset Roboflow (gitignoré, re-téléchargeable) |
| `runs/drone_yolo11n/` | logs d'entraînement, courbes, checkpoints |
| `drone_yolo11n.pt` | **le livrable** — à copier dans `tracker/models/` |
