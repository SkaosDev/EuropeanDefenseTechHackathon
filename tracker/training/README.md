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

## Mode Mac Apple Silicon (M1/M2/M3) optimisé

Le script détecte automatiquement le Mac et **sature le matériel** sans rien régler :
GPU via **MPS**, **tous les cœurs CPU** pour le data-loading, demi-précision (AMP),
et exploitation de la mémoire unifiée (batch 32 sur 24 Go). Objectif : **~1 s/it**.

```bash
python train_drone.py                 # auto : MPS + batch 32 + workers 8 + AMP
python train_drone.py --fast          # plus rapide : batch 48 + imgsz 512
python train_drone.py --fraction 0.3  # itère vite sur 30% du dataset (test setup)
```

À savoir (réalité technique, pas une limite du script) :

- **La puce neuronale (ANE) ne sert PAS à l'entraînement.** Elle n'est accessible
  qu'en *inférence* via CoreML. PyTorch/MPS l'ignore. Aucun framework d'entraînement
  ne peut l'utiliser. On maximise donc **GPU (MPS) + CPU (dataloader)**, pas l'ANE.
- **`cache ram` est volontairement évité** : ~30k images décodées dépassent 24 Go et
  **gèleraient le Mac**. Le cache disque décode une fois puis relit sans re-décoder.
- **~1 s/it ≠ entraînement court.** À 1 s/it, batch 32, ~21k images train ≈ **11 min/epoch**.
  Donc **100 epochs ≈ 18 h** même au max. Pour un modèle utilisable vite sur Mac :
  `--epochs 50 --fast` (l'early-stopping coupe souvent avant), ou `--fraction 0.5`
  pour un premier jet, puis relancer en plein dataset.
- **Si gel/OOM** : baisse `--batch` (ex. `--batch 24`). Ferme les apps lourdes
  (navigateur) — la RAM est partagée entre le GPU et le système.

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
