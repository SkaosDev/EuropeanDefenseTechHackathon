# 🎯 Tracker embarqué — Détection, suivi & estimation de distance

Pipeline de vision embarqué **100 % offline** pour **Raspberry Pi 5 + PiCam Module 3**.
Détecte des objets, les suit en continu (identifiants persistants), estime leur classe et
leur **distance** à la caméra, le tout sur un **tableau de bord web unique** (vidéo + stats
+ journal) et des logs JSON structurés.

> Conçu pour un hackathon militaire. Développable sur PC (fallback webcam), déployable sur Pi.

---

## ✨ Fonctionnalités

- **Deux modèles au choix, tous en local** (`detection.model` ou `--model`) :
  - **`drone`** *(défaut)* — YOLO11n **entraîné maison** sur ~30,7k images (3 classes :
    drone / bird / airplane), léger (~5,4 Mo) — **~10-15 fps sur Pi 5 en NCNN**.
  - **`base`** — YOLOv8n COCO, 80 classes générales, **~7 fps** (rapide, mais ne voit pas les drones).
- **Tracking** ByteTrack : IDs persistants + trajectoires.
- **Estimation de distance** monoculaire (focale + largeur réelle), lissée par track.
- **Tableau de bord unique** (`/`) : vidéo annotée + cibles + journal (poussé en SSE).
- **Priorité drone** : mis en avant (rouge, listé en premier).
- **Robustesse** : fallback caméra, arrêt propre par signaux, redémarrage délégué à systemd.
- **Offline total** : aucune dépendance cloud une fois le modèle téléchargé.

---

## 🛸 Détection de drones (important)

Thème **contre-drone** (Shahed, FPV fibre optique…). **YOLOv8n COCO ne sait PAS détecter un
drone** → le modèle `drone` est un **YOLO11n entraîné maison** (les anciens modèles tiers
flying/yolov8x ont été retirés : détection médiocre et bien trop lourds pour le Pi).
On choisit avec `detection.model` ou `--model drone|base`.

| `--model` | Modèle | Classes | Vitesse (Pi 5) |
|-----------|--------|---------|----------------|
| **`drone`** *(défaut)* | YOLO11n custom (~5,4 Mo, `training/train_drone.py`) | drone, bird, airplane | **~10-15 fps (NCNN)** |
| `base` | yolov8n COCO (~6 Mo) | 80 classes | ~7 fps |

- **Entraînement** : dataset [drone_bird_uav_aircraft (Roboflow, CC BY 4.0)](https://universe.roboflow.com/donia-mceky/drone_bird_uav_aircraft-pz6zp),
  ~30 700 images, classe `uav` fusionnée dans `drone`. Les classes **bird/airplane sont des
  distracteurs** : le modèle apprend à NE PAS confondre un oiseau ou un avion avec un drone
  (la cause n°1 de faux positifs en contre-UAS). Workflow complet : `training/README.md`
  (~1-2 h sur RTX 3090).
- **NCNN sur le Pi** : `setup.sh` exporte automatiquement le `.pt` en NCNN
  (`models/drone_yolo11n_ncnn_model/`), et le détecteur charge ce dossier en priorité —
  ~2× plus rapide qu'ONNX sur ARM. En dev (Mac/Windows), le `.pt` est utilisé tel quel.
- **Résolution d'inférence = 640** (et non 320). Les drones sont petits : downscaler les fait
  disparaître. Principe issu de [YOLO-Drone, MDPI 2023](https://www.mdpi.com/2079-9292/12/17/3664)
  (branche haute résolution pour cibles minuscules). Monter à 960/1280 aide les drones lointains.

---

## 🧱 Architecture

```
capture ─▶ detect ──────▶ track ─▶ distance ─▶ overlay ─┬─▶ dashboard web (/  + /video, /events SSE)
 (PiCam/    (YOLO11n drone)  (ByteTrack) (largeur)        └─▶ logs JSON (thread non-bloquant)
  webcam)
```

| Module | Rôle |
|--------|------|
| `camera/capture.py` | Capture PiCam (lazy) avec repli webcam (DSHOW sous Windows), frames BGR |
| `detection/detector.py` | `ObjectDetector` : YOLO (NCNN auto-détecté), `detect()` (+ ByteTrack), filtrage classes |
| `detection/distance.py` | `DistanceEstimator` : distance monoculaire (largeur) caméra→objet |
| `tracking/tracker.py` | `ObjectTracker` : IDs ByteTrack, trajectoires, durée de vie |
| `output/overlay.py` | bboxes, labels (classe + distance), trajectoires, HUD |
| `output/logger.py` | logs JSON-lines + rotation, écriture en thread démon |
| `output/stream.py` | `Dashboard` Flask : page unique `/` + `/video` (MJPEG) + `/events` (SSE) |
| `utils/fps_counter.py` | mesure FPS glissante |
| `common/types.py` | dataclasses partagées (`Detection`, `TrackedObject`) |
| `main.py` | orchestration, signaux, watchdog, options CLI |

---

## 📋 Prérequis

- **Raspberry Pi 5**, Raspberry Pi OS **Bookworm 64-bit**, PiCam Module 3.
- **Python 3.13+**.
- Connexion internet **uniquement au setup** (téléchargement modèle + paquets).

> `picamera2` s'installe via **apt** (`python3-picamera2`), pas via pip. Le venv est créé
> avec `--system-site-packages` pour le rendre visible. Sur PC de dev, le code bascule
> automatiquement sur une webcam USB.

---

## 🚀 Installation

### Sur Raspberry Pi (cible)

```bash
cd tracker
bash setup.sh            # ou : bash setup.sh --service  (pour le service systemd)
```

Le script : vérifie l'OS, installe les paquets système, crée le venv, installe les
dépendances Python, vérifie la présence du **modèle drone** (`drone_yolo11n.pt`, cf.
`training/README.md`) et l'**exporte en NCNN**, télécharge le modèle de base, crée `logs/` et `models/`.

### Sur PC de dev (Windows / Linux, webcam)

```bash
cd tracker
python -m venv venv
venv\Scripts\activate          # Windows  (Linux : source venv/bin/activate)
pip install -r requirements.txt
# (le modèle drone custom est à copier dans models/drone_yolo11n.pt — cf. training/README.md)
```

Dans `config.yaml`, mettre `camera.use_picamera: false` pour forcer la webcam (sinon le
repli est automatique).

---

## ▶️ Usage

```bash
# Lancement (tableau de bord web) — modèle selon detection.model du config
python main.py

# PC de dev (webcam)
python main.py --config config.dev.yaml

# Forcer un modèle : drone (anti-drone, défaut) ou base (COCO généraliste)
python main.py --config config.dev.yaml --model base
python main.py --config config.dev.yaml --model drone
```

Puis ouvrir le **tableau de bord** depuis n'importe quel appareil du réseau local :

- **Tout-en-un** : `http://<ip-du-pi>:5000/` — vidéo annotée + statistiques + journal.

Routes internes utilisées par la page (pas besoin de les ouvrir à la main) :
`/video` (flux MJPEG, source de l'image) et `/events` (SSE : données poussées au navigateur,
une seule connexion, pas de polling).

### Tests standalone par module

Chaque module est testable indépendamment (depuis le dossier `tracker/`) :

```bash
python -m utils.fps_counter
python -m camera.capture
python -m detection.detector    # nécessite ultralytics + un modèle dans models/
python -m detection.distance
python -m tracking.tracker
python -m output.overlay        # écrit overlay_test.png
python -m output.logger
python -m output.stream         # tableau de bord de test sur :5000
```

---

## ⚙️ Configuration (`config.yaml`)

Tout est centralisé : modèle + résolution d'inférence, seuils, classe prioritaire/cibles,
tracking, estimation de distance (FOV + largeurs réelles + lissage) et sorties (tableau de
bord, logs). Voir les commentaires dans le fichier.

Leviers de performance — **ne pas baisser `inference_size` sous 640**, ça détruit la
détection des petits drones. Préférer :
- **L'export NCNN** (fait automatiquement par `setup.sh` sur le Pi) : le détecteur charge
  `models/drone_yolo11n_ncnn_model/` en priorité s'il existe — ~2× plus rapide sur ARM.
- Baisser la **résolution caméra** (l'inférence reste à 640).

---

## 📊 Performances cibles

| Métrique | Objectif |
|----------|----------|
| FPS pipeline | ≥ 15 fps |
| Latence détection | < 200 ms |
| RAM | < 512 MB |
| CPU moyen | < 75 % |

---

## 🛠️ Troubleshooting

| Symptôme | Cause / Solution |
|----------|------------------|
| `picamera2 indisponible ... bascule sur webcam` | Normal sur PC. Sur Pi : `sudo apt install python3-picamera2` et venv `--system-site-packages`. |
| `Aucune caméra disponible` | Vérifier le câble PiCam / index webcam (`camera.source`), `libcamera-hello` pour tester. |
| Modèle drone non trouvé | L'entraîner sur GPU (`training/README.md`) puis copier `drone_yolo11n.pt` dans `models/`. En attendant : `--model base`. |
| FPS bas sur Pi | Vérifier que l'export NCNN existe (`models/drone_yolo11n_ncnn_model/`, relancer `setup.sh` sinon) ; baisser la résolution caméra. |
| Drones lointains non détectés | Monter `detection.inference_size` (960/1280) — au prix du FPS (cf. § Détection de drones). |
| Distances aberrantes | Ajuster `distance.hfov_deg` (FOV réel caméra) et `distance.known_widths_m` par classe. |
| Page vide / image figée | Vérifier qu'un seul process tourne sur le port ; recharger `/`. |

---

## 📁 Logs

`logs/detections.json` — un objet JSON par ligne :

```json
{"timestamp": "2026-06-09T12:00:00+00:00", "track_id": 1, "class_name": "person", "confidence": 0.88, "distance_m": 3.24, "bbox": [10, 10, 60, 120]}
```

Rotation automatique au-delà de `output.log_rotation_mb`. Écriture en thread démon : le
pipeline n'est jamais bloqué par l'I/O disque.