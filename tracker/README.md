# 🎯 Tracker embarqué — Détection, suivi & estimation de distance

Pipeline de vision embarqué **100 % offline** pour **Raspberry Pi 5 + PiCam Module 3**.
Détecte des objets, les suit en continu (identifiants persistants), estime leur classe et
leur **distance** à la caméra, le tout sur un **tableau de bord web unique** (vidéo + stats
+ journal) et des logs JSON structurés.

> Conçu pour un hackathon militaire. Développable sur PC (fallback webcam), déployable sur Pi.

---

## ✨ Fonctionnalités

- **Trois modèles au choix, tous en local** (`detection.model` ou `--model`) :
  - **`flying`** *(défaut, recommandé)* — YOLOv8m objets volants (drone / avion / hélico /
    oiseau), **~3.6 fps**, détecte les drones (85 % en val) — meilleur compromis vitesse/précision.
  - **`drone`** — YOLOv8x fine-tuné drone (1 classe), un peu plus spécialisé mais lourd (**~1.4 fps**).
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
drone** → on fournit deux modèles entraînés à détecter les drones (+ le COCO de base), tous
téléchargés en local. On choisit avec `detection.model` ou `--model flying|drone|base`.

| `--model` | Modèle | Classes | Vitesse (CPU) |
|-----------|--------|---------|---------------|
| **`flying`** *(défaut)* | [Javvanny yolov8m_flying_objects](https://huggingface.co/Javvanny/yolov8m_flying_objects_detection) (~52 Mo) | drone, avion, hélico, oiseau | **~3.6 fps** |
| `drone` | [doguilmak Drone-YOLOv8x](https://github.com/doguilmak/Drone-Detection-YOLOv8x) (~130 Mo) | drone | ~1.4 fps |
| `base` | yolov8n COCO (~6 Mo) | 80 classes | ~7 fps |

- **`flying` est recommandé** : il détecte les drones (85 % en validation) **et** distingue
  avion / hélicoptère / oiseau (utile en contre-UAS pour « savoir ce qui arrive »), tout en
  étant ~2,7× plus rapide que le YOLOv8x. Ses labels d'origine (cyrillique) sont renommés via
  `class_names` dans la config.
- **Résolution d'inférence = 640** (et non 320). Les drones sont petits : downscaler les fait
  disparaître. Principe issu de [YOLO-Drone, MDPI 2023](https://www.mdpi.com/2079-9292/12/17/3664)
  (branche haute résolution pour cibles minuscules). Monter à 960/1280 aide les drones lointains.

> ⚠️ **Performance** : même `flying` (~3.6 fps) n'est pas du temps réel fluide sur CPU. Pour le
> Pi : exporter en **NCNN** (`yolo export model=models/flying_yolov8m.pt format=ncnn`) puis
> pointer le `path` du modèle sur le dossier `_ncnn_model` — gros gain sur ARM.

---

## 🧱 Architecture

```
capture ─▶ detect ──────▶ track ─▶ distance ─▶ overlay ─┬─▶ dashboard web (/  + /video, /events SSE)
 (PiCam/    (YOLOv8x drone)  (ByteTrack) (largeur)        └─▶ logs JSON (thread non-bloquant)
  webcam)
```

| Module | Rôle |
|--------|------|
| `camera/capture.py` | Capture PiCam (lazy) avec repli webcam (DSHOW sous Windows), frames BGR |
| `detection/detector.py` | `ObjectDetector` : YOLOv8x drone, `detect()` (+ ByteTrack), filtrage classes |
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
dépendances Python, télécharge le **modèle drone** (`drone_yolov8x.pt`, ~130 Mo), crée `logs/` et `models/`.

### Sur PC de dev (Windows / Linux, webcam)

```bash
cd tracker
python -m venv venv
venv\Scripts\activate          # Windows  (Linux : source venv/bin/activate)
pip install -r requirements.txt
# (setup_dev.py télécharge le modèle drone dans models/drone_yolov8x.pt)
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

# Forcer un modèle : drone (précis, lent) ou base (rapide, généraliste)
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
python -m detection.detector    # YOLOv8x drone (nécessite ultralytics + le modèle)
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

Leviers de performance (YOLOv8x est lourd) — **ne pas baisser `inference_size` sous 640**, ça
détruit la détection des petits drones. Préférer :
- **Exporter le modèle en NCNN/ONNX** : `yolo export model=models/drone_yolov8x.pt format=ncnn`
  puis pointer `model_path` sur le résultat (gros gain sur Pi/CPU).
- Un **modèle drone plus léger** (yolov8n/m fine-tuné) si disponible.
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
| Modèle non trouvé | Relancer `setup.sh`, ou télécharger `best.pt` (HuggingFace doguilmak) vers `models/drone_yolov8x.pt`. |
| FPS très bas (~1 fps) | YOLOv8x est lourd. Exporter en NCNN/ONNX (`yolo export model=… format=ncnn`), modèle plus léger, ou baisser la résolution caméra. |
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