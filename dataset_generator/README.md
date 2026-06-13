# dataset_generator — Simulateur de données d'entraînement counter-UAS

Génère un **dataset synthétique** pour entraîner un modèle de **détection + anticipation**
de drones (alerte précoce anti-drone). Le modèle ne voit **jamais la vraie trajectoire** :
il consomme uniquement un **flux d'événements de détection bruités** émis par un réseau de
capteurs simulés, et doit prédire **(1) la cible**, **(2) la classe du drone** et
**(3) la trajectoire future**.

> Usage **défensif / éducatif** (hackathon). Coordonnées issues d'OSINT public
> (ISW, Defense Express, militarnyi, Wikipedia/Wikidata/OSM), WGS-84.

## Pipeline

```
spawn drone (classe) A→B
  → vraie trajectoire (cinématique réelle, géo réelle)         [VÉRITÉ-TERRAIN]
  → signatures physiques émises selon la classe
  → capteurs (grille nationale, 5 modalités) -> ÉVÉNEMENTS bruités + FP/FN  [ENTRÉE MODÈLE]
  → labels : cible (65 classes) · classe drone (4) · trajectoire future
```

- **4 classes** : Shahed-136, Gerbera (leurre), FPV fibre optique, Lancet — chacune avec une
  enveloppe cinématique et des **forces de signature** par modalité distinctes.
- **5 modalités de capteurs** : optique, acoustique, vibration sismique, fibre DAS, RF/SIGINT.
  Réalisme : le **FPV fibre n'émet aucun RF** ; le **Gerbera est confondable** optiquement
  avec le Shahed ; la distance utilisée est **oblique** (un drone haut échappe aux capteurs
  courte portée).
- **12 sites de lancement réels** (origines) → **65 cibles réelles** (villes, TPP/HPP/NPP,
  bases aériennes, ports, industrie de défense) sur ~25 oblasts.

## Installation

```bash
python3 -m venv dataset_generator/venv
dataset_generator/venv/bin/pip install -r dataset_generator/requirements.txt
```

## Utilisation

Depuis la racine du dépôt :

```bash
# Génère tout (CSV + GeoJSON + sequences.npz + map.html) dans out/
python -m dataset_generator.main --n-drones 2000 --seed 42 --out dataset_generator/out

# Valide le dataset (contrôles cinématiques, géo, FP/FN, invariants, séquences)
python -m dataset_generator.validate --out dataset_generator/out

# (Re)construire seulement les séquences ou la carte depuis les CSV
python -m dataset_generator.sequence_prep --out dataset_generator/out
python -m dataset_generator.visualize     --out dataset_generator/out
```

Options : `--seed`, `--dt` (pas de temps s), `--grid-km` (densité capteurs),
`--no-prep`, `--no-viz`. ~50 s pour 2000 drones.

## Sorties (`out/`)

| Fichier | Rôle |
|---|---|
| `detection_events.csv` | **ENTRÉE MODÈLE** : flux d'événements `event_id, scenario_id, t, drone_id, sensor_id, sensor_lat/lon, modality, est_class, confidence, bearing_est, range_est`. `drone_id` vide = faux positif (clutter). |
| `ground_truth_trajectories.csv` | Vérité-terrain (PAS une entrée du modèle) : `t, drone_id, lat, lon, alt, speed, bearing, drone_class, origin, dest_*, objective`. |
| `targets.csv` | Les 65 cibles = espace de labels (`dest_id, name, oblast, zone_type, objective, lat, lon, pop`). |
| `sensors.csv` / `sensors.geojson` | Réseau de capteurs (+ lignes fibre DAS). |
| `trajectories.geojson` | LineStrings pour la carte. |
| `sequences.npz` + `dataset_meta.json` | **Tenseurs prêts LSTM** (voir ci-dessous). |
| `map.html` | Carte folium interactive (pitch). |

## Charger les séquences pour l'entraînement

`sequences.npz` (clés numpy) — chaque échantillon = un scénario observé jusqu'à un certain
**taux d'observation** (`observation_fractions` = augmentation "alerte précoce") :

| Clé | Forme | Description |
|---|---|---|
| `X` | `[N, max_len, 17]` | séquence d'événements (features normalisées, cf. `dataset_meta.json → feature_names`) |
| `mask` | `[N, max_len]` | 1 = événement réel, 0 = padding |
| `y_target` | `[N]` | id de la cible (0..64) — **tête distribution sur cibles** |
| `y_class` | `[N]` | classe du drone (0..3, ordre `class_order`) — **tête classe** |
| `y_future` | `[N, n_future, 2]` | positions futures normalisées (lat,lon) — **tête trajectoire** |
| `y_future_mask` | `[N, n_future]` | 1 = point futur valide |
| `is_val` | `[N]` | 1 = split validation (split **par scénario**, anti-fuite) |
| `scenario_id`, `obs_fraction` | `[N]` | métadonnées |

```python
import numpy as np, json
d = np.load("dataset_generator/out/sequences.npz")
meta = json.load(open("dataset_generator/out/dataset_meta.json"))
Xtr, Xva = d["X"][d["is_val"] == 0], d["X"][d["is_val"] == 1]
# 17 features : meta["feature_names"] ; dénormaliser lat/lon via meta["bbox"]
```

## Configuration

Tout est dans `config.yaml` (commenté) : sites, classes + signatures, capteurs (grille,
portées, bruit, matrices de confusion, ratio de FP), routage (matrice O-D, zones DCA à
contourner, corridors), cinématique, et paramètres des séquences. Aucun magic number dans le code.

## Limites assumées (synthétique)

- La grille de capteurs couvre une bbox rectangulaire (inclut un peu la mer Noire / zones
  frontalières) — simplification documentée, pas un polygone Ukraine exact.
- Pas de relief réel (DEM), pas de vagues coordonnées multi-drones (1 drone = 1 trajectoire
  indépendante). Signatures et matrices de confusion = ordres de grandeur plausibles, à régler.
- Coordonnées des cibles = centroïdes/positions publiques approximatives (~10–100 m).
