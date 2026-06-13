# dataset_generator — Simulateur de données d'entraînement counter-UAS

Génère un **dataset synthétique** pour entraîner un modèle de **détection + anticipation**
de drones (alerte précoce anti-drone). Le modèle ne voit **jamais la vraie trajectoire** :
il consomme uniquement un **flux d'événements de détection bruités** émis par un réseau de
capteurs simulés, et doit prédire **(1) la cible**, **(2) la classe du drone** et
**(3) la trajectoire future**.

> Usage **défensif / éducatif** (hackathon). Coordonnées issues d'OSINT public
> (ISW, ACLED, Defense Express, militarnyi, air-alarms.in.ua, OHCHR), WGS-84.

## Pipeline

```
spawn drone (classe) A→B
  → vraie trajectoire (cinématique réelle, géo réelle)         [VÉRITÉ-TERRAIN]
  → signatures physiques émises selon la classe
  → capteurs (5 modalités, sur le territoire UA) -> ÉVÉNEMENTS bruités + FP  [ENTRÉE MODÈLE]
  → labels : cible (28 classes) · classe drone (4) · trajectoire future
```

- **4 classes** : Shahed-136, Gerbera (leurre), FPV fibre optique, Lancet — chacune avec une
  enveloppe cinématique et des **forces de signature** par modalité distinctes.
- **5 modalités de capteurs** : **acoustique** (réseaux denses type Sky Fortress / Zvook),
  **optique/IR**, **RF/SIGINT**, **fibre DAS**, **observateur** (signalement citoyen ePPO +
  groupes mobiles). *Pas de vibration sismique* (pas de déploiement réel).
  Réalisme : le **FPV fibre n'émet aucun RF** (accroché par l'acoustique) ; le **Gerbera est
  confondable** optiquement avec le Shahed ; distance **oblique** (un drone haut échappe aux
  capteurs courte portée).
- **Placement réaliste, clippé au territoire ukrainien** (`geo.point_in_geojson` +
  `ua_border.json`) : acoustique en grille dense, optique autour des sites + villes, RF longue
  portée, DAS le long de lignes fibre reliant les villes, observateur en zones peuplées.
  Conséquence : **aucune détection tant que le drone n'est pas sur/près du territoire**.
- **12 sites de lancement réels** (origines) → **28 cibles curées** (villes, NPP, bases
  aériennes, ports, industrie de défense).
- **Priors de ciblage réels** : `priority` par cible (intensité observée par oblast) ×
  **affinité (classe × type de zone)** (Shahed→villes/ports/industrie, Lancet→bases aériennes,
  FPV→front). Les fréquences de cible reflètent donc des données OSINT, pas un tirage uniforme.

## Installation

```bash
python3 -m venv dataset_generator/venv
dataset_generator/venv/bin/pip install -r dataset_generator/requirements.txt
```

## Utilisation

Depuis la racine du dépôt (ou via le lanceur : `./run.sh regen`) :

```bash
# Génère tout (CSV + GeoJSON + sequences.npz) dans out/ ; la démo utilise 10000 drones
python -m dataset_generator.main --n-drones 10000 --seed 1 --out dataset_generator/out --no-viz

# (Re)construire seulement les séquences ou la carte depuis les CSV
python -m dataset_generator.sequence_prep --out dataset_generator/out
python -m dataset_generator.visualize     --out dataset_generator/out
```

Options : `--seed`, `--dt` (pas de temps s), `--grid-km` (densité des modalités en grille :
acoustique/RF), `--no-prep`, `--no-viz`.

## Sorties (`out/`)

| Fichier | Rôle |
|---|---|
| `detection_events.csv` | **ENTRÉE MODÈLE** : `event_id, scenario_id, t, drone_id, sensor_id, sensor_lat/lon, modality, est_class, confidence, bearing_est, range_est`. `drone_id` vide = faux positif (clutter). |
| `ground_truth_trajectories.csv` | Vérité-terrain (PAS une entrée du modèle) : `t, drone_id, lat, lon, alt, speed, bearing, drone_class, origin, dest_*, objective`. |
| `targets.csv` | Les 28 cibles = espace de labels (`dest_id, name, oblast, zone_type, objective, lat, lon, pop`). |
| `sensors.csv` / `sensors.geojson` | Réseau de capteurs (+ lignes fibre DAS). |
| `trajectories.geojson` | LineStrings pour la carte. |
| `sequences.npz` + `dataset_meta.json` | **Tenseurs prêts LSTM** (voir ci-dessous). |

## Charger les séquences pour l'entraînement

`sequences.npz` (clés numpy) — chaque échantillon = un scénario observé jusqu'à un certain
**taux d'observation** (`observation_fractions` = augmentation "alerte précoce") :

| Clé | Forme | Description |
|---|---|---|
| `X` | `[N, max_len, 17]` | séquence d'événements (features normalisées, cf. `dataset_meta.json → feature_names`) |
| `mask` | `[N, max_len]` | 1 = événement réel, 0 = padding |
| `y_target` | `[N]` | id de la cible (0..27) — **tête distribution sur cibles** |
| `y_class` | `[N]` | classe du drone (0..3, ordre `class_order`) — **tête classe** |
| `y_future` | `[N, n_future, 2]` | positions futures normalisées (lat,lon) — **tête trajectoire** |
| `y_future_mask` | `[N, n_future]` | 1 = point futur valide |
| `is_val` | `[N]` | 1 = split validation (split **par scénario**, anti-fuite) |
| `scenario_id`, `obs_fraction` | `[N]` | métadonnées |

Les 17 features = `dataset_meta.json["feature_names"]` (one-hot sur 5 modalités + 4 classes
estimées + temps/position/relèvement/portée normalisés). Dénormaliser lat/lon via `meta["bbox"]`.

## Configuration

Tout est dans `config.yaml` (commenté) : sites, 28 cibles + `priority`, classes + signatures,
capteurs (placement par modalité, portées, bruit, matrices de confusion, ratio de FP, lignes
DAS), routage (priority × affinité classe×zone, zones DCA, corridors), cinématique, séquences.
`ua_border.json` = polygone du territoire pour le clip des capteurs. Aucun magic number dans le code.

## Limites assumées (synthétique)

- Capteurs clippés au **polygone national ukrainien** (Natural Earth basse résolution) — la
  ligne d'occupation réelle n'est pas modélisée ; le ciblage du front est géré par le routage.
- Pas de relief réel (DEM), pas de vagues coordonnées multi-drones (1 drone = 1 trajectoire).
- Signatures, matrices de confusion et poids `priority`/affinité = ordres de grandeur
  plausibles dérivés d'OSINT, à régler.
- Coordonnées des cibles = positions publiques approximatives (~10–100 m).
