# Handoff → instance « réseau de neurones + simulation/interface »

Réponses détaillées aux 8 questions, basées sur le dataset **réellement généré** par
`dataset_generator/`. Tous les schémas, exemples et chiffres ci-dessous sont exacts
(2000 drones, `--seed 42`).

> ⚠️ **À lire en premier — un malentendu probable sur le format des données.**
> Tes questions parlent de « départ / arrivée / route estimée ». Or **ce n'est PAS** la forme
> de l'entrée du modèle. Le modèle **ne reçoit jamais** la trajectoire, ni le départ, ni
> l'arrivée, ni une polyline. Il reçoit **un flux d'ÉVÉNEMENTS de détection bruités** émis par
> des capteurs fixes (un point par détection : quel capteur, quand, quelle modalité, classe
> estimée, confiance, cap estimé). La « route » n'existe que de façon **implicite et
> partielle** dans le nuage d'événements. Le départ/arrivée/trajectoire propre existent
> uniquement comme **vérité-terrain (labels + animation)**, pas comme entrée.
> Tout le design en découle.

---

## 0. Délai
Le générateur de données est **déjà fini, testé et validé** (13/13 contrôles). Ton périmètre
en ~1 jour = **modèle + backend de simulation temps réel + interface**. C'est faisable :
le dataset est prêt, petit (7008 séquences), un LSTM léger s'entraîne en **quelques minutes sur CPU**.
Vise : entraînement offline ce soir → poids figés → démo temps réel qui **rejoue** des scénarios.

---

## 1. Données d'entraînement

### Deux niveaux de fichiers (dans `out/`)
- **Brut, lisible** : `detection_events.csv` (= entrée), `ground_truth_trajectories.csv`
  (= vérité-terrain, NE PAS donner au modèle), `targets.csv` (= labels), `sensors.csv`.
- **Prêt-à-entraîner** : `sequences.npz` + `dataset_meta.json` (tenseurs numpy, déjà
  fenêtrés/normalisés/splittés). **Utilise celui-ci** pour le modèle.

### `detection_events.csv` — L'ENTRÉE (format long, 1 ligne = 1 détection)
Colonnes exactes :
`event_id, scenario_id, t, drone_id, sensor_id, sensor_lat, sensor_lon, modality, est_class, confidence, bearing_est, range_est`

Exemple (2 détections réelles d'un même drone par le même capteur optique, puis 2 clutters) :
```
event_id,scenario_id,t,drone_id,sensor_id,sensor_lat,sensor_lon,modality,est_class,confidence,bearing_est,range_est
0,0,10900.0,0.0,opt_778,46.9222252,32.4640371,optical,gerbera,0.0528,62.63,5820.66
1,0,10920.0,0.0,opt_778,46.9222252,32.4640371,optical,shahed136,0.05,64.83,4451.25
34,0,2759.36,,opt_184,50.9581279,24.2838802,optical,fpv_fiber,0.2173,216.35,
35,0,11026.07,,aco_727,52.7660247,34.2350163,acoustic,fpv_fiber,0.1691,69.61,
```
Notes importantes :
- `scenario_id` = **clé de regroupement d'une « piste »** (un drone réel + son clutter). Un
  scénario = un drone. Le modèle traite les événements d'**un scénario** comme une séquence.
- `drone_id` **vide** = **faux positif (clutter)** non lié à un vrai drone. (En pandas, la
  colonne est lue en float ⇒ `NaN` pour le clutter, `0.0, 1.0, …` sinon.) Le modèle ne sait
  PAS lesquels sont des FP — c'est tout l'intérêt.
- `t` = secondes depuis le spawn du drone du scénario (chaque scénario a sa propre horloge à 0).
- `modality` ∈ {optical, acoustic, vibration, das, rf}. `est_class` ∈ {shahed136, gerbera,
  fpv_fiber, lancet} (jamais « clutter » : un FP porte quand même une classe estimée plausible).
- `bearing_est` = cap estimé capteur→drone en degrés (bruité ; large bruit pour vibration/das).
- `range_est` = distance estimée en **mètres**, **seulement pour l'optique** (capteur monoculaire,
  cf. tracker) ; **vide** pour les autres modalités (pas de télémétrie passive fiable).

### `targets.csv` — l'espace de labels (65 cibles curées)
`dest_id, name, oblast, zone_type, objective, lat, lon, pop`
```
dest_id,name,oblast,zone_type,objective,lat,lon,pop
0,Kyiv,Kyiv,city,population,50.4501,30.5234,3109000
1,Kharkiv,Kharkiv,city,population,50.0028,36.2304,1444000
2,Odesa,Odesa,city,population,46.4858,30.7326,1010000
```
`zone_type` ∈ {city, power_tpp, power_hpp, power_npp, airbase, port, defense_industry}.
`objective` ∈ {population, energy, military, logistics, industry} (label de + haut niveau, gratuit).

### `ground_truth_trajectories.csv` — VÉRITÉ-TERRAIN (animation + labels uniquement)
`t, drone_id, lat, lon, alt, speed, bearing, drone_class, origin, dest_id, dest_name, dest_oblast, dest_zone_type, objective`
```
t,drone_id,lat,lon,alt,speed,bearing,drone_class,origin,dest_id,dest_name,dest_oblast,dest_zone_type,objective
0.0,0,45.4461,39.4217,155.14,49.52,286.97,shahed136,Korenovsk,50,Kulbakino AB,Mykolaiv,airbase,military
```
(speed en m/s, bearing en degrés, alt en m, pas de temps = 10 s.)

### Volumes (2000 drones, seed 42)
- 45 432 événements : **32 430 réels / 13 002 clutter (28,6 % FP)**.
- Par modalité : optical 23 822 · acoustic 15 568 · rf 3 560 · vibration 2 440 · das 42.
- Classes (vérité) : shahed136 1083 · gerbera 510 · lancet 250 · fpv_fiber 157.
- ⇒ **7008 séquences** d'entraînement (5608 train / 1400 val).
- Régénérable à volonté : `python -m dataset_generator.main --n-drones N --seed S` (~50 s/2000).

---

## 2. Modèle / sortie → **les trois têtes sont déjà labellisées**, fais (a)+(b)+(c)

`sequences.npz` (clés numpy, formes exactes) :
| Clé | Forme | dtype | Rôle |
|---|---|---|---|
| `X` | (7008, 64, 17) | float32 | séquence d'événements (64 = max_len, 17 features) |
| `mask` | (7008, 64) | float32 | 1 = événement réel, 0 = padding (→ `pack_padded`/masking) |
| `y_target` | (7008,) | int64 | **(a)** id cible 0..64 → tête classif (CrossEntropy) |
| `y_class` | (7008,) | int64 | **(c)** classe drone 0..3 → tête classif |
| `y_future` | (7008, 12, 2) | float32 | **(b)** 12 positions futures (lat,lon **normalisées bbox**) |
| `y_future_mask` | (7008, 12) | float32 | 1 = point futur valide (masque la perte MSE) |
| `is_val` | (7008,) | int8 | 1 = split validation (split **par scénario**, anti-fuite) |
| `scenario_id`, `obs_fraction` | (7008,) | — | métadonnées (cf. §6 pour obs_fraction) |

Les **17 features** par événement (ordre dans `dataset_meta.json → feature_names`) :
`[dt_norm, slat_norm, slon_norm, mod_optical, mod_acoustic, mod_vibration, mod_das, mod_rf,
est_shahed136, est_gerbera, est_fpv_fiber, est_lancet, confidence, bearing_sin, bearing_cos,
range_norm, has_range]`
(modalité et est_class déjà en one-hot ; cap en sin/cos ; range_norm = range/10000 m, `has_range`=0 si absent.)

### Reco modèle (léger, ~1 j)
- `Embedding/Linear(17→64) → LSTM(64→128, 1-2 couches, batch_first) → dernier état (via mask)`
  puis **3 têtes** : `Linear(128→65)` (cible), `Linear(128→4)` (classe), `Linear(128→24)` puis
  reshape (12×2) (trajectoire future).
- Perte = `CE(target) + 0.3·CE(class) + λ·MSE(future)·future_mask`. (a) prioritaire (pilote
  l'interception), garde un petit poids sur (b)/(c).
- **Déséquilibres à gérer** : classes drone (Shahed 54 %) → `class_weight`. Cibles : 65 classes
  dont certaines rares → `label_smoothing`, et **regarde aussi le top-3 accuracy** (l'interface
  montre les meilleures hypothèses, pas une seule).
- **Dé-normaliser** les sorties trajectoire : `lat = slat_norm*(53-44)+44`, `lon = slon_norm*(40.5-22)+22` (bbox dans meta).
- Faisable **CPU**, quelques minutes. Pas besoin de GPU.

### Liste des cibles candidates (ta question)
**Curée**, pas dérivée : ce sont les **65 sites sensibles réels** de `targets.csv` (villes,
centrales TPP/HPP/NPP, bases aériennes, ports, industrie de défense), `dest_id` = index de classe
0..64. `y_target` pointe dedans. L'interface affiche les top-k `dest_id` → flèches rouges vers leurs lat/lon.

---

## 3. Capteurs / physique → **simplifié mais paramétrable** (déjà fait, tout dans `config.yaml`)

Modèle de détection actuel (par pas de temps, par capteur à portée) :
`Pd = base_pd · clip(1 - d_oblique/portée_eff, 0,1)^falloff`, puis Bernoulli ;
`portée_eff = portée_base × force_d'émission(classe, modalité)`. `d_oblique = hypot(d_sol, altitude)`
⇒ un drone haut échappe naturellement aux capteurs courte portée. Bruit gaussien sur
confiance/cap/range, matrices de confusion par modalité, faux positifs (Poisson).
**Pas** de propagation/SNR fin — volontaire pour la démo, mais tous les leviers sont en config.

### 5 modalités, **réseau fixe pré-positionné** sur une grille nationale (jitterée)
| Modalité | grid_km | portée_base | base_pd | cap ? | range ? |
|---|---|---|---|---|---|
| optical | 32 | 9 km | 0.90 | oui | oui (monoculaire) |
| acoustic | 36 | 7 km | 0.85 | oui | non |
| vibration | 45 | 0.8 km | 0.80 | non | non |
| rf | 55 | 20 km | 0.88 | oui | non |
| das (fibre, linéaire) | lignes | 1.5 km transverse | 0.75 | non | non |

Densité fixée par `grid_km` (surchargeable `--grid-km`). **Chaque capteur devine la classe**
(via matrice de confusion par modalité), pas juste la présence — c'est ce qui donne `est_class`.
Réalisme clé : **le FPV fibre n'émet aucun RF** (emission rf=0 ⇒ 0 événement RF, invariant
testé) ; le **Gerbera est confondable** optiquement avec le Shahed mais plus discret en acoustique.

---

## 4. Classes de drones (4) — profils exacts (dans `config.yaml`)

| Classe | spawn | vitesse | altitude | portée | émissions [opt, aco, vib, das, rf] |
|---|---|---|---|---|---|
| **Shahed-136** | 0.55 | 165–200 km/h | 100–2000 m | ~1000 km | 0.80, 0.90, 0.30, 0.30, 0.20 |
| **Gerbera (leurre)** | 0.25 | 120–160 km/h | 100–2000 m | ~500 km | 0.40, 0.45, 0.15, 0.15, 0.15 |
| **FPV fibre** | 0.08 | 90–150 km/h | 5–120 m | ~20 km | 0.30, 0.30, 0.10, 0.20, **0.00 (RF)** |
| **Lancet** | 0.12 | 100–160 km/h | 50–1000 m | ~60 km | 0.40, 0.50, 0.15, 0.15, 0.50 |

Valeurs « plausibles » (pas du classifié) — à ajuster librement en config. Shahed/Gerbera partent
des 12 sites de lancement réels ; FPV/Lancet (courte portée) partent d'un **point avancé près du
front** (cibles restreintes aux oblasts de front). Matrices de confusion complètes dans `config.yaml → sensors.confusion`.

---

## 5. Stack / interface → **ta proposition est la bonne**, avec un atout

React + carte (Leaflet ou MapLibre) + backend Python + WebSocket = ✅. **CPU suffit.**

**L'atout** : le simulateur est en Python et **réutilisable en direct**. Le backend peut importer
`dataset_generator` et appeler `simulator.simulate_drone(...)` pour générer un scénario à la volée
(vraie trajectoire + flux d'événements horodatés), puis **rejouer les événements dans l'ordre
temporel** via WebSocket. Pas besoin de pré-générer pour la démo : spawn(ville/cible/classe) →
le sim produit la piste → tu streames les événements tick par tick.

Schéma backend conseillé :
```
POST /spawn {origin?, target?, class?}  -> simulator.simulate_drone -> (gt, events triés par t)
WS /stream  : à chaque tick d'horloge, push les events dont t <= horloge
             -> accumule la fenêtre d'events du scénario
             -> features (même encodage que sequence_prep, 17 dims) -> modèle
             -> push {target_topk: [(dest_id,p)], pred_class, pred_future[12x2]}
Front: trajectoire bleue (foncé=events reçus / clair=reste via gt), flèches rouges = top-k cibles avec %
```
Important : **réutilise exactement l'encodage de features de `sequence_prep._encode_window`**
(mêmes 17 colonnes, mêmes normalisations, `dataset_meta.json`) côté inference, sinon décalage train/serve.

---

## 6. Le « wow » minimum pour le jury

**Indispensable** : la **prédiction de cible qui se met à jour en direct** pendant que le drone
avance et que les événements arrivent — les flèches rouges (%) se resserrent vers la vraie cible
à mesure que la piste se confirme. C'est exactement ce pour quoi le dataset est conçu :
l'augmentation `obs_fraction` ∈ {0.25, 0.5, 0.75, 1.0} entraîne le modèle à prédire **tôt**, sur
peu d'événements ⇒ la convergence live est « gratuite ».

**Bonus** (si le temps) :
- afficher la **classe estimée** qui se stabilise (et le cas FPV fibre « invisible au RF » comme
  démonstration de finesse capteurs) ;
- la **trajectoire future prédite** (tête (b)) superposée à la vraie ;
- montrer les **faux positifs** clignoter et le modèle qui ne se laisse pas berner ;
- un curseur de **densité de capteurs** (`--grid-km`) montrant l'impact sur la précision.

Démo idéale : déclenche 2-3 scénarios (un Shahed long Nord→Kyiv, un groupe Sud→Odesa avec
leurres Gerbera, un FPV près du front) et laisse le jury voir la prédiction se verrouiller.

---

## Démarrage rapide pour l'autre instance
```bash
# le dataset est déjà dans dataset_generator/out/ (régénérable)
python -m dataset_generator.main --n-drones 4000 --seed 1 --out dataset_generator/out  # plus de data si besoin
python - <<'PY'
import numpy as np, json
d = np.load("dataset_generator/out/sequences.npz")
meta = json.load(open("dataset_generator/out/dataset_meta.json"))
Xtr, ytr = d["X"][d["is_val"]==0], d["y_target"][d["is_val"]==0]
print(Xtr.shape, "features:", meta["feature_names"])
PY
```
Doc complète : `dataset_generator/README.md`. Tous les paramètres : `dataset_generator/config.yaml`.
```
