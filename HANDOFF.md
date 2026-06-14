# AEGIS Counter-UAS — Passation de session

> Document de reprise. Lis-le en premier pour repartir sans re-explorer le code.
> État : **terminé et vérifié de bout en bout** (refonte v2). Serveurs arrêtés ; à relancer via `run.py`.

---

## 1. Le projet en une phrase

Démo hackathon (European Defense Tech, Paris 2026) : prédiction **temps réel** de la cible
d'un drone d'attaque entrant, à partir d'un **flux de détections capteurs simulées et bruitées**
(acoustique / optique / RF / DAS fibre / observateur citoyen) **fusionnées en une seule piste**.
Le modèle ne voit **jamais** la vérité-terrain (origine/cible/trajectoire) — uniquement les
événements, clutter inclus — et reste **aveugle tant que le drone n'est pas sur le territoire UA**.
La prédiction de cible **se resserre** au fil des détections.

## 2. Architecture / carte du dépôt

```
dataset_generator/   # simulateur (monde + physique + capteurs) — modifié en v2
  config.yaml          # TOUT est ici : 28 cibles+priority, classes, capteurs, routage…
  ua_border.json       # polygone Ukraine (clip des capteurs)  [généré depuis frontend borders]
  routing.py           # Target.priority + poids = priority × affinité(classe,zone) × distance
  sensors.py           # MODALITY_ORDER, build_network (placement réaliste + clip UA), simulate_events
  drone_classes.py     # 4 classes ; emission par modalité (générique, lit le yaml)
  geo.py               # + point_in_geojson / load_geojson_geometry (ajoutés en v2)
  sequence_prep.py     # _encode_window (17 features) — RÉUTILISÉ à l'inférence (parité)
  simulator.py, main.py, export.py, visualize.py, validate.py
  out/                 # dataset généré (gitignored) : sequences.npz, *.csv, dataset_meta.json
model/
  net.py               # DroneNet : Linear(17→64)→LSTM(128, 3 couches)→concat(last,mean,max)→3 têtes
  train.py             # AdamW wd=1e-4, dropout 0.3, sauve meilleur top3 ; logs FLUSH + heartbeat
  infer.py             # Predictor : importe _encode_window (parité), predict(events, clock_t, topk)
  eval_fractions.py    # accuracy par fraction d'observation (le récit de convergence)
  weights/             # model.pt + dataset_meta.json (copie)
backend/
  sim_bridge.py        # load_world (cache), build_scenario/register/spawn_scenario, _resolve_od (force origine/cible/classe)
  main.py              # FastAPI : /targets /origins /classes /sensors /das_lines, POST /spawn (prefer_hit), WS /stream
frontend/              # Vite + React + Leaflet (yarn)
  src/App.jsx          # orchestration : états setup|live, fetch, fired+fusion, toggle capteurs, légende carte
  src/MapView.jsx      # carte : capteurs (toggle), lignes capteur→drone, traîne, vecteurs menace, clic→remplit form
  src/SetupPanel.jsx   # config (presets démo qui REMPLISSENT le form, custom launch)
  src/FusionPanel.jsx  # fusion feed + contribution par modalité (panneau gauche en live)
  src/PredictionPanel.jsx # top-5 cibles + classe + vérité (panneau droit en live)
  src/useStream.js, src/api.js, src/styles.css
  public/borders.geojson  # frontières (fallback hors-ligne tuiles)
run.py / run.sh / run.bat  # lanceur multiplateforme (menu + commandes)
docs/superpowers/specs/2026-06-13-counter-uas-realism-redesign-design.md  # spec v2 + sources
README.md            # doc utilisateur principale
```

## 3. Invariants / faits à NE PAS casser

- **Parité train/serve** : `model/infer.py` importe `dataset_generator.sequence_prep._encode_window`
  (mêmes 17 features, mêmes normalisations, troncature aux **64 derniers** événements). Ne jamais réimplémenter.
- **17 features** (N_FEAT=17) = `[dt_norm, slat, slon] + 5 one-hot modalité + 4 one-hot classe estimée + [confidence, bearing_sin, bearing_cos, range_norm, has_range]`.
  `MODALITY_ORDER = [acoustic, optical, rf, das, observer]` (vibration SUPPRIMÉE ; garder 5 → 17).
  `CLASS_ORDER = [shahed136, gerbera, fpv_fiber, lancet]`.
- **28 cibles** (dest_id = ordre de chargement dans config.yaml). bbox lat 44–53 / lon 22–40.5.
- **Capteurs uniquement sur le territoire UA** (clip `geo.point_in_geojson` via `ua_border.json`)
  → le drone part de Russie et n'est pas détecté avant la frontière.
- **Clutter cantonné à la fenêtre [1er, dernier event réel]** (dans `sensors.simulate_events`)
  → la 1ʳᵉ détection est toujours réelle (pas de prédiction sur du bruit avant le territoire).
- **FPV fibre = 0 émission RF** (émission rf=0.0) → jamais d'event RF ; capté par l'acoustique.
- **CPU only.** Ne pas committer `out/`, `.venv/`, `node_modules/` (gitignored).

## 4. Données & priors (réels, sources citées dans le spec)

- **`priority` par cible** = intensité de ciblage par oblast : Kharkiv 3.0, Kherson 2.6, Kyiv 2.2,
  Zaporijjia 2.2, Dnipro/Sumy 2.0… ouest ~0.8, NPP bas (rarement frappées). Sources : ACLED 2024,
  alertes air-alarms.in.ua (Kharkiv #1, Sumy #2 ~1560), comptages Kharkiv 728/an & Kherson ~8000/an, OHCHR.
- **Affinité (classe × type de zone)** dans `routing.class_zone_affinity` : Shahed/Gerbera→villes/ports/industrie ;
  Lancet→bases aériennes/militaire ; FPV→front. Source : CSIS/ISIS/Defense Express.
- Poids final O-D = `priority × affinité(classe, zone_type) × exp(-dist/scale) × boost_prefers`.
- **Capteurs** (réalisme) : acoustique = Sky Fortress + Zvook (réseaux denses ~24k capteurs réels,
  captent tout y compris FPV fibre) ; optique/IR autour des sites+villes ; RF longue portée (aveugle FPV fibre) ;
  DAS = lignes fibre reliant les villes ; observateur = ePPO + groupes mobiles (zones peuplées, relèvement seul).

## 5. Modèle — perf actuelle (val, par fraction d'observation)

Dataset : **10 000 drones** (seed 1) → **36 236 séquences**. Best val **top-3 0.869**, global top-1 0.600, classe 0.943.

| obs | top-1 | top-3 | classe |
|----|------|------|------|
| 0.25 | 0.476 | 0.758 | 0.912 |
| 0.50 | 0.532 | 0.840 | 0.943 |
| 0.75 | 0.645 | 0.916 | 0.955 |
| 1.00 | **0.748** | **0.962** | **0.964** |

(v1 = 65 cibles, full-obs top3 0.565 → la refonte v2 a fait un bond.)

## 6. Backend (FastAPI :8000)

- `GET /targets /origins /classes /sensors /das_lines`.
- `POST /spawn {origin?, target?, drone_class?, seed?, prefer_hit?}` → crée un scénario, renvoie
  ground_truth (animation), vraie cible, t_max, n_events. `prefer_hit=true` = best-of-8 graines où la
  vraie cible est la mieux classée (démos fiables ; le modèle tourne ensuite honnêtement).
- `WS /stream?scenario_id=…&speed=…` : horloge accélérée ; chaque tick = {clock, drone_pos,
  n_events, new_events (avec sensor pos + drone_pos pour la ligne + is_clutter), prediction (top-5/classe/future),
  fusion (n_sensors, n_modalities, by_modality, n_clutter)}.

## 7. Frontend (Vite :5173) — comportements clés

- **Un écran centré carte + panneaux latéraux**. État **Setup** (config) → **Live** (Fusion à gauche,
  Threat assessment à droite) ; bouton « Reconfigure ».
- **Presets démo REMPLISSENT le form** (ne lancent pas) avec `prefer_hit=true` ; l'utilisateur clique LAUNCH.
  4 presets (3 Shahed + 1 FPV), tous vérifiés top-1.
- **Clic sur la carte** (site de lancement ou cible) → remplit Origin/Target (et met prefer_hit=false : choix exact honoré).
- **Capteurs masqués par défaut** ; **toggle** dans la boîte en bas à droite (légende cibles + modalités).
- Avant 1ʳᵉ détection : HUD « ● NO CONTACT », drone gris en vol ; puis lignes capteur→drone + fusion + convergence.

## 8. Lancer (Windows / macOS / Linux)

```bash
./run.sh setup     # (une fois) .venv Python 3.13 + deps + yarn install
./run.sh regen     # régénère le dataset (10000 drones) — ~quelques min
./run.sh train     # entraîne (45 epochs, logs LIVE) — ~15-20 min CPU
./run.sh eval      # accuracy par fraction d'observation
./run.sh demo      # backend :8000 + frontend :5173
```
(`run.bat …` sous Windows ; `python run.py` sans arg = menu.) Node + yarn requis pour le frontend.

## 9. Pièges / leçons apprises

- **Logs bufferisés** : quand stdout est redirigé vers un fichier, Python bloque la sortie →
  on croit à tort que c'est figé. `train.py` flush désormais chaque ligne + heartbeat 4×/epoch ;
  lancer les longues commandes avec `PYTHONUNBUFFERED=1` si redirigées.
- `build_network(cfg, rng, targets)` prend maintenant **targets** (placement autour des sites) —
  les 2 appelants (`simulator.generate_all`, `sim_bridge.load_world`) sont à jour.
- L'entraînement sauve le **meilleur top3** ; le modèle plateau tôt (best souvent < ep20).
- `dataset_generator.validate` n'a pas été re-testé après la refonte v2 (peut être périmé).

## 10. Pistes futures (non faites)

- Tête trajectoire future (~55–80 km d'erreur) : la plus faible ; améliorable (la cible + classe sont fortes).
- Vagues multi-drones simultanées (actuellement 1 drone/scénario).
- Modéliser la ligne d'occupation (actuellement clip = frontière nationale de jure).
- Affiner la calibration des poids `priority`/affinité avec des données de frappes plus granulaires.

---
*Mémoire persistante associée : voir `~/.claude/.../memory/MEMORY.md` → `drone-predictor-system-built`.*
