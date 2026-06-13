# European Defense Tech Hackathon Paris 2026 — AEGIS Counter-UAS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hackathon](https://img.shields.io/badge/Event-Paris%202026-blue)](https://defensetech-hackathon.eu)

**AEGIS** : prédiction temps réel de la cible d'un drone d'attaque entrant, à partir d'un flux
de **détections capteurs simulées et bruitées** (acoustique / optique / RF / DAS fibre /
observateur citoyen) **fusionnées** en une seule piste. Tout est simulé — aucun capteur réel.

Le modèle **ne voit jamais la vérité-terrain** : seulement le flux d'événements (clutter inclus),
et il reste **aveugle tant que le drone n'est pas détecté sur le territoire ukrainien**. La
prédiction de cible **se resserre** à mesure que les détections s'accumulent.

```
dataset_generator/   # monde + physique + capteurs (simulateur)
model/               # LSTM multi-têtes : cible (28) + classe (4) + trajectoire future
backend/             # FastAPI : POST /spawn + WS /stream (rejeu tick par tick)
frontend/            # interface React/Leaflet (carte + fusion + prédiction)
run.py / run.sh / run.bat   # lanceur multiplateforme
```

## Démarrage (Windows / macOS / Linux)

Prérequis : **Python 3.13** et **Node.js + yarn**.

```bash
# macOS / Linux            |   Windows
./run.sh setup             |   run.bat setup      # .venv + deps Python + yarn install
./run.sh regen             |   run.bat regen      # régénère le dataset (10000 drones)
./run.sh train             |   run.bat train      # entraîne le modèle (~minutes, CPU)
./run.sh demo              |   run.bat demo       # backend :8000 + frontend :5173
```

Puis ouvre **http://localhost:5173** et clique un scénario de démo. Lancer le script **sans
argument** ouvre un menu interactif.
Commandes : `setup · regen [N] [seed] · train [epochs] · eval · backend · frontend · demo`.

## Réalisme (données réelles)

- **28 cibles curées** (villes, NPP, bases aériennes, ports, industrie de défense).
- **Priors de ciblage réels** par oblast **et par type de drone** (ACLED, alertes aériennes,
  comptages Kharkiv/Kherson, OHCHR ; Shahed→villes/énergie/ports, Lancet→bases/militaire).
- **Capteurs réalistes** : acoustique (Sky Fortress + Zvook, dense, accroche tout y compris FPV
  fibre), optique/IR (sites + villes), RF/EW (longue portée, **aveugle aux FPV fibre**), DAS
  (réseau fibre reliant les villes), observateur (ePPO + groupes mobiles) — **sur le territoire
  ukrainien uniquement**.
- Conception détaillée + sources : `docs/superpowers/specs/2026-06-13-counter-uas-realism-redesign-design.md`.

## Interface

Écran centré carte + panneaux latéraux. **Setup** (scénario + presets réalistes) →
**Live** : panneau **Fusion** (capteurs → 1 piste, contribution par modalité, détections en
direct) à gauche, **Threat assessment** (top-5 cibles + classe) à droite. Sur la carte : drone
animé, traîne bleue, **capteurs qui s'allument + lignes vers le drone**, vecteurs de menace
rouges (%). **Zoome** pour révéler capteurs et câbles fibre.

## Notes

- **CPU only.** Parité train/serve : l'inférence importe `_encode_window` du générateur.
- Fallback hors-ligne : `frontend/public/borders.geojson` (la carte s'affiche même sans tuiles).
- Validation : `./run.sh eval` (accuracy par fraction d'observation).

---
*Le module `tracker/` et `test_kurokesu/` (matériel caméra Kurokesu C3) sont indépendants de la démo AEGIS.*
