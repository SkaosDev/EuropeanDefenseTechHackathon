# AEGIS — Brief technique pour le module d'intervention / décision / stratégie

> À donner à l'instance qui rédige le prompt Claude Code. Toutes les infos « codebase »
> sont vérifiées dans le dépôt (voir `HANDOFF.md` pour l'archi générale). Les décisions
> produit/safety ont été tranchées par Clément (section **Décisions validées**).

---

## 0. Résumé en une ligne

On ajoute un **planner d'interception côté backend** qui consomme la prédiction temps réel
top-k + classe (déjà produite par le modèle), raisonne sur une **flotte d'intercepteurs réels
par site**, décide **auto vs humain** via un gate de safety, simule l'engagement avec une
**issue hit/miss probabiliste**, et expose le tout sur le **WS `/stream`** + un **nouveau
panneau frontend** sous `PredictionPanel`. Pas de refonte : on réutilise l'archi existante.

---

## 1. Le contrat de données EXISTANT (ce que le planner peut consommer aujourd'hui)

### 1.1 Sortie du modèle — `Predictor.predict(events, clock_t)` (`model/infer.py`)
Renvoie (ou `None` tant qu'aucun event) :
```python
{
  "target_topk": [{"dest_id": int, "name": str, "p": float}, ...],   # trié desc, top-5 dans le stream
  "pred_class":   str,        # "shahed136" | "gerbera" | "fpv_fiber" | "lancet"
  "pred_class_p": float,      # confiance classe
  "pred_future":  [[lat, lon], ... x12],   # tête trajectoire future (n_future=12)
}
```

### 1.2 Tick WS `/stream` (`backend/main.py`, fonction `stream`)
Chaque tick (toutes les `tick_real=0.1 s` réelles, horloge `clock` accélérée) :
```python
{
  "type": "tick",
  "clock":     float,          # secondes de simulation (accélérées)
  "drone_pos": [lat, lon],     # position VRAIE du drone, interpolée depuis la vérité-terrain
  "n_events":  int,
  "new_events": [ {t, sensor_lat, sensor_lon, modality, est_class, confidence, is_clutter, drone_pos}, ... ],
  "prediction": <objet 1.1 ou null>,
  "fusion":     {by_modality, n_sensors, n_clutter, n_real, n_modalities},
}
```
Fin de scénario : `{"type":"done","clock":t_max}`.

### 1.3 `POST /spawn` renvoie (avant le stream)
`scenario_id`, `ground_truth` (≤500 points pour l'animation), `drone_class`(+label),
`true_dest_id`/`true_dest_name`/`true_dest_lat`/`true_dest_lon` (vérité, pour l'overlay),
`origin`, **`t_max`** (durée totale de la traj en s de sim), `n_events`.

### 1.4 Endpoints carte (`GET`)
- `/targets` → 28 cibles : `{dest_id, name, oblast, zone_type, objective, lat, lon, pop}`.
- `/origins` → 12 sites de lancement RU `{name, lat, lon, region}`.
- `/classes` → `{name, label, origin_mode, range_km}`.
- `/sensors`, `/das_lines`.

### 1.5 Géométrie déjà dispo — `dataset_generator/geo.py` (À RÉUTILISER, ne pas réimplémenter)
`distance_m`, `initial_bearing`, `destination_point(lat,lon,bearing,dist_m)`,
`bearing_to_unit`/`unit_to_bearing`, `angle_diff`, `haversine_m_vec`,
`point_to_polyline_m`, `nearest_point_on_polyline`, `in_bbox`, `point_in_geojson`.
→ tout ce qu'il faut pour calculer un point de croisement / un temps de vol / un PCA
(point de plus courte approche) entre l'intercepteur et la menace.

### 1.6 Cinématique menace — `dataset_generator/config.yaml`
Vitesses **réelles déjà modélisées** (km/h) :
| classe | speed_kmh | range_km | note |
|---|---|---|---|
| shahed136 | **165–200** (moy ~182) | 1000 | la menace de référence |
| gerbera | 120–160 | 500 | leurre |
| fpv_fiber | 90–150 | 20 | pas d'émission RF |
| lancet | 100–160 | 60 | vise militaire/airbase |
`sim.dt_s = 10`, `reach_radius_m = 1500`. Les 12 villes >250k hab. génèrent déjà des
**zones DCA** (`kinematics.dca`, rayon 28 km / influence 55 km) que le drone évite — réutilisables
comme proxy de « zones défendues » mais ce **ne sont pas** des batteries (pas d'asset modélisé).

---

## 2. ⚠️ Points de correction / pièges (à dire au prompteur)

1. **`drone_pos` du tick = VÉRITÉ-TERRAIN** (interpolée), le modèle ne la voit jamais. Pour rester
   honnête, le planner doit raisonner sur la **piste estimée** (issue de la fusion capteurs /
   `pred_future`), pas sur `drone_pos`. Pour la démo on peut afficher l'engagement contre la
   position vraie, mais le **time-to-impact et le choix de cible doivent venir de la prédiction**.
2. **Ne PAS dériver le time-to-impact de `t_max`** (c'est la vérité-terrain). Le calculer :
   `dist(piste estimée → cible prédite) / vitesse_estimée(classe prédite)`. C'est la version honnête
   et ça « se resserre » au fil des détections comme la prédiction.
3. **La tête `pred_future` est la sortie la plus FAIBLE du modèle** (HANDOFF §10 : ~55–80 km d'erreur).
   → Planifier l'interception contre les **coordonnées de la/les cible(s) prédite(s)** (`target_topk`
   → lat/lon via `/targets`), pas contre les 12 points bruités de `pred_future`. Utiliser `pred_future`
   seulement pour dessiner un corridor indicatif.
4. **Perf modèle réelle** (HANDOFF §5) : top-1 **0.476 @25% obs → 0.748 @100% obs** (global 0.60),
   top-3 0.76→0.96, classe 0.91→0.96. → tout seuil de gate doit être calibré là-dessus (cf §4).
5. **1 drone par scénario** aujourd'hui (le simulateur ne fait pas de vague). Garder l'archi
   extensible mais ne pas casser le 1v1.

---

## 3. Données RÉELLES sur les intercepteurs (recherche 2025–2026, à utiliser pour l'asset model)

Clément veut des intercepteurs **réels**. Modèles de référence anti-Shahed actuellement employés :

| Intercepteur | Vitesse | Portée / rayon | Plafond | Taux de succès | Coût | Note |
|---|---|---|---|---|---|---|
| **STING** (Wild Hornets) | jusqu'à **315 km/h** (P1-SUN ~280) | **25 km** (jusqu'à 37) | 7–11 km | **80–90 %** | ~$2 100–2 500 | quad FPV anti-Shahed, ~1000 Shahed abattus en 4 mois |
| **Octopus-100** | ~**300 km/h** (sources : 400–450) | **30 km** rayon combat | 4,5 km | guidage terminal IA autonome | — | UA en a commandé 8 000 ; production UK |
| **Besomar 3210** | **200 km/h** | — | — | réutilisable | — | anti-Gerbera/recon (cibles plus lentes) |

**Synthèse pour le modèle d'assets :**
- Vitesse intercepteur ≈ **300 km/h** → **~1,65× plus rapide** que le Shahed (182 km/h). Régime
  **poursuite + point d'avance** (et non croisement serré). Écart de vitesse réaliste et confortable.
- **Rayon d'engagement effectif ≈ 30 km** par site (prendre 25–37 km selon le site).
- **P(succès) de base 0,80–0,90** → super baseline pour l'issue hit/miss (§ décision 3).
- **Asymétrie de coût** : intercepteur ~$2 000 vs Shahed ~$50 000 → justifie d'envoyer **2 intercepteurs**
  sur une menace critique sans « coûter cher ». Le coût se mesure surtout en **nb d'intercepteurs**.
- **Spin-up court** (FPV/lancement rapide + guidage terminal autonome) → ~30–60 s plausible.

Sources :
- [STING — Wild Hornets (site officiel)](https://wildhornets.com/en/sting-interceptor)
- [STING interceptors, EW, components — Ukraine's Arms Monitor](https://ukrainesarmsmonitor.substack.com/p/drone-warfare-in-ukraine-sting-interceptors)
- [$2,500 STING down 1,000 Shaheds in 4 months — Interesting Engineering](https://interestingengineering.com/military/ukraine-sting-interceptor-drone-russian-shaheds)
- [Guide to Ukrainian Interceptor Drones — Covert Shores (H.I. Sutton)](https://www.hisutton.com/Ukrainian-Interceptor-Drones.html)
- [Besomar 3210, premier intercepteur réutilisable — Army Recognition](https://www.armyrecognition.com/news/army-news/2025/ukraine-fields-besomar-3210-first-reusable-interceptor-to-counter-russian-geran-2-drones)
- [Octopus-100 entre en production UK — The Defense News](https://www.thedefensenews.com/news-details/Octopus-100-Interceptor-Drones-Enter-UK-Production-Under-Build-with-Ukraine/)
- [Octopus adds a layer to Ukraine's air defences — RUSI](https://www.rusi.org/explore-our-research/publications/commentary/octopus-adds-additional-layer-ukraines-air-defences)

---

## 4. Décisions VALIDÉES par Clément (réponses aux 6 questions)

### Q1 — Modèle d'intercepteurs (assets)
- **Chaque site défensif peut lancer des intercepteurs**, avec un **niveau d'équipement variable**
  selon le site (les grandes villes / sites prioritaires en ont plus). On peut dériver les sites
  défendus des **28 cibles** (`/targets`) — au minimum les `zone_type=city` à forte `pop`/`priority`,
  voire toutes les cibles avec un nb d'intercepteurs ∝ priority/pop.
- Intercepteurs **réels** (cf §3) : vitesse ~**300 km/h**, rayon ~**30 km**, P(succès) base **0,85**,
  spin-up ~**45 s**, **1–2 dispo/site** (plus sur sites prioritaires).
- **Régime de vitesse : intercepteur plus rapide que la menace** (poursuite + point d'avance).

### Q2 — Calcul d'interception
- Géométrie : **point de croisement analytique léger** (poursuite + lead) en réutilisant `geo.py`.
- **NOUVEAU (important) : on ne défend pas que la cible top-1.** Le planner doit considérer **toute
  ville/cible menacée le long du corridor prédit**. Cas explicite donné par Clément : *si le drone
  passe très près d'une autre ville que la cible prédite, et que les calculs estiment l'interception
  possible, on peut engager pour la protéger* — sous réserve que ce soit **fiable et pas trop coûteux**.
  → Pour chaque cible candidate (top-k + villes proches du corridor), calculer time-to-closest-approach
  et faisabilité d'interception depuis les sites voisins ; engager pour protéger la/les plus menacée(s).
- Incertitude : planifier contre le corridor le plus probable tant qu'une cible domine ; si plusieurs
  cibles proches en proba → viser un point qui **couvre l'enveloppe** (ou attendre 1 détection de plus).

### Q3 — Coût + « bonne » stratégie
- **Coût = nb d'intercepteurs (poids fort)** puis temps de vol, sous contrainte P(intercept) ≥ seuil.
  Asymétrie de coût (§3) → envoyer 2 intercepteurs reste acceptable pour une menace critique.
- « Bonne stratégie » = **fiable et pas trop coûteuse** : P(intercept) ≥ seuil **ET** croisement
  **avant un rayon de sécurité** autour de la ville. *(Valeurs de seuil/rayon : voir §5 — défauts
  proposés, à confirmer rapidement si besoin ; sinon partir sur P≥0,7 et rayon 10 km.)*

### Q4 — Gate auto vs humain  *(le cœur safety)*
Déclencheur **AUTO** (toutes conditions) :
- `P(top-1 cible)` **≥ ~0,70** *(et NON 0,85 — le modèle ne l'atteint quasi jamais ; cf §2.4)*
- **ET convergence suffisante** : la prédiction s'est resserrée (fraction d'observation / stabilité
  du top-1 sur N ticks, ou `pred_class_p` ≥ seuil).
- **ET classe identifiée non ambiguë** (drone d'attaque non habité, ex. shahed/lancet, `pred_class_p` haut).
- **ET ∃ stratégie d'interception fiable** (P(intercept) ≥ seuil) **et pas trop coûteuse**.

Escalade **HUMAINE** si l'un de : (a) confiance cible insuffisante, (b) aucune stratégie viable,
(c) trop critique / time-to-impact < seuil, (d) cas ambigu / risque collatéral.

Override humain **visible même en branche auto** (meaningful human control — argument Alta Ares).

### Q5 — Issue simulée + scope
- **Issue : modéliser P(succès) → tirer hit/miss** (marge géométrique : temps de vol intercepteur
  vs menace, marge de portée → P, puis résultat affiché). Plus parlant au jury. Baseline P 0,80–0,90.
- **Scope : 1v1 d'abord, archi prête pour la vague** (liste d'assets + allocation pensées pour N menaces,
  mais le simulateur reste 1 drone/scénario — ne pas le modifier maintenant).

### Q6 — UI : machine à états du nouveau panneau (sous « Threat assessment »)
États : **MONITORING → EVALUATING** (stratégies candidates avec P + coût, qui s'affinent) →
**COMMITTED** (verrouillée + sim + issue hit/miss) **ou HANDOVER** (sim stoppée + « opération
transférée aux opérateurs humains » + dossier complet). Override humain visible en COMMITTED.
Dossier de handover : piste menace, cibles prédites + probas, classe + confiance, time-to-impact,
stratégies tentées + **raison du rejet**, reco. *(à compléter librement)*

---

## 5. Défauts numériques proposés (à confirmer ou laisser tels quels)

| Paramètre | Défaut proposé | Source |
|---|---|---|
| Vitesse intercepteur | 300 km/h | STING/Octopus réels (§3) |
| Rayon engagement / site | 30 km | STING/Octopus (§3) |
| P(succès) base | 0,85 | taux réel 80–90 % (§3) |
| Spin-up | 45 s | FPV/autonome (§3) |
| Intercepteurs / site | 1–2 (plus si priority haute) | décision Clément |
| Seuil auto P(top-1) | **0,70** | calibré sur perf modèle (§2.4) |
| Seuil P(intercept) pour engager | 0,70 | à confirmer |
| Rayon de sécurité ville | 10 km | à confirmer |
| Time-to-impact critique (→ humain) | à définir (ex. < 90 s) | à confirmer |

---

## 6. Forme d'implémentation suggérée (réutilise l'archi, pas de refonte)

- **Backend** : nouveau module `backend/planner.py` (assets dérivés de `/targets` + config dédiée,
  calcul croisement via `geo.py`, gate auto/humain, P(succès)→hit/miss). Le `stream` de
  `backend/main.py` appelle le planner après chaque `pred = pred.predict(...)` et ajoute un champ
  **`intervention`** au tick (état machine + stratégies candidates + issue), sans toucher au reste
  du contrat. Éventuel `GET /assets` pour afficher les sites/inventaire sur la carte.
- **Frontend** : nouveau composant sous `PredictionPanel.jsx` (même colonne droite) consommant
  `live.intervention` ; overlay intercepteurs + trajectoires + point de croisement sur `MapView.jsx`.
- **Invariants à respecter** : parité train/serve intacte (ne pas toucher `infer.py`/`sequence_prep`),
  ne pas utiliser la vérité-terrain pour décider (cf §2), CPU-only.
