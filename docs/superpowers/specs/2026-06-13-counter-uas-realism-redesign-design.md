# Counter-UAS demo — refonte réalisme (v2)

## Contexte

La v1 (modèle LSTM + backend FastAPI + frontend React/Leaflet) fonctionne, mais le retour
utilisateur demande plus de réalisme et un meilleur récit de démo :

1. Trop de cibles (65) → réduire pour gagner en précision.
2. Les fréquences de ciblage doivent refléter des **données réelles** (telle ville
   vraiment plus visée qu'une autre), pas des poids arbitraires.
3. Montrer beaucoup mieux la **détection multi-capteurs** et la **fusion** de leurs données.
4. **Séparer** la configuration de l'attaque de l'analyse temps réel (deux écrans).
5. Améliorer la **précision** de la cible prédite.
6. **Placement réaliste** des capteurs (pas de caméras partout ; acoustique dense ; DAS
   reliant les villes) et **aucune détection/prédiction avant que le drone ne soit sur/près
   du territoire ukrainien**.

Contrainte levée par l'utilisateur : on peut désormais modifier `dataset_generator/`
(config + sensors + geo + routing) — nécessaire pour les points 1, 2, 6.

## Sources (données réelles)

**Priors de ciblage :**
- ACLED 2024 : Kharkiv = oblast le plus meurtrier après Donetsk ; Sumy 4e (devant
  Zaporizhzhia/Dnipropetrovsk). <https://acleddata.com/report/bombing-submission-russian-targeting-civilians-and-infrastructure-ukraine>
- Alertes aériennes par oblast : Kharkiv #1, Sumy #2 (**1 560** en 2024).
  <https://air-alarms.in.ua/en> · <https://mezha.net/eng/bukvy/sumy-region-ranks-second-in-ukraine-for-air-raid-alerts-in-2025/>
- Kharkiv : 421 frappes aériennes (S1 2025), 728 attaques (2025).
  <https://gwaramedia.com/en/russia-hit-kharkiv-with-421-air-attacks-in-the-first-half-of-2025-killed-25-people-including-1-child/>
- Kherson : ~8 000 attaques sur la communauté en 2024.
  <https://intent.press/en/news/war/2025/russian-army-intensifies-attacks-on-kherson-community-over-2500-strikes-in-2025/>
- OHCHR : victimes concentrées Donetsk/Kharkiv/Zaporizhzhia/Kherson/Sumy/Dnipropetrovsk/Chernihiv.
  <https://ukraine.ohchr.org/en/Protection-of-Civilians-in-Armed-Conflict-October-2025>

**Ciblage par type de drone :**
- Shahed-136 : masse sur villes + énergie + ports (stratégique/terreur). <https://missilethreat.csis.org/missile/shahed-131-and-136/> · <https://isis-online.org/isis-reports/may-2025-updated-analysis-of-russian-shahed-136-deployment-against-ukraine>
- Gerbera : leurre accompagnant les Shahed (depuis juil. 2024) → mêmes axes (villes/ports), saturation DCA.
- Lancet (40–70 km) : artillerie, DCA, radars, aéronefs près du front → bases aériennes/militaire. <https://en.defence-ua.com/analysis/analysis_of_russian_forces_use_of_shahed_131136_drones_fpv_drones_and_lancet_barrage_munitions_in_september-11867.html>

**Capteurs réels :**
- Acoustique : Sky Fortress + Zvook + Pokrova, **14 000–24 000+ capteurs**, ~400–500 $/pièce,
  micro+téléphone, portée 150–450 m, déjà nationwide ; **Zvook détecte les FPV fibre**.
  <https://united24media.com/war-in-ukraine/sky-fortress-ukraines-acoustic-detection-system-that-tracks-drones-cheap-and-fast-9451> · <https://www.army.mil/article/292099/> · <https://militarnyi.com/en/news/ukraine-develops-acoustic-detector-for-fpv-drones/>
- RF/EW : **aveugle aux FPV fibre** (aucune émission radio). <https://www.twz.com/news-features/ukraine-discloses-new-method-to-defeat-russian-fiber-optic-controlled-fpv-drones>
- Optique/IR : caméras + IA sur positions/sites, pas partout. <https://defence-blog.com/ukrainian-firm-develops-ai-drone-detection-software-for-frontline-use/>
- DAS : fibre télécom → réseau de capteurs, localise (~1,47°), tout-temps. <https://www.nec-labs.com/blog/drone-detection-and-localization-using-enhanced-fiber-optic-acoustic-sensor-and-distributed-acoustic-sensing-technology/>
- Observateur citoyen : app **ePPO** (180 000+ téléchargements) + groupes feu mobiles
  (imageurs thermiques) → relèvement, excellent vs Shahed. <https://lieber.westpoint.edu/civilians-reporting-cell-phones-direct-participation-hostilities/>

**Honnêteté** : les comptages fiables et publics sont au niveau **oblast** (+ quelques
grandes villes). Les poids par cible sont un mapping **ordinal→cardinal transparent** dérivé
de ces classements (chaque poids commenté avec sa source), pas des chiffres inventés.

## Design

### A. Cibles curées → 28 sites
- **12 villes** : Kyiv, Kharkiv, Odesa, Dnipro, Zaporijjia, Lviv, Kryvyi Rih, Mykolaiv,
  Vinnytsia, Poltava, Sumy, Kherson.
- **4 NPP** : Zaporizhzhia, South Ukraine, Rivne, Khmelnytskyi.
- **5 bases aériennes** : Starokostiantyniv, Myrhorod, Vasylkiv, Ozerne, Kulbakino.
- **3 ports** : Pivdennyi, Chornomorsk, Izmail (distincts des villes).
- **4 industries défense** : Pivdenmash, Motor Sich, Antonov/Hostomel, Artem.
- Supprimés : TPP (9) + HPP (5), villes/bases/ports secondaires, et **doublons co-localisés**
  (ex. « Odesa port » ≡ Odesa, « Ivchenko-Progress » ≡ Motor Sich) qui nuisent à la précision.

### B. Priors de ciblage (données réelles)
- Nouveau champ `priority` par cible dans `config.yaml`, calibré sur les sources ci-dessus,
  avec commentaire de source par cible.
- Tiers d'intensité par oblast (ordinal→cardinal) :
  - Très haut : Kharkiv, Kherson, Zaporizhzhia, Sumy, Dnipropetrovsk (front).
  - Haut : Kyiv (capitale, frappe profonde prioritaire).
  - Moyen : Mykolaiv, Odesa, Poltava.
  - Bas : Lviv, Vinnytsia, ouest (frappes énergie/logistique périodiques).
- **Affinité (classe × type de zone)** : en plus du prior par oblast, une matrice par classe
  de drone (données réelles ci-dessus), p.ex. (multiplicateurs indicatifs) :
  - shahed136 : city 1.0, port 0.9, defense_industry 0.8, power_npp 0.6, airbase 0.5
  - gerbera (≈ shahed, leurre) : city 1.0, port 0.9, defense_industry 0.7, power_npp 0.5, airbase 0.5
  - lancet (militaire/front) : airbase 1.0, defense_industry 0.8, city 0.5, port 0.4, power_npp 0.3
  - fpv_fiber (front, très courte portée) : city 0.7, airbase 0.6, defense_industry 0.5, port 0.3, power_npp 0.2
- Poids final d'une cible = `zone_type_weight` × `priority(oblast/cible)` × `affinité(classe, zone_type)`.
  Appliqué dans `routing.py` (`_target_weights` hub + branche forward). Le modèle apprend ces
  base-rates ET la corrélation classe→type-de-cible : inférer la classe depuis les events affine
  fortement la cible (« c'est un Lancet → base aérienne du front »). Tôt = base-rate, puis la
  géométrie affine → renforce la convergence.

### C. Modèle de capteurs réaliste (5 modalités, vibration supprimée)
`MODALITY_ORDER = [acoustic, optical, rf, das, observer]` (toujours 5 → N_FEAT reste 17).
Placement **non uniforme**, **clippé au territoire UA** (voir D).

| Modalité | Base réelle | Placement | Portée (sim) | Cap | Dist | Détecte FPV fibre |
|---|---|---|---|---|---|---|
| **acoustic** | Sky Fortress + Zvook (2 réseaux, denses) | Très dense : villes/infra + corridors + écran frontière | courte (nœud ≈ cluster local, ~3–5 km abstrait) | ✗ | ✗ | **Oui** (clé) |
| **optical** | Caméras+IA sites/positions | Sites haute valeur **+ large rayon autour**, villes, qq installations frontière | moyenne (~5–9 km) | ✓ | ✓ (mono) | partiel |
| **rf** | EW/SIGINT | Front + sites + villes du front **+ large rayon**, longue portée | longue (~15–25 km) | ✓ | ✗ | **Non** (0) |
| **das** | DAS sur fibre télécom | **Réseau dense** reliant la plupart des villes équipées + antennes vers frontières (sur territoire) | transverse courte (~1–2 km) | ✗ | ✗ | oui (passage proche) |
| **observer** | ePPO + groupes mobiles | Zones peuplées (villes/villages) nationwide | moyenne (~6–8 km, où il y a population) | ✓ (relèvement) | ✗ | faible |

- Émissions par classe (drone_classes) étendues à `observer`, `vibration` retirée partout.
- Matrices de confusion : `observer` diffuse (humain peu précis) ; `das` peu discriminant ;
  `rf` excellent sur émetteurs et **0 sur FPV fibre** (inchangé).
- Acoustique = couche primaire qui « accroche » la plupart des drones (FPV fibre incluse).

### D. Territoire & « pas de prédiction avant détection »
- Ajout `geo.point_in_polygon` (ray-casting). Chargement du polygone **Ukraine** (depuis le
  `borders.geojson` déjà présent, ou un polygone UA dédié dans le générateur).
- `sensors.build_network` ne place QUE des capteurs dont le point est dans le territoire UA.
- Conséquence : le drone part de Russie, **vole sans être détecté**, et la **1ʳᵉ détection =
  1ʳᵉ prédiction** à l'entrée/approche du territoire. Aucun event (donc aucune prédiction)
  avant. L'UI affiche « **AUCUN CONTACT — en vol, non détecté** » jusqu'au premier event.

### E. Visualisation de la fusion multi-capteurs (titre de la démo)
- WS enrichi : chaque détection envoie `sensor_id`, position capteur, modalité, classe
  estimée, confiance, `is_clutter`, **+ position vraie du drone à cet instant** (pour la ligne).
  Tick agrégé : tally par modalité pour la piste courante.
- Carte (écran live) : chaque capteur qui détecte **pulse** + **ligne capteur→drone**
  (couleur = modalité, s'estompe).
- **Panneau « Fusion feed »** : flux des détections par modalité **fusionnant en une piste** ;
  clutter barré/grisé (rejeté) ; bandeau « N capteurs · M modalités → 1 piste ».
- **Contribution par modalité** : compteur acoustic/optical/rf/das/observer pour la piste
  (ex. « RF : 0 — FPV fibre invisible au RF »).
- **Capteurs visibles au zoom, pas en permanence** : dézoomé, le réseau est masqué (sinon
  illisible) ; au-delà d'un seuil de zoom, on affiche les **icônes réelles** par modalité —
  caméra (optique), micro (acoustique), mât/antenne (RF), **câble fibre tracé** (DAS),
  observateur/téléphone (observer). Les détections **pulsent + ligne capteur→drone** quel que
  soit le zoom. Endpoint `/sensors` enrichi (modalité + type) + géométrie des lignes DAS.

### F. UI : un écran centré carte, panneaux latéraux (deux états)
Un **seul écran** centré sur la carte, avec **panneaux latéraux** (pas de page séparée) :
- **État Setup** : panneau de config (gauche) au premier plan — formulaire
  (classe/cible/origine/vitesse) + presets + bouton Lancer ; carte de contexte derrière.
- **État Live** (au lancement) : panneau **gauche = Fusion feed**, panneau **droit = prédiction**
  (top-5 / classe / contribution par modalité), carte au centre (drone, traîne, capteurs+lignes,
  vecteurs menace) ; config repliée avec bouton « reconfigurer ».
- Bascule d'état `view: 'setup' | 'live'` dans `App.jsx` (pas de router).

**Presets de démo hyper-réalistes** (exemples typiques réels, classe↔cible cohérente) :
1. **Shahed-136 → Kyiv** — frappe stratégique nocturne longue distance (capitale/énergie).
2. **Shahed-136 → port d'Odesa** — campagne Mer Noire (sud).
3. **Lancet → base aérienne du front** — munition rôdeuse vs aviation/DCA près du front.
4. **FPV fibre → près du front (Kharkiv)** — très courte portée, **0 RF** (furtif).
Chaque preset utilise `prefer_hit` (choix d'une graine illustrative ; le modèle tourne ensuite
honnêtement sur le scénario).

### G. Précision améliorée (point 5)
- 28 cibles (vs 65) + suppression doublons co-localisés + priors (oblast + classe×zone) → gain
  mécanique important.
- **Dataset ×2 : ~10 000 drones** (l'entraînement est rapide).
- **Modèle plus profond** : LSTM `n_layers` 2 → 3 (hidden 128, dropout maintenu). Toujours
  `concat(last+mean+max)`, N_FEAT=17. Ré-éval par fraction d'observation (top-3 attendu
  nettement > v1).

## Fichiers à modifier
- `dataset_generator/config.yaml` : cibles (28 + `priority`), `drone_classes.*.emission`
  (retire `vibration`, ajoute `observer`), `sensors.modalities` (retire vibration, ajoute
  observer + densités/portées réalistes + paramètres de placement), `sensors.das.lines`
  (réseau dense reliant les villes), matrices de confusion.
- `dataset_generator/sensors.py` : `MODALITY_ORDER`, `build_network` (placement par modalité
  ciblé villes/sites/corridors/front + clip territoire), `simulate_events` (modalité observer).
- `dataset_generator/geo.py` : `point_in_polygon`, chargement polygone UA.
- `dataset_generator/routing.py` : `Target.priority` + multiplication dans `_target_weights`
  et la branche forward.
- `model/` : `train.py` arch `n_layers` 2 → 3 (N_FEAT=17 conservé) ; ré-entraînement sur ~10 000 drones.
- `backend/main.py` : enrichir le payload WS (détail capteur + tally modalité + pos vraie) ;
  `sim_bridge.py` inchangé (territoire géré dans sensors).
- `frontend/` : `App.jsx` (deux vues), `MapView.jsx` (capteurs+lignes, état no-contact),
  nouveaux `FusionFeed.jsx` + `ModalityBreakdown.jsx`, `ControlPanel.jsx` (écran setup).

## Vérification
1. Régénérer le dataset → contrôler : 28 cibles ; 5 modalités (acoustic/optical/rf/das/observer) ;
   FPV-fibre RF = 0 ; capteurs tous dans le territoire ; distribution des cibles cohérente
   avec les priors (Kyiv/Kharkiv en tête).
2. Ré-entraîner → `eval_fractions` : top-3 full-obs attendu nettement > v1 (moins de classes).
3. Test WS : (a) aucun event/prédiction avant l'entrée territoire ; (b) prédiction qui se
   resserre ; (c) lignes capteur→drone + fusion feed + tally par modalité.
4. Frontend : build + dev ; bascule Setup→Live ; état « AUCUN CONTACT ».

## Décisions validées
- Modalité **observer** (ePPO + groupes feu mobiles) en remplacement de vibration — **confirmée**.
- Liste des 28 cibles — **acceptée** (ajustable en cours d'implémentation si besoin).
- 5 modalités `acoustic, optical, rf, das, observer` (N_FEAT reste 17).
- Priors **par oblast ET par (classe × type de zone)**, basés sur données réelles citées.
- **UI = un écran centré carte + panneaux latéraux** (Setup → Live par changement d'état).
- **Capteurs visibles au zoom uniquement** (icônes réelles + câble fibre DAS).
- **Dataset ~10 000 drones + LSTM 3 couches**.
- Presets de démo **hyper-réalistes** (Shahed→Kyiv, Shahed→port d'Odesa, Lancet→base du front, FPV fibre→Kharkiv).
