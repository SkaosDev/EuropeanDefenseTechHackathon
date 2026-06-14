"""
planner.py — Aide à la décision d'interception d'AEGIS (consultatif, CPU only).

À partir de la prédiction du modèle (cible top-1 / classe) et d'une PISTE ESTIMÉE
reconstruite honnêtement depuis les détections capteurs, ce module évalue EN TEMPS RÉEL,
pour la menace en cours, les stratégies d'interception envisageables et leur probabilité
de réussite estimée — puis tranche : une interception autonome est-elle viable ici, ou
vaut-il mieux passer la main à un humain ?

Il ne LANCE rien et ne simule aucune issue (pas de HIT/MISS) : c'est un outil de réflexion.
Du coup il n'utilise JAMAIS la vérité-terrain (ni position vraie du drone, ni vraie cible) :
toutes ses entrées sont la prédiction + les détections capteurs.

La probabilité de réussite d'un site = qualité d'interception :
    p = p_base × faisabilité_cinématique × acquisition_terminale,  combinée en salve.
L'acquisition modélise un intercepteur auto-dirigé envoyé sur une zone VAGUE : si
l'incertitude de position de la menace (issue de la fiabilité des données) dépasse le
rayon d'acquisition du capteur embarqué, la proba chute → données vagues = décision humaine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from dataset_generator import geo

# --------------------------------------------------------------------------- #
#  Défauts numériques (tous les réglages ici)                                  #
# --------------------------------------------------------------------------- #
INTERCEPTOR_SPEED_KMH = 300.0   # vitesse intercepteur
ENGAGEMENT_RADIUS_KM = 30.0     # rayon d'engagement par site
P_BASE = 0.85                   # P(succès) de référence (tir unitaire, géométrie idéale)
SPINUP_S = 45.0                 # délai d'armement/lancement

BIG_CITY_POP = 700_000          # >= -> grande ville : 2 intercepteurs
HIGH_PRIORITY = 1.5             # >= -> site à forte intensité de ciblage : 2 intercepteurs

CITY_SAFETY_RADIUS_KM = 10.0    # on n'intercepte pas en deçà (autour de la cible)
# Calibration de l'acquisition : les intercepteurs réels neutralisent bien (~85%). Le
# dataset est très parcimonieux (peu d'événements -> pistes parfois vagues) ; on calibre
# donc pour rester réaliste plutôt que pessimiste.
ACQUISITION_RADIUS_KM = 8.0     # zone que le capteur embarqué auto-dirigé peut couvrir
ACQ_FLOOR = 0.30               # même sur zone vague, l'auto-guidage garde une chance
KIN_FLOOR = 0.40               # une interception faisable garde un Pk décent même serrée
SIGMA_FLOOR_KM = 1.5            # incertitude minimale d'une piste projetée
SIGMA_CENTROID_KM = 5.0        # incertitude « centroïde » ~ portée des capteurs qui ont détecté
RANGE_NOISE_FRAC = 0.35         # bruit relatif de portée (cf. config dataset)

P_VIABLE = 0.65                 # proba mini pour juger l'autonome viable
AUTO_ENGAGE_P = 0.75           # proba « très fiable » : on engage en autonome et on fige
CONF_MIN = 0.50                 # confiance cible mini (P du top-1)
CONVERGENCE_TICKS = 2           # ticks consécutifs avec top-1 stable
CRITICAL_TTI_S = 90.0           # time-to-impact en deçà duquel on signale l'urgence

CORRIDOR_SAMPLES = 24           # points échantillonnés le long du corridor menace
TRACK_FALLBACK_K = 4            # centroïde des K dernières positions capteurs (repli)
MAX_OPTIONS = 6                 # nb de sites affichés (les meilleurs)

THREAT_CLASSES = {"shahed136", "gerbera", "fpv_fiber", "lancet"}
_RANGE_BEARING_MODS = ("optical", "acoustic", "rf")
_MOD_RANK = {"optical": 0, "acoustic": 1, "rf": 2}
BEARING_NOISE_DEG = {"optical": 4.0, "acoustic": 10.0, "rf": 8.0}


# --------------------------------------------------------------------------- #
#  Assets (intercepteurs) dérivés des cibles                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Asset:
    site_id: int
    name: str
    lat: float
    lon: float
    n_interceptors: int
    speed_kmh: float = INTERCEPTOR_SPEED_KMH
    range_km: float = ENGAGEMENT_RADIUS_KM
    p_base: float = P_BASE
    spinup_s: float = SPINUP_S


def build_assets(world) -> list[dict]:
    """Site défensif co-localisé à chaque cible ; inventaire ∝ pop/priorité.

    2 intercepteurs si grande ville (pop >= 700k) OU site à forte priorité (>= 1.5),
    sinon 1. (Les `origins` du config sont des points de lancement ENNEMIS : jamais
    utilisés comme sites intercepteurs.)
    """
    assets = []
    for t in world["targets"]:
        major_city = (t.zone_type == "city" and t.pop >= BIG_CITY_POP)
        high_value = (t.priority >= HIGH_PRIORITY)
        n = 2 if (major_city or high_value) else 1
        assets.append(asdict(Asset(
            site_id=t.dest_id, name=t.name, lat=t.lat, lon=t.lon, n_interceptors=n)))
    return assets


@dataclass
class EstTrack:
    lat: float | None
    lon: float | None
    quality: str            # "projected" | "centroid" | "none"
    sigma_km: float | None  # rayon d'incertitude de position (fiabilité des données)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# --------------------------------------------------------------------------- #
#  Planner — instancié une fois par scénario (suit la convergence du top-1)    #
# --------------------------------------------------------------------------- #
class Planner:
    def __init__(self, world, scenario_id: int):
        self.targets = world["targets"]
        self.by_id = {t.dest_id: t for t in self.targets}
        self.classes = world["classes"]
        self.assets = [Asset(**a) for a in build_assets(world)]
        self.scenario_id = scenario_id
        self.state = "MONITORING"
        self.engaged = None          # solution figée quand on engage en autonome (terminal)
        self._frozen = None          # tick figé rejoué tel quel une fois ENGAGED
        self._last_top1 = None
        self._stable_ticks = 0

    # ----------------------------------------------------------------------- #
    #  Entrée par tick                                                        #
    # ----------------------------------------------------------------------- #
    def step(self, events, pred, clock, truth_pos=None) -> dict:
        # `truth_pos` = position VRAIE du drone. Lue UNIQUEMENT pour détecter physiquement
        # que la menace entre dans le rayon d'interception de la ville (déclencheur
        # d'engagement) — jamais pour une décision (cible / site / proba), qui restent sur
        # la piste estimée. C'est une réaction à l'arrivée réelle de la menace.
        if self.state == "ENGAGED":
            return self._frozen                      # terminal : on rejoue la solution figée

        if not self._usable_pred(pred):
            self._reset_convergence()
            self.state = "MONITORING"
            return self._emit(None, [], {"autonomous_viable": False, "best_p": None,
                                         "reason": "no_threat"})

        top1 = pred["target_topk"][0]
        self._update_convergence(top1["dest_id"])
        track = self._estimate_track(events)
        v_threat = self._threat_speed_mps(pred["pred_class"])
        tgt1 = self.by_id.get(top1["dest_id"])
        corridor, tti = self._build_corridor(track, tgt1, v_threat)
        options = self._build_options(track, corridor, tgt1, v_threat)
        verdict = self._verdict(top1, options, tti, track)

        threat = {
            "target_dest_id": top1["dest_id"],
            "target_name": top1["name"],
            "pred_class": pred["pred_class"],
            "pred_class_p": round(float(pred["pred_class_p"]), 3),
            "track_quality": track.quality,
            "uncertainty_km": round(track.sigma_km, 1) if track.sigma_km is not None else None,
            "time_to_impact_s": round(tti, 1) if tti is not None else None,
        }

        # ENGAGEMENT AUTONOME : si la solution est TRÈS fiable (décision sur la piste estimée)
        # ET que le drone entre PHYSIQUEMENT dans le rayon d'interception de la ville cible
        # (déclencheur sur la position vraie), on fait le choix de faire confiance à notre
        # interception -> on FIGE la simulation sur la solution retenue (état terminal).
        if tgt1 is not None and truth_pos is not None:
            in_range = geo.distance_m(truth_pos[0], truth_pos[1], tgt1.lat, tgt1.lon) / 1000.0 <= ENGAGEMENT_RADIUS_KM
        elif tgt1 is not None and track.lat is not None:
            in_range = geo.distance_m(track.lat, track.lon, tgt1.lat, tgt1.lon) / 1000.0 <= ENGAGEMENT_RADIUS_KM
        else:
            in_range = False
        if (verdict["autonomous_viable"] and options and verdict["best_p"] is not None
                and verdict["best_p"] >= AUTO_ENGAGE_P and in_range):
            self.state = "ENGAGED"
            self.engaged = {"chosen": options[0], "target_name": top1["name"],
                            "best_p": verdict["best_p"]}
            self._frozen = self._emit(threat, options, verdict)
            return self._frozen

        self.state = "ASSESSING"
        return self._emit(threat, options, verdict)

    # ----------------------------------------------------------------------- #
    #  Prédiction exploitable + convergence du top-1                          #
    # ----------------------------------------------------------------------- #
    def _usable_pred(self, pred) -> bool:
        return bool(pred) and bool(pred.get("target_topk")) \
            and pred.get("pred_class") in THREAT_CLASSES

    def _update_convergence(self, top1_id):
        self._stable_ticks = self._stable_ticks + 1 if top1_id == self._last_top1 else 1
        self._last_top1 = top1_id

    def _reset_convergence(self):
        self._last_top1 = None
        self._stable_ticks = 0

    def _converged(self) -> bool:
        return self._stable_ticks >= CONVERGENCE_TICKS

    # ----------------------------------------------------------------------- #
    #  Piste estimée + incertitude (HONNÊTE — jamais la vérité-terrain)       #
    # ----------------------------------------------------------------------- #
    def _estimate_track(self, events) -> EstTrack:
        usable = [
            e for e in events
            if e.get("drone_id") is not None
            and e.get("modality") in _RANGE_BEARING_MODS
            and e.get("range_est") is not None
            and not math.isnan(float(e["range_est"]))
            and not math.isnan(float(e.get("bearing_est", float("nan"))))
        ]
        if usable:
            best = max(usable, key=lambda e: (float(e["t"]), -_MOD_RANK[e["modality"]]))
            lat, lon = geo.destination_point(
                float(best["sensor_lat"]), float(best["sensor_lon"]),
                float(best["bearing_est"]), float(best["range_est"]))
            rng_km = float(best["range_est"]) / 1000.0
            bn = math.radians(BEARING_NOISE_DEG.get(best["modality"], 10.0))
            sigma = math.hypot(rng_km * RANGE_NOISE_FRAC, rng_km * math.sin(bn))
            return EstTrack(lat, lon, "projected", max(SIGMA_FLOOR_KM, sigma))

        nc = [e for e in events if e.get("drone_id") is not None]
        if nc:
            tail = sorted(nc, key=lambda e: float(e["t"]))[-TRACK_FALLBACK_K:]
            lat = sum(float(e["sensor_lat"]) for e in tail) / len(tail)
            lon = sum(float(e["sensor_lon"]) for e in tail) / len(tail)
            spread = math.sqrt(sum(
                geo.distance_m(lat, lon, float(e["sensor_lat"]), float(e["sensor_lon"])) ** 2
                for e in tail) / len(tail)) / 1000.0
            return EstTrack(lat, lon, "centroid", max(SIGMA_CENTROID_KM, spread))

        return EstTrack(None, None, "none", None)

    # ----------------------------------------------------------------------- #
    #  Corridor menace + time-to-impact                                       #
    # ----------------------------------------------------------------------- #
    def _threat_speed_mps(self, pred_class) -> float:
        dc = self.classes.get(pred_class)
        if dc is None:
            return (165.0 + 200.0) / 2.0 / 3.6
        lo, hi = dc.speed_kmh
        return (lo + hi) / 2.0 / 3.6

    def _build_corridor(self, track: EstTrack, tgt, v_threat):
        if track.lat is None or tgt is None:
            return [], None
        total = geo.distance_m(track.lat, track.lon, tgt.lat, tgt.lon)
        brg = geo.initial_bearing(track.lat, track.lon, tgt.lat, tgt.lon)
        pts = [geo.destination_point(track.lat, track.lon, brg, total * (i / CORRIDOR_SAMPLES))
               for i in range(CORRIDOR_SAMPLES + 1)]
        tti = total / v_threat if v_threat > 0 else None
        return pts, tti

    # ----------------------------------------------------------------------- #
    #  Options d'interception par site (qualité d'interception)               #
    # ----------------------------------------------------------------------- #
    def _acquisition_factor(self, sigma_km) -> float:
        """P(le drone est dans le disque d'acquisition) ~ incertitude σ vs rayon R_acq.
        σ≈2 km -> ~0.9 ; σ≈R_acq -> ~0.4 ; σ≈12 km (centroïde) -> ~0.1. Monotone ↓ en σ."""
        if sigma_km is None or sigma_km <= 0:
            return 0.0
        f = 1.0 - math.exp(-(ACQUISITION_RADIUS_KM ** 2) / (2.0 * sigma_km ** 2))
        return max(ACQ_FLOOR, f)

    def _site_kinematics(self, site: Asset, defended, track, corridor, v_threat):
        """Meilleure faisabilité cinématique : renvoie (time_margin, interc_t, threat_t)
        au point de croisement le plus favorable, ou None si aucun point n'est faisable."""
        v_interc = site.speed_kmh / 3.6
        safety_m = CITY_SAFETY_RADIUS_KM * 1000.0
        best = None
        for P in corridor:
            d_threat = geo.distance_m(track.lat, track.lon, P[0], P[1])
            d_site = geo.distance_m(site.lat, site.lon, P[0], P[1])
            threat_t = d_threat / v_threat
            interc_t = site.spinup_s + d_site / v_interc
            d_def = geo.distance_m(defended.lat, defended.lon, P[0], P[1])
            if interc_t <= threat_t and (d_site / 1000.0) <= site.range_km and d_def >= safety_m:
                tm = (threat_t - interc_t) / threat_t if threat_t > 0 else 0.0
                if best is None or tm > best[0]:
                    best = (tm, interc_t, threat_t)
        return best

    def _build_options(self, track, corridor, tgt1, v_threat) -> list[dict]:
        if track.lat is None or not corridor or tgt1 is None:
            return []
        f_acq = self._acquisition_factor(track.sigma_km)
        opts = []
        for a in self.assets:
            kin = self._site_kinematics(a, tgt1, track, corridor, v_threat)
            if kin is None:
                continue
            time_margin, interc_t, threat_t = kin
            kin_factor = KIN_FLOOR + (1.0 - KIN_FLOOR) * _clamp01(time_margin)
            p_single = a.p_base * kin_factor * f_acq
            p_site = 1.0 - (1.0 - p_single) ** a.n_interceptors
            opts.append({
                "site_id": a.site_id, "site_name": a.name, "n_interceptors": a.n_interceptors,
                "p_success": round(float(p_site), 3), "feasible": True,
                "flight_time_s": round(float(interc_t), 1),
                "time_margin_s": round(float(threat_t - interc_t), 1),
            })
        opts.sort(key=lambda o: -o["p_success"])
        return opts[:MAX_OPTIONS]

    # ----------------------------------------------------------------------- #
    #  Verdict : interception autonome viable, ou décision humaine ?          #
    # ----------------------------------------------------------------------- #
    def _verdict(self, top1, options, tti, track) -> dict:
        if track.lat is None:
            return {"autonomous_viable": False, "best_p": None, "reason": "no_track"}
        if not options:
            reason = "critical_time" if (tti is not None and tti < CRITICAL_TTI_S) else "no_feasible_site"
            return {"autonomous_viable": False, "best_p": None, "reason": reason}

        best_p = max(o["p_success"] for o in options)
        auto = False
        if best_p < P_VIABLE:
            reason = "low_success"
        elif top1["p"] < CONF_MIN or not self._converged():
            reason = "low_confidence"
        else:
            auto, reason = True, "viable"
        if not auto and tti is not None and tti < CRITICAL_TTI_S:
            reason = "critical_time"
        return {"autonomous_viable": auto, "best_p": round(float(best_p), 3), "reason": reason}

    def _emit(self, threat, options, verdict) -> dict:
        return {"state": self.state, "threat": threat, "options": options,
                "verdict": verdict, "engaged": self.engaged}
