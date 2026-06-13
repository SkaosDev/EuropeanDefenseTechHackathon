"""
sim_bridge.py — pont vers le simulateur existant (import seulement, jamais modifié).

Charge le "monde" une seule fois (config, classes, origines, cibles, zones DCA, réseau
de capteurs) puis crée des scénarios à la volée. `simulate_drone` tire origine/cible/classe
au hasard ; ici on appelle directement ses sous-fonctions pour pouvoir FORCER un choix
(origine / cible / classe), tout en gardant des défauts "plausibles" via `choose_od`.

Renvoie pour chaque scénario : les lignes de vérité-terrain (pour l'animation côté front)
et les événements triés par `t` (pour le rejeu temps réel au modèle, clutter inclus).
"""
from __future__ import annotations

import os
import threading

import numpy as np
import yaml

from dataset_generator import geo, kinematics, routing, sensors
from dataset_generator.drone_classes import CLASS_ORDER, load_drone_classes, sample_class

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)
CONFIG = os.path.join(REPO, "dataset_generator", "config.yaml")
WORLD_SEED = 1   # même graine que la régénération du dataset -> grille de capteurs cohérente

_world = None
_world_lock = threading.Lock()
_scenarios = {}
_counter = [0]


def load_world():
    """Construit (une fois) le monde partagé. Thread-safe."""
    global _world
    if _world is None:
        with _world_lock:
            if _world is None:
                with open(CONFIG) as f:
                    cfg = yaml.safe_load(f)
                rng = np.random.default_rng(WORLD_SEED)
                targets = routing.build_targets(cfg)
                _world = {
                    "cfg": cfg,
                    "classes": load_drone_classes(cfg),
                    "origins": cfg["origins"],
                    "targets": targets,
                    "dca": routing.build_dca_zones(cfg, targets),
                    "net": sensors.build_network(cfg, rng, targets),
                }
    return _world


def _resolve_target(target, targets):
    if target is None:
        return None
    if isinstance(target, int):
        return targets[target]
    for t in targets:               # par nom
        if t.name == target:
            return t
    try:                            # par dest_id sous forme de chaîne
        return targets[int(target)]
    except (ValueError, IndexError):
        raise ValueError(f"Cible inconnue : {target!r}")


def _find_origin(origins, origin):
    for o in origins:
        if o["name"] == origin:
            return o
    raise ValueError(f"Origine inconnue : {origin!r}")


def _resolve_od(rng, dc, origins, targets, cfg, origin, target):
    """Détermine (olat, olon, oname, tgt) en honorant les choix forcés."""
    tgt = _resolve_target(target, targets)

    # origine explicite
    if origin is not None:
        o = _find_origin(origins, origin)
        if tgt is None:                                   # cible pondérée depuis cette origine
            w = routing._target_weights(o, targets, dc, cfg)
            if w.sum() > 0:
                tgt = targets[rng.choice(len(targets), p=w / w.sum())]
            else:
                tgt = min(targets, key=lambda t: geo.distance_m(o["lat"], o["lon"], t.lat, t.lon))
        return o["lat"], o["lon"], o["name"], tgt

    # ni origine ni cible -> tirage plausible standard
    if tgt is None:
        return routing.choose_od(rng, dc, origins, targets, cfg)

    # cible explicite, origine à déterminer selon le mode de la classe
    if dc.origin_mode == "hub":
        weights = np.array(
            [routing._target_weights(o, targets, dc, cfg)[tgt.dest_id] for o in origins])
        if weights.sum() > 0:
            o = origins[rng.choice(len(origins), p=weights / weights.sum())]
        else:
            o = min(origins, key=lambda o: geo.distance_m(o["lat"], o["lon"], tgt.lat, tgt.lon))
        return o["lat"], o["lon"], o["name"], tgt

    # mode forward : point de lancement avancé (réplique routing.choose_od, branche forward)
    hub = min(origins, key=lambda o: geo.distance_m(tgt.lat, tgt.lon, o["lat"], o["lon"]))
    bearing = geo.initial_bearing(tgt.lat, tgt.lon, hub["lat"], hub["lon"])
    launch_km = dc.range_km * rng.uniform(0.4, 0.85)
    olat, olon = geo.destination_point(tgt.lat, tgt.lon, bearing, launch_km * 1000.0)
    return olat, olon, f"forward:{tgt.oblast}", tgt


# Seuils d'événements "regardables" par classe (FPV fibre émet très peu : ~0.8 evt/scénario).
_MIN_EVENTS = {"shahed136": 8, "gerbera": 6, "lancet": 4, "fpv_fiber": 2}


def _generate_once(rng, w, origin, target, drone_class):
    cfg, classes, origins, targets, dca, net = (
        w["cfg"], w["classes"], w["origins"], w["targets"], w["dca"], w["net"])
    dc = classes[drone_class] if drone_class else sample_class(rng, classes)
    olat, olon, oname, tgt = _resolve_od(rng, dc, origins, targets, cfg, origin, target)
    waypoints = routing.build_waypoints(rng, olat, olon, tgt, cfg)
    corridor = routing.maybe_pick_corridor(rng, olat, olon, tgt, cfg)
    traj = kinematics.simulate_trajectory(rng, olat, olon, waypoints, tgt, dc, cfg, dca, corridor)
    if len(traj["t"]) == 0:
        return None
    n = len(traj["t"])
    gt_rows = [{
        "t": float(traj["t"][i]), "lat": float(traj["lat"][i]), "lon": float(traj["lon"][i]),
        "alt": float(traj["alt"][i]), "speed": float(traj["speed"][i]),
        "bearing": float(traj["bearing"][i]),
    } for i in range(n)]
    events = sorted(sensors.simulate_events(rng, traj, dc, net, cfg, drone_id=0),
                    key=lambda e: e["t"])
    return {"gt_rows": gt_rows, "events": events, "dc": dc, "tgt": tgt, "oname": oname}


def build_scenario(origin=None, target=None, drone_class=None, seed=None,
                   min_events=None, max_tries=25):
    """
    Construit un scénario (NON enregistré). Paramètres absents = tirés au hasard plausible.

    Pour garantir une démo "regardable", on retire plusieurs graines jusqu'à obtenir au
    moins `min_events` événements (défaut dépendant de la classe). On renvoie le meilleur
    essai si le seuil n'est jamais atteint (cas FPV fibre, volontairement furtif).
    """
    w = load_world()
    if drone_class is not None and drone_class not in w["classes"]:
        raise ValueError(f"Classe inconnue : {drone_class!r} (attendu {CLASS_ORDER})")
    if min_events is None:
        min_events = _MIN_EVENTS.get(drone_class, 3)

    best = None
    for attempt in range(max_tries):
        s = None if seed is None else seed + attempt
        scn = _generate_once(np.random.default_rng(s), w, origin, target, drone_class)
        if scn is None:
            continue
        if best is None or len(scn["events"]) > len(best["events"]):
            best = scn
        if len(scn["events"]) >= min_events:
            break
    if best is None:
        raise RuntimeError("impossible de générer une trajectoire (origine/cible incohérentes)")
    return best


def register(scn):
    """Enregistre un scénario et lui attribue un id (consultable par le WS)."""
    sid = _counter[0]
    _counter[0] += 1
    scn["scenario_id"] = sid
    _scenarios[sid] = scn
    return sid


def spawn_scenario(**kwargs):
    """Construit puis enregistre un scénario (chemin par défaut)."""
    scn = build_scenario(**kwargs)
    register(scn)
    return scn


def get_scenario(sid):
    return _scenarios.get(int(sid))
