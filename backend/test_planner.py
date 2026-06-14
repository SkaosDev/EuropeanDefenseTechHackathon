"""
test_planner.py — tests offline de l'aide à la décision d'interception (sans WebSocket).

Vérifie : MONITORING -> ASSESSING ; que la proba par site reflète la FIABILITÉ des données
(piste précise vs vague) ; que le verdict bascule autonome/humain selon proba + confiance ;
et qu'aucune décision ne lit la vérité-terrain (events sans clé 'drone_pos', Planner sans
vraie cible).

Lancer :  python -m pytest backend/test_planner.py -q   (depuis la racine du repo)
"""
import pytest

from dataset_generator import geo
from backend import sim_bridge
from backend.planner import (
    Planner, build_assets, P_VIABLE, CONF_MIN, BIG_CITY_POP,
)


@pytest.fixture(scope="module")
def world():
    return sim_bridge.load_world()


def fake_pred(top1_id, name, p=0.92, cls="shahed136", cls_p=0.96, others=()):
    topk = [{"dest_id": top1_id, "name": name, "p": p}]
    for did, nm, pp in others:
        topk.append({"dest_id": did, "name": nm, "p": pp})
    return {"target_topk": topk, "pred_class": cls, "pred_class_p": cls_p,
            "pred_future": [[0.0, 0.0]] * 12}


def fake_event(t, drone_id, slat, slon, brg, rng_m, mod="optical"):
    # aucune clé 'drone_pos'/vérité-terrain
    return {"t": t, "drone_id": drone_id, "sensor_id": "s0",
            "sensor_lat": slat, "sensor_lon": slon, "modality": mod,
            "est_class": "shahed136", "confidence": 0.8, "bearing_est": brg, "range_est": rng_m}


def event_precise(target, bearing_from_target, dist_m, t=1.0):
    """Détection optique précise dont la projection tombe à `dist_m` de `target`."""
    est_lat, est_lon = geo.destination_point(target.lat, target.lon, bearing_from_target, dist_m)
    slat, slon = est_lat - 0.001, est_lon
    brg = geo.initial_bearing(slat, slon, est_lat, est_lon)
    rng = geo.distance_m(slat, slon, est_lat, est_lon)
    return fake_event(t, 0, slat, slon, brg, rng, mod="optical")


def event_vague(target, bearing_from_target, dist_m, t=1.0):
    """Détection 'das' (sans portée) -> piste centroïde, position vague."""
    slat, slon = geo.destination_point(target.lat, target.lon, bearing_from_target, dist_m)
    return fake_event(t, 0, slat, slon, 0.0, float("nan"), mod="das")


def assess(p, ev, pred, ticks=2):
    """Joue `ticks` fois pour converger le top-1, renvoie le dernier intervention."""
    out = None
    for k in range(1, ticks + 1):
        out = p.step(ev, pred, clock=float(k))
    return out


# --------------------------------------------------------------------------- #
def test_build_assets_inventory(world):
    assets = {a["site_id"]: a for a in build_assets(world)}
    kyiv = world["targets"][0]
    assert kyiv.name == "Kyiv" and assets[0]["n_interceptors"] == 2
    npp = next(t for t in world["targets"] if t.name == "Zaporizhzhia NPP")
    assert assets[npp.dest_id]["n_interceptors"] == 1
    assert len(assets) == len(world["targets"])


def test_monitoring_without_prediction(world):
    p = Planner(world, scenario_id=1)
    out = p.step([], None, clock=0.0)
    assert out["state"] == "MONITORING"
    assert out["options"] == [] and out["threat"] is None
    assert out["verdict"]["autonomous_viable"] is False


def test_assessing_precise_track_is_viable(world):
    kyiv = world["targets"][0]
    ev = [event_precise(kyiv, 90.0, 180_000.0)]
    pred = fake_pred(kyiv.dest_id, kyiv.name, p=0.92)
    p = Planner(world, scenario_id=2)
    out = assess(p, ev, pred, ticks=2)

    assert out["state"] == "ASSESSING"
    assert out["threat"]["target_name"] == "Kyiv"
    assert out["threat"]["track_quality"] == "projected"
    assert out["options"], "des sites capables attendus"
    # triées par proba décroissante
    ps = [o["p_success"] for o in out["options"]]
    assert ps == sorted(ps, reverse=True)
    # le site de la cible elle-même fait partie des options
    assert any(o["site_id"] == kyiv.dest_id for o in out["options"])
    assert out["verdict"]["autonomous_viable"] is True
    assert out["verdict"]["best_p"] >= P_VIABLE


def test_data_reliability_drives_probability(world):
    """Même géométrie, mais piste VAGUE (centroïde) -> incertitude plus grande et proba
    plus basse qu'avec une piste précise. La fiabilité des données module la proba."""
    kyiv = world["targets"][0]
    pred = fake_pred(kyiv.dest_id, kyiv.name, p=0.92)

    p1 = Planner(world, scenario_id=3)
    precise = assess(p1, [event_precise(kyiv, 90.0, 180_000.0)], pred)

    p2 = Planner(world, scenario_id=4)
    vague = assess(p2, [event_vague(kyiv, 90.0, 180_000.0)], pred)

    assert precise["threat"]["track_quality"] == "projected"
    assert vague["threat"]["track_quality"] == "centroid"
    assert vague["threat"]["uncertainty_km"] > precise["threat"]["uncertainty_km"]
    assert vague["verdict"]["best_p"] < precise["verdict"]["best_p"]


def test_low_target_confidence_recommends_human(world):
    kyiv = world["targets"][0]
    ev = [event_precise(kyiv, 90.0, 180_000.0)]
    pred = fake_pred(kyiv.dest_id, kyiv.name, p=CONF_MIN - 0.1)   # confiance trop basse
    p = Planner(world, scenario_id=5)
    out = assess(p, ev, pred, ticks=2)
    assert out["verdict"]["autonomous_viable"] is False
    assert out["verdict"]["reason"] == "low_confidence"


def test_no_feasible_site_when_too_close(world):
    kyiv = world["targets"][0]
    ev = [event_precise(kyiv, 180.0, 3_000.0)]   # ~3 km : dans le rayon de sécurité
    pred = fake_pred(kyiv.dest_id, kyiv.name, p=0.92)
    p = Planner(world, scenario_id=6)
    out = assess(p, ev, pred, ticks=2)
    assert out["options"] == []
    assert out["verdict"]["autonomous_viable"] is False
    assert out["verdict"]["reason"] in ("critical_time", "no_feasible_site")


def test_engages_when_threat_enters_city_range(world):
    kyiv = world["targets"][0]
    ev = [event_precise(kyiv, 90.0, 180_000.0)]          # piste précise, solution très fiable
    pred = fake_pred(kyiv.dest_id, kyiv.name, p=0.92)
    p = Planner(world, scenario_id=8)
    p.step(ev, pred, clock=1.0)                           # convergence
    # viable mais drone encore loin (vérité ~220 km) -> on n'engage pas
    out = p.step(ev, pred, clock=2.0, truth_pos=[kyiv.lat - 2.0, kyiv.lon])
    assert out["state"] == "ASSESSING" and out["verdict"]["autonomous_viable"] is True
    # le drone entre dans le rayon d'interception de la ville (vérité ~20 km) -> ENGAGED figé
    near = geo.destination_point(kyiv.lat, kyiv.lon, 90.0, 20_000.0)
    out = p.step(ev, pred, clock=3.0, truth_pos=list(near))
    assert out["state"] == "ENGAGED"
    assert out["engaged"]["best_p"] >= 0.80
    assert out["engaged"]["chosen"]["site_id"] == kyiv.dest_id
    # terminal : rejoue la solution figée
    assert p.step(ev, pred, clock=4.0, truth_pos=[kyiv.lat, kyiv.lon])["state"] == "ENGAGED"


def test_no_ground_truth_dependency(world):
    kyiv = world["targets"][0]
    ev = [event_precise(kyiv, 90.0, 120_000.0)]
    assert "drone_pos" not in ev[0]
    p = Planner(world, scenario_id=7)                 # aucune vraie cible passée
    out = p.step(ev, fake_pred(kyiv.dest_id, kyiv.name), clock=1.0)
    assert out["state"] in ("MONITORING", "ASSESSING")
