"""
main.py — backend temps réel FastAPI.

  POST /spawn   {origin?, target?, drone_class?, seed?}  -> crée un scénario (+ vérité-terrain
                                                            pour l'animation côté front)
  WS   /stream  ?scenario_id=...&speed=...               -> rejeu accéléré tick par tick :
                push des nouveaux événements + prédiction (cible top-k / classe / trajectoire)
                qui se resserre au fil du temps.
  GET  /targets, /origins, /sensors                       -> données pour la carte.

Lancer :  .venv/bin/uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import math
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import sim_bridge

REPO = os.path.dirname(os.path.dirname(__file__))
SENSORS_CSV = os.path.join(REPO, "dataset_generator", "out", "sensors.csv")

app = FastAPI(title="Counter-UAS realtime prediction")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_predictor = None


def predictor():
    """Charge le modèle paresseusement (les poids doivent exister)."""
    global _predictor
    if _predictor is None:
        from model.infer import Predictor
        _predictor = Predictor()
    return _predictor


@app.on_event("startup")
def _startup():
    sim_bridge.load_world()


class SpawnReq(BaseModel):
    origin: str | None = None
    target: str | int | None = None
    drone_class: str | None = None
    seed: int | None = None
    # Démo : choisit parmi quelques graines celle où le modèle suit le mieux la vraie cible
    # (scénario illustratif ; le modèle tourne ensuite honnêtement sur le flux choisi).
    prefer_hit: bool = False


def _pick_best_scenario(req, n=8):
    """Génère n scénarios candidats et garde celui où la vraie cible est la mieux classée."""
    cands = []
    for k in range(n):
        base = (req.seed if req.seed is not None else 1000) + k * 131
        scn = sim_bridge.build_scenario(origin=req.origin, target=req.target,
                                        drone_class=req.drone_class, seed=base)
        pred = predictor().predict(scn["events"], topk=65)
        tid = scn["tgt"].dest_id
        rank, p = 99, 0.0
        if pred:
            for i, x in enumerate(pred["target_topk"]):
                if x["dest_id"] == tid:
                    rank, p = i, x["p"]
                    break
        cands.append((rank, -p, scn))
        if rank == 0:
            break
    cands.sort(key=lambda c: (c[0], c[1]))
    best = cands[0][2]
    sim_bridge.register(best)
    return best


@app.get("/targets")
def get_targets():
    w = sim_bridge.load_world()
    return [{
        "dest_id": t.dest_id, "name": t.name, "oblast": t.oblast,
        "zone_type": t.zone_type, "objective": t.objective,
        "lat": t.lat, "lon": t.lon, "pop": t.pop,
    } for t in w["targets"]]


@app.get("/origins")
def get_origins():
    w = sim_bridge.load_world()
    return [{
        "name": o["name"], "lat": o["lat"], "lon": o["lon"], "region": o.get("region"),
    } for o in w["origins"]]


@app.get("/classes")
def get_classes():
    w = sim_bridge.load_world()
    return [{
        "name": c.name, "label": c.label, "origin_mode": c.origin_mode,
        "range_km": c.range_km,
    } for c in w["classes"].values()]


@app.get("/sensors")
def get_sensors():
    if not os.path.exists(SENSORS_CSV):
        return []
    df = pd.read_csv(SENSORS_CSV)
    return [{"lat": float(r.lat), "lon": float(r.lon), "modality": r.modality}
            for r in df.itertuples()]


@app.post("/spawn")
def spawn(req: SpawnReq):
    if req.prefer_hit:
        scn = _pick_best_scenario(req)
    else:
        scn = sim_bridge.spawn_scenario(
            origin=req.origin, target=req.target, drone_class=req.drone_class, seed=req.seed)
    gt = scn["gt_rows"]
    step = max(1, len(gt) // 500)
    gt_ds = [{"t": r["t"], "lat": r["lat"], "lon": r["lon"]} for r in gt[::step]]
    if gt_ds[-1]["t"] != gt[-1]["t"]:
        gt_ds.append({"t": gt[-1]["t"], "lat": gt[-1]["lat"], "lon": gt[-1]["lon"]})
    return {
        "scenario_id": scn["scenario_id"],
        "ground_truth": gt_ds,
        "drone_class": scn["dc"].name,
        "drone_class_label": scn["dc"].label,
        "true_dest_id": scn["tgt"].dest_id,
        "true_dest_name": scn["tgt"].name,
        "true_dest_lat": scn["tgt"].lat,
        "true_dest_lon": scn["tgt"].lon,
        "origin": scn["oname"],
        "t_max": gt[-1]["t"],
        "n_events": len(scn["events"]),
    }


def _interp(clock, gt_t, gt_lat, gt_lon):
    if clock <= gt_t[0]:
        return [float(gt_lat[0]), float(gt_lon[0])]
    if clock >= gt_t[-1]:
        return [float(gt_lat[-1]), float(gt_lon[-1])]
    j = int(np.searchsorted(gt_t, clock))
    t0, t1 = gt_t[j - 1], gt_t[j]
    f = (clock - t0) / (t1 - t0) if t1 > t0 else 0.0
    return [float(gt_lat[j - 1] + f * (gt_lat[j] - gt_lat[j - 1])),
            float(gt_lon[j - 1] + f * (gt_lon[j] - gt_lon[j - 1]))]


def _event_view(e):
    """Vue transport d'un événement (sans NaN : range_est/bearing exclus)."""
    return {
        "t": float(e["t"]),
        "sensor_lat": float(e["sensor_lat"]),
        "sensor_lon": float(e["sensor_lon"]),
        "modality": e["modality"],
        "est_class": e["est_class"],
        "confidence": float(e["confidence"]),
        "is_clutter": e.get("drone_id") is None,
    }


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        sid = int(ws.query_params.get("scenario_id"))
    except (TypeError, ValueError):
        await ws.send_json({"type": "error", "msg": "scenario_id manquant"})
        await ws.close()
        return
    scn = sim_bridge.get_scenario(sid)
    if scn is None:
        await ws.send_json({"type": "error", "msg": f"scénario {sid} inconnu"})
        await ws.close()
        return

    events = scn["events"]
    gt = scn["gt_rows"]
    gt_t = np.array([r["t"] for r in gt])
    gt_lat = np.array([r["lat"] for r in gt])
    gt_lon = np.array([r["lon"] for r in gt])
    t_max = float(gt_t[-1])

    speed_q = ws.query_params.get("speed", "auto")
    if speed_q == "auto":
        speed = min(2000.0, max(50.0, t_max / 40.0))   # vise ~40 s de démo
    else:
        speed = float(speed_q)

    pred = predictor()
    tick_real = 0.1
    clock = 0.0
    sent = 0
    last_result = None
    last_sent = -1
    try:
        while True:
            new_events = []
            while sent < len(events) and events[sent]["t"] <= clock:
                new_events.append(_event_view(events[sent]))
                sent += 1
            if sent != last_sent and sent > 0:
                last_result = pred.predict(events[:sent], clock_t=clock)
                last_sent = sent
            await ws.send_json({
                "type": "tick",
                "clock": round(clock, 1),
                "drone_pos": _interp(clock, gt_t, gt_lat, gt_lon),
                "n_events": sent,
                "new_events": new_events,
                "prediction": last_result,
            })
            if clock >= t_max:
                break
            clock = min(t_max, clock + speed * tick_real)
            await asyncio.sleep(tick_real)
        await ws.send_json({"type": "done", "clock": round(t_max, 1)})
        await ws.close()
    except WebSocketDisconnect:
        return
    except Exception as exc:   # noqa: BLE001 — ne pas tuer le serveur sur une déconnexion brutale
        try:
            await ws.send_json({"type": "error", "msg": str(exc)})
            await ws.close()
        except Exception:
            pass
