# AEGIS — Counter-UAS Early-Warning Demo

Real-time prediction of which sensitive target an incoming drone is heading for, from a
stream of **noisy, simulated sensor detections** (optical / acoustic / vibration / DAS /
RF), with ~28 % clutter mixed in. Everything is simulated — no real sensors.

```
dataset_generator/   # EXISTING world+physics+sensor simulator (imported, never modified)
model/               # LSTM multi-head: target (65) + drone class (4) + future path (12)
backend/             # FastAPI: POST /spawn + WS /stream replays a scenario tick-by-tick
frontend/            # Vite + React + Leaflet command-center UI
```

The model **never sees ground truth** (origin / target / trajectory) — only the event
stream. The "wow": as the drone advances and more detections arrive, the predicted target
distribution **tightens** toward the true objective.

## One-time setup

```bash
# 1. Python env (stable 3.13) + ML/web deps
python3 -m venv .venv
.venv/bin/pip install -r requirements-ml.txt

# 2. (already done) regenerate dataset — 5000 drones
.venv/bin/python -m dataset_generator.main --n-drones 5000 --seed 1 --out dataset_generator/out --no-viz

# 3. train the model (CPU, a few minutes) -> model/weights/model.pt
.venv/bin/python -m model.train --epochs 60

# 4. frontend deps (Node 24 + yarn)
cd frontend && yarn install && cd ..
```

## Run the demo (two terminals)

```bash
# Terminal A — backend
.venv/bin/uvicorn backend.main:app --port 8000

# Terminal B — frontend
cd frontend && yarn dev      # http://localhost:5173
```

Open http://localhost:5173, click a **demo preset** (or build a custom launch), and watch
the drone move, the blue trail grow, and the red threat-vectors / % bars converge.

## Demo scenarios (preset buttons)
1. **Shahed-136 · North → Kyiv** — long-range one-way attack drone.
2. **Gerbera wave · South → Odesa** — decoys saturating air defense.
3. **FPV fibre · near the front** — short range, **zero RF signature** (highlights the
   stealthy fiber-optic threat the RF net cannot see).

## Validation commands
```bash
.venv/bin/python -m model.infer                 # replay a random scenario, print top-5 prediction
.venv/bin/python -m model.eval_fractions        # accuracy by observation fraction (the convergence story)
```

## Notes
- **Train/serve parity**: inference imports `dataset_generator.sequence_prep._encode_window`
  verbatim (same 17 features, same normalization, same last-64 truncation). Never reimplemented.
- **Offline tiles fallback**: country borders are vendored at `frontend/public/borders.geojson`,
  so the map still renders if venue wifi can't load CARTO tiles.
- **CPU only**, no GPU. `dataset_generator/` is imported, never modified.
