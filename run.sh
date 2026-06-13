#!/usr/bin/env bash
# Lanceur AEGIS (macOS / Linux).
#   ./run.sh                 menu interactif
#   ./run.sh setup|regen|train|eval|backend|frontend|demo
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
exec "$PY" "$DIR/run.py" "$@"
