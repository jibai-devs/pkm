#!/usr/bin/env bash
# Evolve `default_dragapult_darwinian` against a past submission bundle.
#
# Usage:
#   darwinian_ml/evolve.sh [-- EXTRA_ARGS...]
#   darwinian_ml/evolve.sh -- --generations 50 --population 16
#   darwinian_ml/evolve.sh -- --max-hours 6
#
# EXTRA_ARGS are forwarded verbatim to `python -m darwinian_ml.evolve`.
# Runs in the background with its output streamed here; stop gracefully with
#   touch darwinian_ml/runs/default_dragapult_darwinian/STOP
# which finishes the generation in flight, keeps the best checkpoint, and exits.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Test for a leading "--" before consuming $1, or `evolve.sh -- --foo` would
# swallow the separator as a positional argument.
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

if [[ -f ".venv/Scripts/python.exe" ]]; then
    PY=".venv/Scripts/python.exe"
elif [[ -f ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

OUT="darwinian_ml/runs/default_dragapult_darwinian"
mkdir -p "$OUT"
LOGFILE="$OUT/evolve.log"
rm -f "$OUT/STOP"

echo "Evolving default_dragapult_darwinian. Log: ${LOGFILE}"
echo "Stop with: touch ${OUT}/STOP"

"$PY" -m darwinian_ml.evolve "${EXTRA_ARGS[@]}" >"$LOGFILE" 2>&1 &
PID=$!

tail -f "$LOGFILE" &
TAIL=$!

cleanup() {
    touch "$OUT/STOP"
    if kill -0 "$PID" 2>/dev/null; then
        echo ""
        echo "stopping after the generation in flight..."
        wait "$PID" 2>/dev/null || true
    fi
    kill "$TAIL" 2>/dev/null || true
    rm -f "$OUT/STOP"
    echo "stopped. best checkpoint: ${OUT}/best.pt"
}
trap cleanup INT TERM

while kill -0 "$PID" 2>/dev/null; do sleep 2; done
kill "$TAIL" 2>/dev/null || true
echo "evolution exited -- see ${LOGFILE}"
