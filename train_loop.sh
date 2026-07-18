#!/usr/bin/env bash
# Train an agent until stopped, or for a fixed wall-clock duration.
# Usage: ./train_loop.sh [agent] [games] [eval_every] [duration_seconds]
#   agent             agent profile name (default: 03_pult_munki)
#   games             games per iteration (default: 16, matches `just train`)
#   eval_every        iterations between evals + checkpoint saves (default: 10)
#   duration_seconds  auto-stop after this long; 0 = run until stopped (default: 0)
#
# Stop it (before the duration elapses, if any) one of two ways:
#   - running interactively in a real terminal: type "stop" + Enter
#   - running detached (e.g. launched in the background for you):
#     `touch agents/<agent>/STOP`
#
# Runs `pkm train` as a background process with an effectively unbounded
# --iterations count. Progress is only checkpointed every `eval_every`
# iterations (pkm/rl/train.py), so stopping can lose up to that many
# iterations of the current run -- it never corrupts or rewinds anything
# already saved.

set -u

AGENT="${1:-03_pult_munki}"
GAMES="${2:-16}"
EVAL_EVERY="${3:-10}"
DURATION="${4:-0}"
LOGFILE="agents/${AGENT}/train_loop.log"
STOPFILE="agents/${AGENT}/STOP"

mkdir -p "agents/${AGENT}"
rm -f "$STOPFILE"

END_TS=0
if [ "$DURATION" -gt 0 ]; then
    END_TS=$(($(date +%s) + DURATION))
fi

echo "Training '${AGENT}' (games=${GAMES}, eval_every=${EVAL_EVERY})."
if [ "$DURATION" -gt 0 ]; then
    echo "Auto-stopping after ${DURATION}s."
fi
if [ -t 0 ]; then
    echo "Type 'stop' + Enter at any time to stop. Log: ${LOGFILE}"
else
    echo "Not an interactive shell -- stop by creating: ${STOPFILE}"
fi

uv run pkm train --agent "$AGENT" --iterations 1000000 --games "$GAMES" \
    --eval-every "$EVAL_EVERY" >"$LOGFILE" 2>&1 &
TRAIN_PID=$!

cleanup() {
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "stopping training (pid $TRAIN_PID)..."
        kill "$TRAIN_PID" 2>/dev/null
        wait "$TRAIN_PID" 2>/dev/null
    fi
    rm -f "$STOPFILE"
    echo "stopped. latest checkpoint: agents/${AGENT}/checkpoints/ppo_latest.pt"
}
trap cleanup INT TERM

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if [ -e "$STOPFILE" ]; then
        cleanup
        exit 0
    fi
    if [ "$END_TS" -gt 0 ] && [ "$(date +%s)" -ge "$END_TS" ]; then
        echo "duration elapsed"
        cleanup
        exit 0
    fi
    if [ -t 0 ]; then
        if read -r -t 1 line && [ "$line" = "stop" ]; then
            cleanup
            exit 0
        fi
    else
        sleep 1
    fi
done

echo "training process exited on its own -- check ${LOGFILE}"
