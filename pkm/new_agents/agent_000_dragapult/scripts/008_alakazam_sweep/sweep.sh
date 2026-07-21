#!/usr/bin/env bash
#
# 008_alakazam_sweep/sweep.sh — Optuna hyperparameter sweep for the **alakazam**
# deck. Each trial runs a short training + eval-vs-random and reports the
# objective; the SQLite study is resumable. Finds good PPO hyperparameters
# (lr / entropy / clip / epochs / minibatch / gamma / lam) to feed a longer run.
#
# --deck alakazam is hard-wired. Shaping stays deck-agnostic (prize_potential)
# unless TUNE_REWARDS=1 (which would sweep the *Dragapult* reward stack — leave
# it off for alakazam).
#
# Configurable via env vars (see block). Launch detached with
#   TMUX_SESSION=alksweep scripts/008_alakazam_sweep/sweep.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ============================ CONFIGURABLES ==================================
EXP="${EXP:-008_alakazam_sweep}"
STUDY="${STUDY:-alakazam_ppo}"        # resumable Optuna study (sqlite)
OUTPUT="${OUTPUT:-}"
ENGINE="${ENGINE:-local-nix}"
TMUX_SESSION="${TMUX_SESSION:-}"
# tmux stdout mirror lives WITH the run's artifacts under pkm_data (not the source tree).
DATA_ROOT="${OUTPUT:-$REPO_ROOT/pkm_data/new_agents/agent_000_dragapult}"
LOG="${LOG:-$DATA_ROOT/experiments/$EXP/logs/stdout.log}"

DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-xl}"                  # sweep at the deployment net for transferable params
TRIALS="${TRIALS:-30}"               # Optuna trials
UPDATES="${UPDATES:-24}"             # PPO updates per trial (short; ranks early learning)
GAMES="${GAMES:-16}"                 # games per update
WORKERS="${WORKERS:-8}"              # rollout workers per trial
EVAL_GAMES="${EVAL_GAMES:-128}"
OBJECTIVE="${OBJECTIVE:-net_winrate}" # curve_auc | final_winrate | peak_winrate | net_winrate
SEED="${SEED:-0}"
RESET="${RESET:-0}"                  # 1 = delete an objective-mismatched study first
# =============================================================================

CMD=(uv run pkm new_agents 000_dragapult sweep
     --deck alakazam --model "$MODEL"
     --trials "$TRIALS" --updates "$UPDATES" --games "$GAMES" --workers "$WORKERS"
     --eval-games "$EVAL_GAMES" --objective "$OBJECTIVE" --study "$STUDY"
     --seed "$SEED" --device "$DEVICE" --experiment "$EXP" --engine "$ENGINE")
[[ -n "$OUTPUT" ]] && CMD+=(--output-dir "$OUTPUT")
[[ "$RESET" == "1" ]] && CMD+=(--reset)

echo "==============================================================================="
echo " 008_alakazam_sweep  ·  deck=alakazam  ·  Optuna study '$STUDY'"
printf ' %-12s %s\n' model "$MODEL" trials "$TRIALS" "upd/trial" "$UPDATES" \
                     objective "$OBJECTIVE" device "$DEVICE"
echo " Command: ${CMD[*]}"
echo "==============================================================================="
echo
cd "$REPO_ROOT"

if [[ -n "$TMUX_SESSION" ]]; then
    command -v tmux >/dev/null || { echo "ERROR: tmux not installed" >&2; exit 3; }
    tmux has-session -t "$TMUX_SESSION" 2>/dev/null && {
        echo "ERROR: tmux session '$TMUX_SESSION' exists (kill it or pick another)" >&2; exit 3; }
    mkdir -p "$(dirname "$LOG")"
    launch=$(printf '%q ' "${CMD[@]}")
    tmux new-session -d -s "$TMUX_SESSION" -c "$REPO_ROOT" \
        "export LD_LIBRARY_PATH=$(printf '%q' "$LD_LIBRARY_PATH"); \
         $launch 2>&1 | tee $(printf '%q' "$LOG"); \
         echo; echo '[sweep exited]'; exec bash"
    echo "launched in tmux '$TMUX_SESSION'  (attach: tmux attach -t $TMUX_SESSION)"
    echo "  log: tail -f $LOG"
    exit 0
fi
exec "${CMD[@]}"
