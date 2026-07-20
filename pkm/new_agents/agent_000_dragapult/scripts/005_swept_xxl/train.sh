#!/usr/bin/env bash
#
# 005_swept_xxl/train.sh — the swept heuristic recipe (Optuna best trial 16,
# curve_auc 96.7%) on an "XXL" net: swept 18-term reward weights + prize_margin
# aux + swept PPO hyperparameters, EXCEPT the learning rate.
#
# There is no "xxl" preset (max preset is xl: 4L / d_state 512 / d_entity 256).
# XXL here = xl pushed deeper+wider via overrides:
#     n_layers 6 · d_state 768 · d_entity 384 · d_opt 384 · d_global 256 · d_card 96
# ~11.9M params (11.3M deployed after the aux head is stripped at pack).
#
# ⚠  LEARNING RATE DEVIATES FROM THE SWEEP ON PURPOSE. The sweep ran on the SMALL
#    net and found lr 4.68e-4; that lr was already unstable on the large net
#    (TRAINING.md §10) and would be worse on a deeper/wider XXL. So we use the
#    proven-stable lr 1e-4 here. Everything else (entropy/clip/epochs/minibatch/
#    gamma/lam + the 18 reward weights) is the swept trial-16 value. If you want
#    to trust the swept lr instead, change --lr below and watch the first ~20
#    updates for divergence (val loss blow-up / entropy collapse).
#
# Usage (positional, all optional):
#     scripts/005_swept_xxl/train.sh [mode] [exp] [updates] [games] [workers] [engine] [force]
#
#   mode/exp/updates/games/workers/engine/force — as in the other 00N scripts.
#   Defaults: train · 005_swept_xxl · 1024 · 16 · 8 · local-nix · <none>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

MODE="${1:-train}"
EXP="${2:-005_swept_xxl}"
UPDATES="${3:-1024}"
GAMES="${4:-16}"
WORKERS="${5:-8}"
ENGINE="${6:-local-nix}"
FORCE="${7:-}"

if [[ "$MODE" != "train" && "$MODE" != "resume" ]]; then
    echo "ERROR: mode must be 'train' or 'resume' (got '$MODE')" >&2
    exit 2
fi

# --- XXL architecture (xl preset + deeper/wider overrides) --------------------
MODEL_FLAGS=(
    --model xl
    --n-layers 6
    --d-state 768
    --d-entity 384
    --d-opt 384
    --d-card 96
    --n-heads 8
)

# --- swept PPO hyperparameters (trial 16) EXCEPT lr (stability; see header) ----
TUNED_FLAGS=(
    --lr 1e-4
    --entropy-coef 0.014037528236254575
    --clip-eps 0.12466547078861025
    --epochs 3
    --minibatch-size 64
    --gamma 0.9901486456977385
    --lam 0.929944571253055
)

# --- auxiliary loss -----------------------------------------------------------
AUX_FLAGS=(--aux-weight prize_margin=0.25)

# --- swept 18-term heuristic reward weights (trial 16) ------------------------
REWARD_FLAGS=(
    --shaping heuristic
    --reward-weight shaping=0.22124198921787636
    --reward-weight board_setup=0.26523068983353526
    --reward-weight budew_setup=0.1470491453196494
    --reward-weight dreepy_field=0.7337736300176154
    --reward-weight energy_penalty=0.5804253728977349
    --reward-weight budew_bonus=0.026423680059480703
    --reward-weight wrong_type_penalty=0.49225355292212963
    --reward-weight dragapult_bonus=0.26557862079433003
    --reward-weight dreepy_spread=0.665043599126133
    --reward-weight xerosic=0.818163371297779
    --reward-weight budew_bench_setup=0.9973854463391766
    --reward-weight dreepy_evolve=0.7754273038519655
    --reward-weight dreepy_bench_charge=0.531930043580297
    --reward-weight dreepy_active_charge=0.7958770523065606
    --reward-weight wasted_resources=0.8677344148174203
    --reward-weight phantom_dive=0.5263123628818893
    --reward-weight drakloak_backup_ready=0.13432290967575222
    --reward-weight budew_redundant=0.3262913990792372
)

CMD=(uv run pkm new_agents 000_dragapult "$MODE"
     --experiment "$EXP" --updates "$UPDATES" --games "$GAMES" --workers "$WORKERS")
if [[ "$MODE" == "train" ]]; then
    CMD+=("${MODEL_FLAGS[@]}" "${TUNED_FLAGS[@]}" "${AUX_FLAGS[@]}" "${REWARD_FLAGS[@]}")
fi
CMD+=(--engine "$ENGINE")
if [[ "$MODE" == "train" && -n "$FORCE" ]]; then
    CMD+=("$FORCE")
fi

echo "==============================================================================="
echo " 005_swept_xxl  (trial-16 swept recipe on XXL net; lr forced to 1e-4)"
echo "-------------------------------------------------------------------------------"
printf ' %-12s %s\n' "mode" "$MODE"; printf ' %-12s %s\n' "exp" "$EXP"
printf ' %-12s %s\n' "updates" "$UPDATES"; printf ' %-12s %s\n' "games" "$GAMES"
printf ' %-12s %s\n' "workers" "$WORKERS"; printf ' %-12s %s\n' "engine" "$ENGINE"
if [[ "$MODE" == "train" ]]; then
    printf ' %-12s %s\n' "model" "xxl (xl + 6L/d_state768/d_entity384/d_opt384/d_card96) ~11.9M params"
    printf ' %-12s %s\n' "lr" "1e-4 (swept 4.68e-4 overridden for stability)"
    printf ' %-12s %s\n' "aux" "prize_margin=0.25"
    printf ' %-12s %s\n' "shaping" "heuristic (18 swept reward weights)"
fi
echo " Command: ${CMD[*]}"
echo "==============================================================================="
echo

cd "$REPO_ROOT"
exec "${CMD[@]}"
