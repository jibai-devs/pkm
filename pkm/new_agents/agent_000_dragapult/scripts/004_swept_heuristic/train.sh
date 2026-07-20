#!/usr/bin/env bash
#
# 004_swept_heuristic/train.sh — PPO self-play using the WINNING parameters from
# the dragapult_heuristic Optuna sweep (best trial 16, curve_auc 96.7%): both the
# tuned PPO hyperparameters AND the tuned 18-term heuristic reward weights,
# plus the prize_margin auxiliary loss.
#
# ⚠  These values were tuned on the SMALL net (the sweep ran --model small), so
#    this script trains --model small and uses the swept lr (4.68e-4) as found.
#    That lr was UNSTABLE on the large net (see TRAINING.md §10) — if you want a
#    large-net variant, drop --lr to ~1e-4 and re-tune; do NOT reuse 4.68e-4 at
#    depth. small+aux is ~106K params; large+aux is ~1.53M (deployed, aux stripped).
#
# The aux loss (prize_margin=0.25) was NOT part of the sweep (the sweep trained
# with aux off), so it's an orthogonal add-on carried over from the 003 default.
# Drop the AUX_FLAGS line to reproduce exactly what the sweep optimized.
#
# Usage (positional, all optional):
#     scripts/004_swept_heuristic/train.sh [mode] [exp] [updates] [games] [workers] [engine] [force]
#
#   mode     train | resume            (default: train)
#   exp      experiment name           (default: 004_swept_heuristic)
#   updates  optimizer updates         (default: 1024)
#   games    self-play games / update  (default: 16)
#   workers  parallel rollout workers  (default: 8)
#   engine   local-nix | local | kaggle (default: local-nix)
#   force    pass --force to overwrite an existing checkpoint without prompting
#
# Examples:
#     scripts/004_swept_heuristic/train.sh                       # fresh 1024-update run
#     scripts/004_swept_heuristic/train.sh train sw01 512 16     # into experiment "sw01"
#     scripts/004_swept_heuristic/train.sh resume sw01 256 16    # continue that run
#
set -euo pipefail

# --- resolve paths (script -> 004_swept_heuristic -> scripts -> agent -> repo) -
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

# --- arguments (with defaults) ------------------------------------------------
MODE="${1:-train}"
EXP="${2:-004_swept_heuristic}"
UPDATES="${3:-1024}"
GAMES="${4:-16}"
WORKERS="${5:-8}"
ENGINE="${6:-local-nix}"
FORCE="${7:-}"

if [[ "$MODE" != "train" && "$MODE" != "resume" ]]; then
    echo "ERROR: mode must be 'train' or 'resume' (got '$MODE')" >&2
    exit 2
fi

# --- net size (as swept) ------------------------------------------------------
MODEL_FLAGS=(--model small)

# --- swept PPO hyperparameters (best trial 16, curve_auc 96.7%; train mode only)
TUNED_FLAGS=(
    --lr 0.0004676045518655474
    --entropy-coef 0.014037528236254575
    --clip-eps 0.12466547078861025
    --epochs 3
    --minibatch-size 64
    --gamma 0.9901486456977385
    --lam 0.929944571253055
)

# --- auxiliary loss (orthogonal add-on; not part of the sweep) ----------------
AUX_FLAGS=(--aux-weight prize_margin=0.25)

# --- swept 18-term heuristic reward weights (best trial 16; train mode only) ---
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

# --- assemble the command -----------------------------------------------------
CMD=(uv run pkm new_agents 000_dragapult "$MODE"
     --experiment "$EXP"
     --updates "$UPDATES"
     --games "$GAMES"
     --workers "$WORKERS")

if [[ "$MODE" == "train" ]]; then
    CMD+=("${MODEL_FLAGS[@]}")
    CMD+=("${TUNED_FLAGS[@]}")
    CMD+=("${AUX_FLAGS[@]}")
    CMD+=("${REWARD_FLAGS[@]}")
fi
CMD+=(--engine "$ENGINE")
if [[ "$MODE" == "train" && -n "$FORCE" ]]; then
    CMD+=("$FORCE")
fi

# --- print what we resolved (verbose) -----------------------------------------
echo "==============================================================================="
echo " 004_swept_heuristic  (Optuna best trial 16 — small net + swept PPO + rewards + aux)"
echo "-------------------------------------------------------------------------------"
printf ' %-12s %s\n' "mode"      "$MODE"
printf ' %-12s %s\n' "exp"       "$EXP"
printf ' %-12s %s\n' "updates"   "$UPDATES"
printf ' %-12s %s\n' "games"     "$GAMES"
printf ' %-12s %s\n' "workers"   "$WORKERS"
printf ' %-12s %s\n' "engine"    "$ENGINE"
printf ' %-12s %s\n' "force"     "${FORCE:-<none>}"
printf ' %-12s %s\n' "repo root" "$REPO_ROOT"
if [[ "$MODE" == "train" ]]; then
    printf ' %-12s %s\n' "model"   "small (as swept)"
    printf ' %-12s %s\n' "tuned"   "lr 4.676e-4 · entropy 1.404e-2 · clip 0.1247 · epochs 3 · minibatch 64 · gamma 0.9901 · lam 0.9299"
    printf ' %-12s %s\n' "aux"     "prize_margin=0.25 (orthogonal; not swept)"
    printf ' %-12s %s\n' "shaping" "heuristic (18 swept reward weights)"
else
    printf ' %-12s %s\n' "config"  "<restored from checkpoint — flags NOT re-passed>"
fi
echo "-------------------------------------------------------------------------------"
echo " Running from: $REPO_ROOT"
echo " Command:"
echo "     ${CMD[*]}"
echo "==============================================================================="
echo

# --- run it -------------------------------------------------------------------
cd "$REPO_ROOT"
exec "${CMD[@]}"
