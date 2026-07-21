#!/usr/bin/env bash
#
# 003_aux_loss/train.sh — the NEW DEFAULT training recipe: large net + PPO
# self-play + heuristic reward stack + the `prize_margin` AUXILIARY LOSS.
#
# What's new vs 002_large_tuned: an auxiliary head predicting the final
# prize-count margin (a dense -6..+6 outcome) is trained off the shared trunk,
# weighted by --aux-weight prize_margin=0.25. It's a training-only signal — the
# head is stripped from the Kaggle bundle at pack time, so inference/parity are
# untouched. Rationale + the menu of other candidate aux tasks:
# docs/00_aux_loss.md.
#
# Turn the aux off (recover the 002 recipe) by dropping the AUX_FLAGS line.
# Add another aux task the same way once it's registered in aux_tasks.py, e.g.
#   --aux-weight opp_has_gust=0.2
#
# Large net is trained at the tuned low LR (1e-4) — the higher tuned LR that
# worked for the small net was unstable at this depth (see TRAINING.md §10).
#
# Usage (positional, all optional):
#     scripts/003_aux_loss/train.sh [mode] [exp] [updates] [games] [workers] [engine] [force]
#
#   mode     train | resume            (default: train)
#   exp      experiment name           (default: 003_aux_loss)
#   updates  optimizer updates         (default: 256)
#   games    self-play games / update  (default: 16)
#   workers  parallel rollout workers  (default: 8)
#   engine   local-nix | local | kaggle (default: local-nix)
#   force    pass --force to overwrite an existing checkpoint without prompting
#
# Examples:
#     scripts/003_aux_loss/train.sh                          # fresh 256-update run
#     scripts/003_aux_loss/train.sh train aux01 200 16       # into experiment "aux01"
#     scripts/003_aux_loss/train.sh resume aux01 100 16      # continue that run
#
set -euo pipefail

# --- resolve paths (script -> 003_aux_loss -> scripts -> agent dir -> repo) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

# --- arguments (with defaults) ------------------------------------------------
MODE="${1:-train}"
EXP="${2:-003_aux_loss}"
UPDATES="${3:-256}"
GAMES="${4:-16}"
WORKERS="${5:-8}"
ENGINE="${6:-local-nix}"
FORCE="${7:-}"

if [[ "$MODE" != "train" && "$MODE" != "resume" ]]; then
    echo "ERROR: mode must be 'train' or 'resume' (got '$MODE')" >&2
    exit 2
fi

# --- large-net architecture (train mode only) ---------------------------------
MODEL_FLAGS=(--model large)

# --- tuned PPO hyperparameters for the LARGE net (low LR; train mode only) -----
TUNED_FLAGS=(
    --lr 1e-4
    --entropy-coef 0.009663651584975216
    --clip-eps 0.14448934467726907
    --epochs 5
    --minibatch-size 32
    --gamma 0.9644139546520633
    --lam 0.917435752072244
)

# --- the auxiliary loss (train mode only) -------------------------------------
AUX_FLAGS=(--aux-weight prize_margin=0.25)

# --- the heuristic reward stack (train mode only) -----------------------------
# Curated starting weights (same set as scripts/train-heuristic.sh). Refine with
# scripts/sweep-heuristic.sh and paste the winners here.
REWARD_FLAGS=(
    --shaping heuristic
    --reward-weight shaping=0.2
    --reward-weight board_setup=0.2
    --reward-weight budew_setup=0.2
    --reward-weight dreepy_field=0.2
    --reward-weight energy_penalty=0.2
    --reward-weight budew_bonus=0.3
    --reward-weight wrong_type_penalty=0.2
    --reward-weight dragapult_bonus=0.3
    --reward-weight dreepy_spread=0.1
    --reward-weight xerosic=0.2
    --reward-weight budew_bench_setup=0.2
    --reward-weight dreepy_evolve=0.3
    --reward-weight dreepy_bench_charge=0.2
    --reward-weight dreepy_active_charge=0.3
    --reward-weight wasted_resources=0.2
    --reward-weight phantom_dive=0.5
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
echo " 003_aux_loss  (large net + tuned PPO + heuristic rewards + prize_margin aux)"
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
    printf ' %-12s %s\n' "model"   "large"
    printf ' %-12s %s\n' "tuned"   "lr 1e-4 · entropy 9.664e-3 · clip 0.1445 · epochs 5 · minibatch 32 · gamma 0.9644 · lam 0.9174"
    printf ' %-12s %s\n' "aux"     "prize_margin=0.25 (training-only head)"
    printf ' %-12s %s\n' "shaping" "heuristic (16 reward terms)"
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
