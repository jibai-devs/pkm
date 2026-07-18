#!/usr/bin/env bash
#
# train-opt-00.sh — PPO self-play training with the winning hyperparameters from
# the Optuna sweep (study=dragapult_ppo, trial 13, curve_auc 91.8%). Standalone
# equivalent of the `train-opt-00` just recipe, but VERBOSE: it prints every
# resolved parameter and the exact command it is about to run before running it.
#
# Tuned flags (injected only in `train` mode; on `resume` the config — including
# minibatch/epochs/lr — is restored from the checkpoint, so they must NOT be
# re-passed):
#     lr 1.476e-3 · entropy 9.664e-3 · clip 0.1445 · epochs 5 ·
#     minibatch 32 · gamma 0.9644 · lam 0.9174
#
# Usage (positional, matching the just recipe order — all optional):
#     scripts/train-opt-00.sh [mode] [exp] [updates] [games] [workers] [engine] [force]
#
#   mode     train | resume            (default: train)
#   exp      experiment name           (default: 000_default)
#   updates  optimizer updates         (default: 256)
#   games    self-play games / update  (default: 16)
#   workers  parallel rollout workers  (default: 8)
#   engine   local-nix | local | kaggle (default: local-nix)
#   force    pass --force to overwrite an existing checkpoint without prompting
#            (default: empty; only meaningful in train mode)
#
# Examples:
#     scripts/train-opt-00.sh                              # fresh 256-update run into 000_default
#     scripts/train-opt-00.sh train opt00 200 16           # fresh run into experiment "opt00"
#     scripts/train-opt-00.sh resume opt00 100 16          # continue that run for 100 more updates
#     scripts/train-opt-00.sh train opt00 256 16 8 local-nix --force
#
set -euo pipefail

# --- resolve paths (script -> agent dir -> repo root) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

# --- arguments (with defaults matching the just recipe) -----------------------
MODE="${1:-train}"
EXP="${2:-000_default}"
UPDATES="${3:-256}"
GAMES="${4:-16}"
WORKERS="${5:-8}"
ENGINE="${6:-local-nix}"
FORCE="${7:-}"

if [[ "$MODE" != "train" && "$MODE" != "resume" ]]; then
    echo "ERROR: mode must be 'train' or 'resume' (got '$MODE')" >&2
    exit 2
fi

# --- the tuned hyperparameters (train mode only) ------------------------------
TUNED_FLAGS=(
    --lr 0.001476395812742625
    --entropy-coef 0.009663651584975216
    --clip-eps 0.14448934467726907
    --epochs 5
    --minibatch-size 32
    --gamma 0.9644139546520633
    --lam 0.917435752072244
)

# --- assemble the command -----------------------------------------------------
# Base CLI: `uv run pkm new_agents 000_dragapult <mode> ...`
CMD=(uv run pkm new_agents 000_dragapult "$MODE"
     --experiment "$EXP"
     --updates "$UPDATES"
     --games "$GAMES"
     --workers "$WORKERS")

if [[ "$MODE" == "train" ]]; then
    CMD+=("${TUNED_FLAGS[@]}")
fi
CMD+=(--engine "$ENGINE")
if [[ "$MODE" == "train" && -n "$FORCE" ]]; then
    CMD+=("$FORCE")
fi

# --- print what we resolved (verbose) -----------------------------------------
echo "==============================================================================="
echo " train-opt-00  (Optuna-tuned PPO self-play)"
echo "-------------------------------------------------------------------------------"
printf ' %-12s %s\n' "mode"     "$MODE"
printf ' %-12s %s\n' "exp"      "$EXP"
printf ' %-12s %s\n' "updates"  "$UPDATES"
printf ' %-12s %s\n' "games"    "$GAMES"
printf ' %-12s %s\n' "workers"  "$WORKERS"
printf ' %-12s %s\n' "engine"   "$ENGINE"
printf ' %-12s %s\n' "force"    "${FORCE:-<none>}"
printf ' %-12s %s\n' "repo root" "$REPO_ROOT"
if [[ "$MODE" == "train" ]]; then
    printf ' %-12s %s\n' "tuned"  "lr 1.476e-3 · entropy 9.664e-3 · clip 0.1445 · epochs 5 · minibatch 32 · gamma 0.9644 · lam 0.9174"
else
    printf ' %-12s %s\n' "tuned"  "<restored from checkpoint — flags NOT re-passed>"
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
