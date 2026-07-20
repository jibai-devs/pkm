#!/usr/bin/env bash
#
# 005_swept_xxl/sweep.sh — Optuna sweep of PPO hyperparameters + the 18-term
# heuristic reward weights (--tune-rewards) trained ON THE XXL NET, so the tuned
# values are validated at the size we actually deploy — unlike the original
# dragapult_heuristic study, which swept on the small net (its lr didn't transfer
# to large/xxl). Uses the new per-dim override flags on the `sweep` command.
#
# XXL arch (matches scripts/005_swept_xxl/train.sh): xl preset + overrides
#     n_layers 6 · d_state 768 · d_entity 384 · d_opt 384 · d_card 96 · n_heads 8
# ~11.9M params per trial.
#
# ⚠  COST. Each trial trains an ~11.9M-param net; that's ~100x the small-net
#    sweep's per-trial cost. So the defaults here are DELIBERATELY SMALL:
#    12 trials x 8 updates/trial (vs 30 x 15 for small). MedianPruner kills weak
#    trials early. Even so this is a multi-hour run and it shares CPU with any
#    concurrent training. Raise trials/updates only if you have the cores/time.
#    Note: the swept lr range still includes high values — a diverging xxl trial
#    just scores poorly and gets pruned, which is fine for a search.
#
# Study is SQLite-backed + resumable at <output>/sweeps/<study>.db. When it
# finishes it prints the best trial's params; paste the winners into
# scripts/005_swept_xxl/train.sh and record them in docs/01_sweep_results.md.
#
# Usage (positional, all optional):
#     scripts/005_swept_xxl/sweep.sh [exp] [trials] [updates] [games] [workers] [study] [objective] [engine]
#   Defaults: 005_swept_xxl · 12 · 8 · 32 · 8 · dragapult_xxl · curve_auc · local-nix
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

EXP="${1:-005_swept_xxl}"
TRIALS="${2:-12}"
UPDATES="${3:-8}"
GAMES="${4:-32}"
WORKERS="${5:-8}"
STUDY="${6:-dragapult_xxl}"
OBJECTIVE="${7:-curve_auc}"
ENGINE="${8:-local-nix}"

# XXL overrides (same as the train script).
XXL_FLAGS=(
    --model xl
    --n-layers 6
    --d-state 768
    --d-entity 384
    --d-opt 384
    --d-card 96
    --n-heads 8
)

CMD=(uv run pkm new_agents 000_dragapult sweep
     --experiment "$EXP"
     --trials "$TRIALS"
     --updates "$UPDATES"
     --games "$GAMES"
     --workers "$WORKERS"
     --study "$STUDY"
     --objective "$OBJECTIVE"
     --tune-rewards
     "${XXL_FLAGS[@]}"
     --engine "$ENGINE")

echo "==============================================================================="
echo " 005_swept_xxl/sweep  (Optuna: PPO + reward weights, trained on the XXL net)"
echo "-------------------------------------------------------------------------------"
printf ' %-12s %s\n' "exp" "$EXP";        printf ' %-12s %s\n' "trials" "$TRIALS"
printf ' %-12s %s\n' "updates" "$UPDATES"; printf ' %-12s %s\n' "games" "$GAMES"
printf ' %-12s %s\n' "workers" "$WORKERS"; printf ' %-12s %s\n' "study" "$STUDY"
printf ' %-12s %s\n' "objective" "$OBJECTIVE"; printf ' %-12s %s\n' "engine" "$ENGINE"
printf ' %-12s %s\n' "arch" "xxl (~11.9M params/trial) — expensive"
echo " Command: ${CMD[*]}"
echo "==============================================================================="
echo

cd "$REPO_ROOT"
exec "${CMD[@]}"
