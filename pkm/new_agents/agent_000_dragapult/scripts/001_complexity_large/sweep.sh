#!/usr/bin/env bash
#
# 001_complexity_large / sweep.sh
# ------------------------------------------------------------------------------
# Optuna sweep at the LARGE architecture (--model large, fixed) that searches the
# PPO hyperparameters AND all 16 heuristic reward-term weights (--tune-rewards).
# Each trial is a short training; the objective is curve_auc (mean of the eval
# learning curve — best signal while win-rate-vs-random saturates near the ceiling).
#
# Run it inside the shared tmux session (see PLAN.md):
#     tmux new-session -d -s pkm-train
#     tmux send-keys -t pkm-train \
#       "cd <repo> && ./pkm/new_agents/agent_000_dragapult/scripts/001_complexity_large/sweep.sh" Enter
#
# The study is SQLite-backed + resumable at
#   pkm_data/.../experiments/001_complexity_large/sweeps/dragapult_large.db
# When it finishes it prints the best trial's params (lr/entropy/... and every
# rw_<term>); paste the winners into train.sh, then launch the full run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

exec uv run pkm new_agents 000_dragapult sweep \
    --experiment 001_complexity_large \
    --model large \
    --tune-rewards \
    --trials 40 \
    --updates 20 \
    --games 32 \
    --workers 16 \
    --eval-games 128 \
    --objective curve_auc \
    --study dragapult_large \
    --seed 0 \
    --output-dir pkm_data/new_agents/agent_000_dragapult \
    --engine local-nix
