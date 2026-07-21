#!/usr/bin/env bash
#
# 001_complexity_large / train.sh
# ------------------------------------------------------------------------------
# Full-length training run for the LARGE network (3-layer pre-LN transformer trunk,
# d_state=384) with the deck-specific heuristic reward stack. This is the explicit
# form of the default example, adapted to the larger architecture.
#
# Run it inside the shared tmux session (see PLAN.md):
#     tmux new-session -d -s pkm-train
#     tmux send-keys -t pkm-train \
#       "cd <repo> && ./pkm/new_agents/agent_000_dragapult/scripts/001_complexity_large/train.sh" Enter
#
# Notes:
#   * --lr is deliberately CONSERVATIVE (3e-4). The Optuna winners (lr 1.48e-3) were
#     tuned for the tiny v1 net and are likely unstable on a deeper model — let
#     sweep.sh find a large-specific lr, then paste the winner here.
#   * --model large fixes the architecture; it is baked into the config hash and
#     every checkpoint, so this experiment is self-contained and reproducible.
#   * On resume the whole config (incl. --model + reward weights) is restored from
#     the checkpoint, so DO NOT re-pass these flags when resuming (use --resume).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

# NixOS: torch's CUDA build needs the driver's libcuda.so, exposed here (not in
# an FHS path). Harmless on non-NixOS / CPU boxes (the dir just won't exist).
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec uv run pkm new_agents 000_dragapult train \
    --experiment 001_complexity_large \
    --updates 512 \
    --games 64 \
    --workers 16 \
    --model large \
    --device cuda \
    --lr 0.0003 \
    --gamma 0.997 \
    --lam 0.95 \
    --clip-eps 0.2 \
    --entropy-coef 0.01 \
    --value-coef 0.5 \
    --epochs 4 \
    --minibatch-size 64 \
    --seed 0 \
    --shaping heuristic \
    --reward-weight shaping=0.2 \
    --reward-weight board_setup=0.2 \
    --reward-weight budew_setup=0.2 \
    --reward-weight dreepy_field=0.2 \
    --reward-weight energy_penalty=0.2 \
    --reward-weight budew_bonus=0.3 \
    --reward-weight wrong_type_penalty=0.2 \
    --reward-weight dragapult_bonus=0.3 \
    --reward-weight dreepy_spread=0.1 \
    --reward-weight xerosic=0.2 \
    --reward-weight budew_bench_setup=0.2 \
    --reward-weight dreepy_evolve=0.3 \
    --reward-weight dreepy_bench_charge=0.2 \
    --reward-weight dreepy_active_charge=0.3 \
    --reward-weight wasted_resources=0.2 \
    --reward-weight phantom_dive=0.5 \
    --eval-every 16 \
    --eval-games 128 \
    --ckpt-every 64 \
    --output-dir pkm_data/new_agents/agent_000_dragapult \
    --engine local-nix \
    --no-resume \
    --tb \
    --wandb-mode offline \
    --run-name 001-complexity-large \
    --force
