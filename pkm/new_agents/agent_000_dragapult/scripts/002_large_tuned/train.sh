#!/usr/bin/env bash
#
# 002_large_tuned / train.sh
# ------------------------------------------------------------------------------
# LARGE net (3-layer trunk) on GPU, with hyperparameters adjusted to fix the
# instability seen in 001_complexity_large. That run showed kl ~0.05 and
# grad_norm ~3.5 (vs the 0.5 clip -> clipped ~7x every step): the small-net lr
# was too aggressive for the deeper net. Changes vs 001:
#
#   lr           3e-4 -> 1e-4   # main fix: smaller steps for the deeper net (lower kl/gnorm)
#   entropy_coef 0.01 -> 0.02   # more exploration + regularization
#   epochs       4    -> 3      # less policy drift per update (further lowers kl)
#   games/update 64   -> 96     # more samples/update -> lower-variance gradients
#   minibatch    64   -> 128    # steadier gradient estimate for the big net
#   dropout      0.0  -> 0.1    # regularize the extra transformer layers (curb self-play overfit)
#
# Everything else (large arch, gamma/lam/clip/value_coef, 512 updates) matches
# 001 so this is a clean A/B on the hyperparameters.
#
# Run inside the shared tmux session (new window):
#   tmux new-window -t pkm-train -n large-tuned
#   tmux send-keys -t pkm-train:large-tuned \
#     "cd <repo> && ./pkm/new_agents/agent_000_dragapult/scripts/002_large_tuned/train.sh" Enter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

# NixOS: torch's CUDA build needs the driver's libcuda.so, exposed here (not in
# an FHS path). Harmless on non-NixOS / CPU boxes (the dir just won't exist).
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec uv run pkm new_agents 000_dragapult train \
    --experiment 002_large_tuned \
    --updates 512 \
    --games 96 \
    --workers 16 \
    --model large \
    --dropout 0.1 \
    --device cuda \
    --lr 0.0001 \
    --gamma 0.997 \
    --lam 0.95 \
    --clip-eps 0.2 \
    --entropy-coef 0.02 \
    --value-coef 0.5 \
    --epochs 3 \
    --minibatch-size 128 \
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
    --run-name 002-large-tuned \
    --force
