#!/usr/bin/env bash
# Train agent_001_transformer end-to-end (self-play + MCTS), on the GPU.
#
# Usage (from anywhere):  bash scripts/train.sh [ITERS] [SIMS] [DECK]
#   ITERS  outer self-play/train iterations (default 20)
#   SIMS   MCTS simulations per decision   (default 10)
#   DECK   deck to play: sample | dragapult | pult_munki (default sample)
#
# Runs the whole thing on cuda and tees a timestamped log into ./logs/.
# Checkpoints land in ./out/ (gitignored). Follow-up: bash scripts/submit.sh
#
# Prefer running inside the shared `pkm-train` tmux session so the long run
# survives detach:
#   tmux new-window -t pkm-train -n train001 -c <repo-root>
#   tmux send-keys  -t pkm-train:train001 "bash pkm/new_agents/agent_001_transformer/scripts/train.sh" Enter
set -euo pipefail

# NixOS: expose the NVIDIA driver's libcuda.so so torch can see the GPU (else
# torch.cuda.is_available() is False and --device cuda silently uses the CPU).
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"
OUT="$AGENT_DIR/out"
LOGDIR="$AGENT_DIR/logs"
mkdir -p "$LOGDIR"

ITERS="${1:-20}"
SIMS="${2:-10}"
DECK="${3:-sample}"
TS="$(date +%Y%m%d_%H%M%S)"

cd "$REPO_ROOT"
echo "training agent_001_transformer: iters=$ITERS sims=$SIMS deck=$DECK out=$OUT" >&2
python -m pkm.new_agents.agent_001_transformer.train \
    --iters "$ITERS" \
    --eval-games 50 \
    --selfplay-games 100 \
    --sims "$SIMS" \
    --deck "$DECK" \
    --device cuda \
    --out "$OUT" 2>&1 | tee "$LOGDIR/train_${DECK}_$TS.log"
