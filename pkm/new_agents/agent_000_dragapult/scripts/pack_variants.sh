#!/usr/bin/env bash
#
# pack_variants.sh
# ------------------------------------------------------------------------------
# Pack the SAME checkpoint twice — once as a plain policy bundle (fast, no
# search) and once with inference-time MCTS (stronger, slower) — so both can be
# submitted and compared on the Kaggle leaderboard. Nothing is submitted here;
# each pack prints the exact `submit` command to run when you've picked one.
#
# Whether a bundle uses MCTS is baked into its weights.pt at pack time
# (InferenceConfig), so the two bundles differ only in that config — same net.
#
# Usage (all optional):
#   scripts/pack_variants.sh [experiment] [K] [checkpoint]
#     experiment  default: 002_large_tuned
#     K           MCTS simulation budget for the search bundle (default: 32)
#     checkpoint  default: newest ckpt_<N>.pt in the experiment
#
# Examples:
#   scripts/pack_variants.sh                       # 002_large_tuned, K=32, latest ckpt
#   scripts/pack_variants.sh 002_large_tuned 64    # deeper search
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

EXP="${1:-002_large_tuned}"
K="${2:-32}"
CKPT_DIR="pkm_data/new_agents/agent_000_dragapult/experiments/$EXP/checkpoints"
CKPT="${3:-$(ls -1v "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | tail -1 || true)}"

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "ERROR: no checkpoint to pack (looked in $CKPT_DIR)" >&2
    exit 1
fi

CLI="uv run --no-sync pkm new_agents 000_dragapult"

echo "==============================================================================="
echo " pack variants  ($EXP)"
echo "-------------------------------------------------------------------------------"
printf ' %-11s %s\n' "experiment" "$EXP"
printf ' %-11s %s\n' "checkpoint" "$CKPT"
printf ' %-11s %s\n' "mcts K"     "$K"
echo "==============================================================================="

echo
echo ">> [1/2] policy bundle (no search)"
$CLI pack --experiment "$EXP" --checkpoint "$CKPT"

echo
echo ">> [2/2] MCTS bundle (K=$K)"
$CLI pack --experiment "$EXP" --checkpoint "$CKPT" --inference mcts -K "$K"

echo
echo ">> both bundles are in $CKPT_DIR/../submissions/ (newest = MCTS)."
echo ">> submit the MCTS one:   $CLI submit --experiment $EXP --message \"$EXP MCTS K=$K\""
echo ">> or submit the policy one: pass its explicit path via --bundle."
