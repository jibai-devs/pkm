#!/usr/bin/env bash
#
# 002_large_tuned / pack_submit.sh
# ------------------------------------------------------------------------------
# Pack the latest checkpoint of the fine-tuned large run and submit it to Kaggle,
# then watch for the score. Picks the newest IMMUTABLE numbered snapshot
# (ckpt_<N>.pt), never latest.pt — that file is rewritten every update while
# training is live, so packing it risks a torn read.
#
# The packed bundle embeds the model architecture (weights.pt carries model_config),
# so the large net rebuilds correctly at inference; Kaggle inference runs on CPU.
#
# Usage (both optional):
#   scripts/002_large_tuned/pack_submit.sh [experiment] [message]
#     experiment  default: 002_large_tuned
#     message     default: "<experiment> <ckpt> (fine-tuned large net)"
#
# Examples:
#   scripts/002_large_tuned/pack_submit.sh                      # latest ckpt of 002_large_tuned
#   scripts/002_large_tuned/pack_submit.sh 002_large_tuned "v3 tuned"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

EXP="${1:-002_large_tuned}"
CKPT_DIR="pkm_data/new_agents/agent_000_dragapult/experiments/$EXP/checkpoints"

# Newest immutable numbered snapshot (version-sorted so ckpt_512 > ckpt_64).
LATEST="$(ls -1v "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | tail -1 || true)"
if [[ -z "$LATEST" ]]; then
    echo "ERROR: no numbered checkpoint (ckpt_*.pt) in $CKPT_DIR" >&2
    echo "       (has $EXP reached its first --ckpt-every yet?)" >&2
    exit 1
fi
MSG="${2:-$EXP $(basename "$LATEST") (fine-tuned large net)}"

CLI="uv run --no-sync pkm new_agents 000_dragapult"

echo "==============================================================================="
echo " pack + submit  ($EXP)"
echo "-------------------------------------------------------------------------------"
printf ' %-11s %s\n' "experiment" "$EXP"
printf ' %-11s %s\n' "checkpoint" "$LATEST"
printf ' %-11s %s\n' "message"    "$MSG"
echo "==============================================================================="
echo

$CLI pack   --experiment "$EXP" --checkpoint "$LATEST"
$CLI submit --experiment "$EXP" --message "$MSG"
echo
echo ">> watching for the score (Ctrl-C to stop watching; the submission still scores) ..."
exec $CLI status --watch --timeout 600
