#!/usr/bin/env bash
# Pack the latest agent_001_transformer checkpoint and submit it to Kaggle.
#
# Usage (from anywhere):  bash scripts/submit.sh ["submission message"]
#
# Packs ./out/latest.pth into a flattened .tar.gz (main.py + weights.pth + pkm/)
# under ./out/submissions/, then `kaggle competitions submit`. Requires a valid
# ~/.kaggle/kaggle.json. Remember to append a row to ./submission_log.md.
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"
OUT="$AGENT_DIR/out"
COMP="pokemon-tcg-ai-battle"
MSG="${1:-agent_001_transformer submission}"

if [[ ! -f "$OUT/latest.pth" ]]; then
    echo "no checkpoint at $OUT/latest.pth — run scripts/train.sh first" >&2
    exit 1
fi

cd "$REPO_ROOT"
python -m pkm.new_agents.agent_001_transformer.pack \
    --checkpoint "$OUT/latest.pth" \
    --out "$OUT/submissions"

BUNDLE="$(ls -t "$OUT"/submissions/*.tar.gz | head -1)"
echo "submitting $BUNDLE to $COMP" >&2
kaggle competitions submit -c "$COMP" -f "$BUNDLE" -m "$MSG"
echo "submitted. check status:  kaggle competitions submissions -c $COMP" >&2
