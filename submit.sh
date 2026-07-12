#!/run/current-system/sw/bin/bash
# Create submission bundle for Kaggle
# Usage: ./submit.sh [agent_name] [policy_path]
#
# The agent name determines which deck and weights to bundle.
# Only 02_dragapult is currently supported.

set -e

AGENT="${1:-02_dragapult}"
POLICY="${2:-agents/${AGENT}/checkpoints/policy.npz}"
if [ "$AGENT" != "02_dragapult" ]; then
    echo "Only 02_dragapult is supported" >&2
    exit 1
fi
TS=$(date +%Y%m%d_%H%M%S)
OUT="submissions/submission_${AGENT}_${TS}.tar.gz"

echo "Creating submission bundle for agent: $AGENT"

# Create submission directory
mkdir -p submission
mkdir -p submissions

# Copy main.py
cp main.py submission/

# Generate flat deck.csv from the agent's deck for kaggle
python -c "
from pkm.data import Deck
Deck.from_csv('deck/${AGENT}.csv').to_csv('submission/deck.csv')
"

# Copy agent code
cp -r pkm submission/

# Bundle the profile's fresh export at the path used by Kaggle inference.
if [ ! -f "$POLICY" ]; then
    echo "Policy weights not found: $POLICY" >&2
    exit 1
fi
cp "$POLICY" submission/pkm/policy.npz

# Create tar.gz
tar -czvf "$OUT" -C submission .

# Cleanup
rm -rf submission

echo "Submission bundle created: $OUT"
echo "Upload with: just upload $OUT"
