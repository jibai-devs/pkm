#!/run/current-system/sw/bin/bash
# Create submission bundle for Kaggle
# Usage: ./submit.sh [agent_name]
#
# The agent name determines which deck gets bundled as submission/deck.csv;
# main.py falls back to it if no deck.csv is present. Run `pkm export --agent
# <agent> pkm/policy.npz` first so the weights being bundled (pkm/policy.npz,
# copied in via `cp -r pkm submission/` below) match this agent.

set -e

AGENT="${1:-02_dragapult}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="submissions/submission_${AGENT}_${TS}.tar.gz"

echo "Creating submission bundle for agent: $AGENT"

# Create submission directory
mkdir -p submission
mkdir -p submissions

# Copy main.py
cp main.py submission/

# Generate flat deck.csv from the agent's deck for kaggle
uv run python -c "
from pkm.data import Deck
Deck.from_csv('deck/${AGENT}.csv').to_csv('submission/deck.csv')
"

# Copy agent code
cp -r pkm submission/

# Create tar.gz
tar -czvf "$OUT" -C submission .

# Cleanup
rm -rf submission

echo "Submission bundle created: $OUT"
echo "Upload with: just upload $OUT"
