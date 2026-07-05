#!/run/current-system/sw/bin/bash
# Create submission bundle for Kaggle
# Usage: ./submit.sh [agent_name]
#
# The agent name determines which deck and weights to bundle.
# Defaults to 00_basic.

set -e

AGENT="${1:-00_basic}"

echo "Creating submission bundle for agent: $AGENT"

# Create submission directory
mkdir -p submission

# Copy main.py
cp main.py submission/

# Generate flat deck.csv from the agent's deck for kaggle
python -c "
from pkm.data import Deck
Deck.from_csv('deck/${AGENT}.csv').to_csv('submission/deck.csv')
"

# Copy agent code
cp -r pkm submission/

# Create tar.gz
tar -czvf submission.tar.gz -C submission .

# Cleanup
rm -rf submission

echo "Submission bundle created: submission.tar.gz"
echo "Upload this file to Kaggle."
