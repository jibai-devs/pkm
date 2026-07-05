#!/run/current-system/sw/bin/bash
# Create submission bundle for Kaggle
# Usage: ./submit.sh

set -e

echo "Creating submission bundle..."

# Create submission directory
mkdir -p submission

# Copy main.py
cp main.py submission/

# Copy deck.csv
cp deck.csv submission/

# Copy agent code
cp -r pkm submission/

# Create tar.gz
tar -czvf submission.tar.gz -C submission .

# Cleanup
rm -rf submission

echo "Submission bundle created: submission.tar.gz"
echo "Upload this file to Kaggle."
