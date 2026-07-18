#!/usr/bin/env bash
# Launch a human-vs-agent match in the terminal TUI.
# Usage: ./battle.sh [opponent] [deck]
#   opponent  agent to play against (default: singaporean_middleman)
#   deck      deck CSV both sides use (default: deck/03_pult_munki.csv)

set -e

OPPONENT="${1:-singaporean_middleman}"
DECK="${2:-deck/03_pult_munki.csv}"

echo "Starting battle: you vs ${OPPONENT} (deck: ${DECK})"
uv run pkm play --p0 human --p1 "$OPPONENT" --deck "$DECK"
