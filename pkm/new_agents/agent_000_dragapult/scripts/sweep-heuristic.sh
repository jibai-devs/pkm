#!/usr/bin/env bash
#
# sweep-heuristic.sh — Optuna sweep that tunes the PPO hyperparameters AND the full
# deck-specific heuristic reward stack together (--tune-rewards: sets shaping to
# 'heuristic' and samples every term in reward_terms.ALL_TERMS in [0, 1] per trial).
# Standalone equivalent of the `sweep-heuristic` just recipe, but VERBOSE: it prints
# every resolved parameter and the exact command before running it.
#
# The study is SQLite-backed at <output>/sweeps/<study>.db (resumable; view with
# `optuna-dashboard sqlite:///.../<study>.db`). Trials are pruned early via reported
# intermediate evals. When it finishes it prints the best trial's params — including
# the winning rw_<term> weights, which you then paste into scripts/train-heuristic.sh.
#
# Usage (positional, all optional):
#     scripts/sweep-heuristic.sh [exp] [trials] [updates] [games] [workers] [study] [objective] [engine]
#
#   exp        experiment name             (default: 000_heuristic_sweep)
#   trials     Optuna trials               (default: 30)
#   updates    PPO updates per trial       (default: 15, keep short)
#   games      self-play games / update    (default: 32)
#   workers    parallel rollout workers    (default: 8)
#   study      Optuna study name (sqlite)  (default: dragapult_heuristic)
#   objective  curve_auc | final_winrate | peak_winrate | net_winrate
#                                          (default: curve_auc)
#   engine     local-nix | local | kaggle  (default: local-nix)
#
# Examples:
#     scripts/sweep-heuristic.sh                                  # 30 trials into dragapult_heuristic
#     scripts/sweep-heuristic.sh heur_sweep1 50 20 32            # 50 trials, 20 updates each
#     scripts/sweep-heuristic.sh heur_sweep1 50 20 32 8 s1 net_winrate
#
set -euo pipefail

# --- resolve paths (script -> agent dir -> repo root) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

# --- arguments (with defaults) ------------------------------------------------
EXP="${1:-000_heuristic_sweep}"
TRIALS="${2:-30}"
UPDATES="${3:-15}"
GAMES="${4:-32}"
WORKERS="${5:-8}"
STUDY="${6:-dragapult_heuristic}"
OBJECTIVE="${7:-curve_auc}"
ENGINE="${8:-local-nix}"

# --- assemble the command -----------------------------------------------------
CMD=(uv run pkm new_agents 000_dragapult sweep
     --experiment "$EXP"
     --trials "$TRIALS"
     --updates "$UPDATES"
     --games "$GAMES"
     --workers "$WORKERS"
     --study "$STUDY"
     --objective "$OBJECTIVE"
     --tune-rewards
     --engine "$ENGINE")

# --- print what we resolved (verbose) -----------------------------------------
echo "==============================================================================="
echo " sweep-heuristic  (Optuna: PPO hyperparameters + heuristic reward weights)"
echo "-------------------------------------------------------------------------------"
printf ' %-12s %s\n' "exp"       "$EXP"
printf ' %-12s %s\n' "trials"    "$TRIALS"
printf ' %-12s %s\n' "updates"   "$UPDATES"
printf ' %-12s %s\n' "games"     "$GAMES"
printf ' %-12s %s\n' "workers"   "$WORKERS"
printf ' %-12s %s\n' "study"     "$STUDY"
printf ' %-12s %s\n' "objective" "$OBJECTIVE"
printf ' %-12s %s\n' "engine"    "$ENGINE"
printf ' %-12s %s\n' "rewards"   "heuristic (all 16 terms sampled per trial)"
printf ' %-12s %s\n' "repo root" "$REPO_ROOT"
echo "-------------------------------------------------------------------------------"
echo " Running from: $REPO_ROOT"
echo " Command:"
echo "     ${CMD[*]}"
echo "==============================================================================="
echo

# --- run it -------------------------------------------------------------------
cd "$REPO_ROOT"
exec "${CMD[@]}"
