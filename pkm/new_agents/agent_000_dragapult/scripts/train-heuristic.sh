#!/usr/bin/env bash
#
# train-heuristic.sh — PPO self-play training with the Optuna-tuned hyperparameters
# (study=dragapult_ppo, trial 13) AND the full deck-specific heuristic reward stack
# (--shaping heuristic + a curated set of --reward-weight starting values). VERBOSE:
# prints every resolved parameter and the exact command before running it.
#
# The reward weights below are a hand-picked *starting point*, not tuned values —
# refine them with scripts/sweep-heuristic.sh, then paste the winners here. Each is
# non-negative: the heuristic functions already carry the sign (penalties return
# negative values), so the weight only scales magnitude. See pkm/rl/encoder.py for
# what each term rewards/penalizes and pkm/rl/reward_terms.py for the term registry.
#
# Tuned + heuristic flags are injected only in `train` mode; on `resume` the whole
# config (hyperparameters AND reward weights) is restored from the checkpoint, so
# they must NOT be re-passed.
#
# Usage (positional, all optional):
#     scripts/train-heuristic.sh [mode] [exp] [updates] [games] [workers] [engine] [force]
#
#   mode     train | resume            (default: train)
#   exp      experiment name           (default: 000_heuristic)
#   updates  optimizer updates         (default: 256)
#   games    self-play games / update  (default: 16)
#   workers  parallel rollout workers  (default: 8)
#   engine   local-nix | local | kaggle (default: local-nix)
#   force    pass --force to overwrite an existing checkpoint without prompting
#            (default: empty; only meaningful in train mode)
#
# Examples:
#     scripts/train-heuristic.sh                            # fresh 256-update run into 000_heuristic
#     scripts/train-heuristic.sh train heur01 200 16        # fresh run into experiment "heur01"
#     scripts/train-heuristic.sh resume heur01 100 16       # continue that run for 100 more updates
#     scripts/train-heuristic.sh train heur01 256 16 8 local-nix --force
#
set -euo pipefail

# --- resolve paths (script -> agent dir -> repo root) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

# --- arguments (with defaults) ------------------------------------------------
MODE="${1:-train}"
EXP="${2:-000_heuristic}"
UPDATES="${3:-256}"
GAMES="${4:-16}"
WORKERS="${5:-8}"
ENGINE="${6:-local-nix}"
FORCE="${7:-}"

if [[ "$MODE" != "train" && "$MODE" != "resume" ]]; then
    echo "ERROR: mode must be 'train' or 'resume' (got '$MODE')" >&2
    exit 2
fi

# --- the tuned PPO hyperparameters (train mode only) --------------------------
TUNED_FLAGS=(
    --lr 0.001476395812742625
    --entropy-coef 0.009663651584975216
    --clip-eps 0.14448934467726907
    --epochs 5
    --minibatch-size 32
    --gamma 0.9644139546520633
    --lam 0.917435752072244
)

# --- the heuristic reward stack (train mode only) -----------------------------
# Curated starting weights (name=value). Terms omitted here keep their registry
# default (0.0 for deck-specific terms). Edit freely / replace with sweep winners.
REWARD_FLAGS=(
    --shaping heuristic
    --reward-weight shaping=0.2               # prize-differential potential (known-good)
    --reward-weight board_setup=0.2           # charged Drakloak backup ready
    --reward-weight budew_setup=0.2           # Budew active early (going 2nd)
    --reward-weight dreepy_field=0.2          # grow the Dreepy line toward 3
    --reward-weight energy_penalty=0.2        # don't over-attach to a full attacker
    --reward-weight budew_bonus=0.3           # take the free turn-2 Itchy Pollen
    --reward-weight wrong_type_penalty=0.2    # avoid same-type 2-energy on the line
    --reward-weight dragapult_bonus=0.3       # actually attack once Dragapult set up
    --reward-weight dreepy_spread=0.1         # spread energy across Dreepy, don't stack
    --reward-weight xerosic=0.2               # Xerosic's Machinations when it swings big
    --reward-weight budew_bench_setup=0.2     # develop bench during the free Budew turn
    --reward-weight dreepy_evolve=0.3         # evolve Dreepy->Drakloak (esp. charged)
    --reward-weight dreepy_bench_charge=0.2   # clean Fire/Psychic progress on the bench
    --reward-weight dreepy_active_charge=0.3  # complete the Fire+Psychic combo on active
    --reward-weight wasted_resources=0.2      # don't attack with cards left to play
    --reward-weight phantom_dive=0.5          # close the turn with the real finisher
)

# --- assemble the command -----------------------------------------------------
CMD=(uv run pkm new_agents 000_dragapult "$MODE"
     --experiment "$EXP"
     --updates "$UPDATES"
     --games "$GAMES"
     --workers "$WORKERS")

if [[ "$MODE" == "train" ]]; then
    CMD+=("${TUNED_FLAGS[@]}")
    CMD+=("${REWARD_FLAGS[@]}")
fi
CMD+=(--engine "$ENGINE")
if [[ "$MODE" == "train" && -n "$FORCE" ]]; then
    CMD+=("$FORCE")
fi

# --- print what we resolved (verbose) -----------------------------------------
echo "==============================================================================="
echo " train-heuristic  (Optuna-tuned PPO + heuristic reward stack)"
echo "-------------------------------------------------------------------------------"
printf ' %-12s %s\n' "mode"      "$MODE"
printf ' %-12s %s\n' "exp"       "$EXP"
printf ' %-12s %s\n' "updates"   "$UPDATES"
printf ' %-12s %s\n' "games"     "$GAMES"
printf ' %-12s %s\n' "workers"   "$WORKERS"
printf ' %-12s %s\n' "engine"    "$ENGINE"
printf ' %-12s %s\n' "force"     "${FORCE:-<none>}"
printf ' %-12s %s\n' "repo root" "$REPO_ROOT"
if [[ "$MODE" == "train" ]]; then
    printf ' %-12s %s\n' "tuned"   "lr 1.476e-3 · entropy 9.664e-3 · clip 0.1445 · epochs 5 · minibatch 32 · gamma 0.9644 · lam 0.9174"
    printf ' %-12s %s\n' "shaping" "heuristic (16 reward terms; see REWARD_FLAGS)"
else
    printf ' %-12s %s\n' "config"  "<restored from checkpoint — flags NOT re-passed>"
fi
echo "-------------------------------------------------------------------------------"
echo " Running from: $REPO_ROOT"
echo " Command:"
echo "     ${CMD[*]}"
echo "==============================================================================="
echo

# --- run it -------------------------------------------------------------------
cd "$REPO_ROOT"
exec "${CMD[@]}"
