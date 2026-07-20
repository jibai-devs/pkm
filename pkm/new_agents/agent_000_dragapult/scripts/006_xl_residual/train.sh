#!/usr/bin/env bash
#
# 006_xl_residual/train.sh — the swept heuristic recipe (Optuna best trial 16) on
# the XL preset WITH the new base-attention residual (--base-residual): a
# uniform-residual trunk (base attn + all extra layers now residual). See
# docs/00_aux_loss.md siblings and encoder.py for the flag.
#
# Arch: --model xl (4L / d_state 512 / d_entity 256 / d_opt 256 / d_card 64) plus
# --base-residual (adds a LayerNorm around the base attention). Reward weights +
# aux + PPO hyperparameters are trial-16 swept values; lr forced to 1e-4 (the
# swept 4.68e-4 was tuned on the small net and is unstable at depth — the
# residual should help stability, so raising lr later is worth a try).
#
# Usage: scripts/006_xl_residual/train.sh [mode] [exp] [updates] [games] [workers] [engine] [force]
#   Defaults: train · 006_xl_residual · 256 · 16 · 8 · local-nix · <none>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

MODE="${1:-train}"
EXP="${2:-006_xl_residual}"
UPDATES="${3:-256}"
GAMES="${4:-16}"
WORKERS="${5:-8}"
ENGINE="${6:-local-nix}"
FORCE="${7:-}"

if [[ "$MODE" != "train" && "$MODE" != "resume" ]]; then
    echo "ERROR: mode must be 'train' or 'resume' (got '$MODE')" >&2
    exit 2
fi

MODEL_FLAGS=(--model xl --base-residual)

TUNED_FLAGS=(
    --lr 1e-4
    --entropy-coef 0.014037528236254575
    --clip-eps 0.12466547078861025
    --epochs 3
    --minibatch-size 64
    --gamma 0.9901486456977385
    --lam 0.929944571253055
)

AUX_FLAGS=(--aux-weight prize_margin=0.25)

REWARD_FLAGS=(
    --shaping heuristic
    --reward-weight shaping=0.22124198921787636
    --reward-weight board_setup=0.26523068983353526
    --reward-weight budew_setup=0.1470491453196494
    --reward-weight dreepy_field=0.7337736300176154
    --reward-weight energy_penalty=0.5804253728977349
    --reward-weight budew_bonus=0.026423680059480703
    --reward-weight wrong_type_penalty=0.49225355292212963
    --reward-weight dragapult_bonus=0.26557862079433003
    --reward-weight dreepy_spread=0.665043599126133
    --reward-weight xerosic=0.818163371297779
    --reward-weight budew_bench_setup=0.9973854463391766
    --reward-weight dreepy_evolve=0.7754273038519655
    --reward-weight dreepy_bench_charge=0.531930043580297
    --reward-weight dreepy_active_charge=0.7958770523065606
    --reward-weight wasted_resources=0.8677344148174203
    --reward-weight phantom_dive=0.5263123628818893
    --reward-weight drakloak_backup_ready=0.13432290967575222
    --reward-weight budew_redundant=0.3262913990792372
)

CMD=(uv run pkm new_agents 000_dragapult "$MODE"
     --experiment "$EXP" --updates "$UPDATES" --games "$GAMES" --workers "$WORKERS")
if [[ "$MODE" == "train" ]]; then
    CMD+=("${MODEL_FLAGS[@]}" "${TUNED_FLAGS[@]}" "${AUX_FLAGS[@]}" "${REWARD_FLAGS[@]}")
fi
CMD+=(--engine "$ENGINE")
if [[ "$MODE" == "train" && -n "$FORCE" ]]; then
    CMD+=("$FORCE")
fi

echo "==============================================================================="
echo " 006_xl_residual  (trial-16 swept recipe on XL + base-attention residual)"
echo "-------------------------------------------------------------------------------"
printf ' %-12s %s\n' "mode" "$MODE"; printf ' %-12s %s\n' "exp" "$EXP"
printf ' %-12s %s\n' "updates" "$UPDATES"; printf ' %-12s %s\n' "engine" "$ENGINE"
if [[ "$MODE" == "train" ]]; then
    printf ' %-12s %s\n' "model" "xl + base_residual (uniform-residual trunk)"
    printf ' %-12s %s\n' "lr" "1e-4"
    printf ' %-12s %s\n' "aux" "prize_margin=0.25 · shaping=heuristic (18 swept)"
fi
echo " Command: ${CMD[*]}"
echo "==============================================================================="
echo
cd "$REPO_ROOT"
exec "${CMD[@]}"
