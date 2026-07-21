#!/usr/bin/env bash
#
# 009_alakazam_xl/train.sh — long XL run for the **alakazam** deck with a cosine
# LR schedule.
#
# Hard-wired for correctness (not overridable):
#   * --deck alakazam
#   * --shaping prize_potential   (deck-agnostic; NO dragapult reward terms)
#
# Defaults: XL net + base_residual, 8192 updates, cosine LR 1e-4 -> 1e-5. Every
# knob below is an env-var override, e.g.  UPDATES=4096 LR=5e-5 scripts/.../train.sh
# Launch detached with  TMUX_SESSION=alk009 scripts/009_alakazam_xl/train.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

# NixOS: expose the NVIDIA driver libs so torch.cuda sees the GPU.
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ============================ CONFIGURABLES ==================================
MODE="${MODE:-train}"                 # train | resume
EXP="${EXP:-009_alakazam_xl}"         # experiment name
OUTPUT="${OUTPUT:-}"                   # artifact root (empty = repo default)
FORCE="${FORCE:-0}"                   # 1 = overwrite existing experiment ckpt
ENGINE="${ENGINE:-local-nix}"
TMUX_SESSION="${TMUX_SESSION:-}"      # non-empty = launch detached in that tmux session
# tmux stdout mirror lives WITH the run's other artifacts under pkm_data (not the
# source tree): <output>/experiments/<EXP>/logs/stdout.log.
DATA_ROOT="${OUTPUT:-$REPO_ROOT/pkm_data/new_agents/agent_000_dragapult}"
LOG="${LOG:-$DATA_ROOT/experiments/$EXP/logs/stdout.log}"

DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"
UPDATES="${UPDATES:-8192}"
GAMES="${GAMES:-16}"

MODEL="${MODEL:-xl}"
BASE_RESIDUAL="${BASE_RESIDUAL:-1}"

# LR: cosine anneal from LR down to LR_MIN over the run.
LR="${LR:-1e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"  # cosine | constant
LR_MIN="${LR_MIN:-1e-5}"
GAMMA="${GAMMA:-0.997}"
LAM="${LAM:-0.95}"
CLIP_EPS="${CLIP_EPS:-0.2}"
ENTROPY="${ENTROPY:-0.01}"
VALUE_COEF="${VALUE_COEF:-0.5}"
EPOCHS="${EPOCHS:-4}"
MINIBATCH="${MINIBATCH:-64}"
SEED="${SEED:-0}"

AUX_PRIZE_MARGIN="${AUX_PRIZE_MARGIN:-0.25}"

EVAL_EVERY="${EVAL_EVERY:-32}"        # eval vs random every N updates (0=never)
EVAL_GAMES="${EVAL_GAMES:-128}"
CKPT_EVERY="${CKPT_EVERY:-128}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
TB="${TB:-1}"
# =============================================================================

[[ "$MODE" == "train" || "$MODE" == "resume" ]] || { echo "MODE must be train|resume" >&2; exit 2; }

CMD=(uv run pkm new_agents 000_dragapult "$MODE"
     --experiment "$EXP" --updates "$UPDATES" --games "$GAMES" --workers "$WORKERS")
[[ -n "$OUTPUT" ]] && CMD+=(--output-dir "$OUTPUT")

if [[ "$MODE" == "train" ]]; then
    CMD+=(--deck alakazam --shaping prize_potential)
    CMD+=(--model "$MODEL")
    [[ "$BASE_RESIDUAL" == "1" ]] && CMD+=(--base-residual)
    CMD+=(--lr "$LR" --lr-schedule "$LR_SCHEDULE" --lr-min "$LR_MIN"
          --gamma "$GAMMA" --lam "$LAM" --clip-eps "$CLIP_EPS"
          --entropy-coef "$ENTROPY" --value-coef "$VALUE_COEF"
          --epochs "$EPOCHS" --minibatch-size "$MINIBATCH" --seed "$SEED")
    [[ "$AUX_PRIZE_MARGIN" != "0" ]] && CMD+=(--aux-weight "prize_margin=$AUX_PRIZE_MARGIN")
    CMD+=(--eval-every "$EVAL_EVERY" --eval-games "$EVAL_GAMES" --ckpt-every "$CKPT_EVERY")
    CMD+=(--device "$DEVICE")
    [[ "$TB" == "1" ]] && CMD+=(--tb) || CMD+=(--no-tb)
    [[ -n "$WANDB_PROJECT" ]] && CMD+=(--wandb-project "$WANDB_PROJECT")
    [[ "$FORCE" == "1" ]] && CMD+=(--force)
fi
CMD+=(--engine "$ENGINE")

echo "==============================================================================="
echo " 009_alakazam_xl  ·  deck=alakazam  ·  XL+base_residual  ·  cosine LR"
printf ' %-14s %s\n' updates "$UPDATES" model "$MODEL(resid=$BASE_RESIDUAL)" \
                     lr "$LR ->[$LR_SCHEDULE]-> $LR_MIN" device "$DEVICE" \
                     shaping "prize_potential" aux "prize_margin=$AUX_PRIZE_MARGIN"
echo " Command: ${CMD[*]}"
echo "==============================================================================="
echo
cd "$REPO_ROOT"

if [[ -n "$TMUX_SESSION" ]]; then
    command -v tmux >/dev/null || { echo "ERROR: tmux not installed" >&2; exit 3; }
    tmux has-session -t "$TMUX_SESSION" 2>/dev/null && {
        echo "ERROR: tmux session '$TMUX_SESSION' exists (kill it or pick another)" >&2; exit 3; }
    mkdir -p "$(dirname "$LOG")"
    launch=$(printf '%q ' "${CMD[@]}")
    tmux new-session -d -s "$TMUX_SESSION" -c "$REPO_ROOT" \
        "export LD_LIBRARY_PATH=$(printf '%q' "$LD_LIBRARY_PATH"); \
         $launch 2>&1 | tee $(printf '%q' "$LOG"); \
         echo; echo '[training exited]'; exec bash"
    echo "launched in tmux '$TMUX_SESSION'  (attach: tmux attach -t $TMUX_SESSION)"
    echo "  log: tail -f $LOG"
    exit 0
fi
exec "${CMD[@]}"
