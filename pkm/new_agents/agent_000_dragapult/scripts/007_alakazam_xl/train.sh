#!/usr/bin/env bash
#
# 007_alakazam_xl/train.sh — dedicated training run for the **alakazam** deck
# (Mega Alakazam / Dudunsparce psychic control).
#
# Two parts are HARD-WIRED for correctness and cannot be overridden here:
#   * --deck alakazam        (this is the alakazam script)
#   * --shaping prize_potential   (deck-agnostic; the dragapult_heuristic reward
#                                  stack must NOT be applied to this deck)
# Everything else is a configurable via environment variable — see the block
# below. Run `scripts/007_alakazam_xl/train.sh` with no args for the defaults,
# or override any knob:  MODEL=large UPDATES=512 LR=3e-4 scripts/.../train.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

# NixOS: put the NVIDIA driver libs on the loader path so torch.cuda actually
# sees the GPU (else DEVICE=cuda errors / DEVICE=auto silently falls back to CPU).
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ============================ CONFIGURABLES ==================================
# Run identity / lifecycle
MODE="${MODE:-train}"                 # train | resume  (resume continues latest.pt)
EXP="${EXP:-007_alakazam_xl}"         # experiment name -> <OUTPUT>/experiments/<EXP>/
OUTPUT="${OUTPUT:-}"                   # artifact root (empty = repo default DATA_DIR)
FORCE="${FORCE:-0}"                   # 1 = overwrite an existing experiment ckpt
ENGINE="${ENGINE:-local-nix}"         # engine backend (local-nix | vendored | kaggle)
# Detached tmux: empty = run in the foreground. Non-empty = launch the run in a
# new detached tmux session of that name and return immediately (survives logout;
# tail LOG or `tmux attach -t <name>` to watch). NB: do NOT use "pkm-train" — that
# session hosts Claude Code itself.
TMUX_SESSION="${TMUX_SESSION:-}"
# tmux stdout mirror lives WITH the run's artifacts under pkm_data (not the source tree).
DATA_ROOT="${OUTPUT:-$REPO_ROOT/pkm_data/new_agents/agent_000_dragapult}"
LOG="${LOG:-$DATA_ROOT/experiments/$EXP/logs/stdout.log}"

# Compute
DEVICE="${DEVICE:-cuda}"              # cuda | cpu | auto (auto: cuda if available)
WORKERS="${WORKERS:-8}"              # parallel self-play rollout workers (CPU)

# Training length
UPDATES="${UPDATES:-256}"            # number of PPO updates
GAMES="${GAMES:-16}"                # self-play games collected per update

# Network architecture
MODEL="${MODEL:-xl}"                 # small | medium | large | xl
BASE_RESIDUAL="${BASE_RESIDUAL:-1}"  # 1 = uniform-residual trunk (recommended for xl)

# PPO hyperparameters
LR="${LR:-1e-4}"                     # Adam LR (1e-4 stable at xl depth)
GAMMA="${GAMMA:-0.997}"              # discount
LAM="${LAM:-0.95}"                   # GAE lambda
CLIP_EPS="${CLIP_EPS:-0.2}"          # PPO clip epsilon
ENTROPY="${ENTROPY:-0.01}"           # entropy bonus
VALUE_COEF="${VALUE_COEF:-0.5}"      # value-loss coefficient
EPOCHS="${EPOCHS:-4}"                # PPO epochs per update
MINIBATCH="${MINIBATCH:-64}"         # minibatch size (decisions)
SEED="${SEED:-0}"                    # RNG seed

# Auxiliary loss (deck-agnostic — safe for any deck)
AUX_PRIZE_MARGIN="${AUX_PRIZE_MARGIN:-0.25}"  # 0 turns it off

# Evaluation + checkpointing cadence
EVAL_EVERY="${EVAL_EVERY:-16}"       # eval vs random every N updates (0 = never)
EVAL_GAMES="${EVAL_GAMES:-128}"      # games per eval
CKPT_EVERY="${CKPT_EVERY:-64}"       # checkpoint snapshot every N updates

# Logging
WANDB_PROJECT="${WANDB_PROJECT:-}"   # non-empty -> log to this Weights & Biases project
TB="${TB:-1}"                        # 1 = TensorBoard on
# =============================================================================

if [[ "$MODE" != "train" && "$MODE" != "resume" ]]; then
    echo "ERROR: MODE must be 'train' or 'resume' (got '$MODE')" >&2
    exit 2
fi

CMD=(uv run pkm new_agents 000_dragapult "$MODE"
     --experiment "$EXP" --updates "$UPDATES" --games "$GAMES" --workers "$WORKERS")
[[ -n "$OUTPUT" ]] && CMD+=(--output-dir "$OUTPUT")

if [[ "$MODE" == "train" ]]; then
    # --- correctness: fixed for the alakazam deck ---
    CMD+=(--deck alakazam --shaping prize_potential)
    # --- architecture ---
    CMD+=(--model "$MODEL")
    [[ "$BASE_RESIDUAL" == "1" ]] && CMD+=(--base-residual)
    # --- PPO ---
    CMD+=(--lr "$LR" --gamma "$GAMMA" --lam "$LAM" --clip-eps "$CLIP_EPS"
          --entropy-coef "$ENTROPY" --value-coef "$VALUE_COEF"
          --epochs "$EPOCHS" --minibatch-size "$MINIBATCH" --seed "$SEED")
    # --- aux ---
    [[ "$AUX_PRIZE_MARGIN" != "0" ]] && CMD+=(--aux-weight "prize_margin=$AUX_PRIZE_MARGIN")
    # --- eval + ckpt cadence ---
    CMD+=(--eval-every "$EVAL_EVERY" --eval-games "$EVAL_GAMES" --ckpt-every "$CKPT_EVERY")
    # --- compute + logging ---
    CMD+=(--device "$DEVICE")
    [[ "$TB" == "1" ]] && CMD+=(--tb) || CMD+=(--no-tb)
    [[ -n "$WANDB_PROJECT" ]] && CMD+=(--wandb-project "$WANDB_PROJECT")
    [[ "$FORCE" == "1" ]] && CMD+=(--force)
fi
CMD+=(--engine "$ENGINE")

echo "==============================================================================="
echo " 007_alakazam_xl  ·  deck=alakazam  ·  shaping=prize_potential (deck-agnostic)"
echo "-------------------------------------------------------------------------------"
printf ' %-14s %s\n' mode "$MODE"       exp "$EXP"          engine "$ENGINE"
if [[ "$MODE" == "train" ]]; then
    printf ' %-14s %s\n' model "$MODEL (base_residual=$BASE_RESIDUAL)" \
                         device "$DEVICE" workers "$WORKERS" \
                         updates "$UPDATES" "games/upd" "$GAMES" \
                         lr "$LR" gamma "$GAMMA" lam "$LAM" \
                         clip_eps "$CLIP_EPS" entropy "$ENTROPY" \
                         epochs "$EPOCHS" minibatch "$MINIBATCH" seed "$SEED" \
                         aux "prize_margin=$AUX_PRIZE_MARGIN" \
                         eval "every $EVAL_EVERY / $EVAL_GAMES games" \
                         ckpt_every "$CKPT_EVERY" \
                         wandb "${WANDB_PROJECT:-off}"
fi
echo " Command: ${CMD[*]}"
echo "==============================================================================="
echo
cd "$REPO_ROOT"

if [[ -n "$TMUX_SESSION" ]]; then
    command -v tmux >/dev/null || { echo "ERROR: tmux not installed" >&2; exit 3; }
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "ERROR: tmux session '$TMUX_SESSION' already exists — pick another " \
             "TMUX_SESSION or kill it (tmux kill-session -t $TMUX_SESSION)." >&2
        exit 3
    fi
    mkdir -p "$(dirname "$LOG")"
    # %q-quote the argv so it survives re-parsing by the pane's shell, and pin
    # LD_LIBRARY_PATH inside the pane (the tmux server may not inherit it). Keep
    # the pane alive after exit so the final output/traceback stays readable.
    launch=$(printf '%q ' "${CMD[@]}")
    tmux new-session -d -s "$TMUX_SESSION" -c "$REPO_ROOT" \
        "export LD_LIBRARY_PATH=$(printf '%q' "$LD_LIBRARY_PATH"); \
         $launch 2>&1 | tee $(printf '%q' "$LOG"); \
         echo; echo '[training exited — press q or Ctrl-b d to leave]'; exec bash"
    echo "launched in tmux session '$TMUX_SESSION'"
    echo "  attach : tmux attach -t $TMUX_SESSION   (detach: Ctrl-b d)"
    echo "  log    : tail -f $LOG"
    exit 0
fi

exec "${CMD[@]}"
