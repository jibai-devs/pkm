#!/usr/bin/env bash
#
# 016_alakazam_exit_attn_large/train.sh — a CONTROLLED A/B against 015: run the
# exact same ExIt recipe, changing ONLY the policy head (combo → attn).
#
# What's new vs 015 (alakazam ExIt combo large):
#   * --policy-head attn --n-dec-layers 3
#         The NEW transformer-decoder head (model.OptionDecoderHead): the L
#         presented options self-attend to each other AND cross-attend to the
#         encoder's per-entity board tokens, then read out one logit each —
#         agent_001's board-attending decoder over OUR option/entity tokens.
#         Unlike the combo head, it emits [B,L] DIRECTLY, so under --method exit
#         its full expressiveness is trained by the MCTS visit-π cross-entropy
#         (no marginalized/indirect caveat) — a cleaner ExIt fit than combo.
#         Depth 3 balances the large preset's 3-layer encoder; 13-token board +
#         short option lists saturate fast, so deeper decoders mostly add params.
#         The higher-leverage capacity knob (if this proves out) is --d-entity,
#         not more decoder layers — deferred, to keep this a clean A/B vs 015.
#   * --workers 8           down from 015's 16 (box is free now; 8 is a good
#                           citizen and the requested setting).
#   * --engine local-nix    015 used kaggle because the nix output was GC'd; it
#                           has since been rebuilt (engine/result resolves), and
#                           the backend is NOT in the config hash, so this is a
#                           pure local-convenience difference.
# Everything else is IDENTICAL to 015 so any eval delta is attributable to the
# head. Compare with `eval --opponent <015 combo ckpt>` and the marginal 723.4
# net (010 ckpt_4608).
#
# LONG "let it cook" run: --updates 8192 sets the cosine T_max; large + sims 32
# makes each update slow — stop when the eval curve is good (latest.pt +
# ckpt_N.pt persist; safe to Ctrl-C / kill anytime).
#
# LAUNCH INTO A PERSISTENT TMUX (required convention):
#     tmux new-session -d -s pkm-train
#     tmux send-keys -t pkm-train \
#         "cd $(git rev-parse --show-toplevel) && bash pkm/new_agents/agent_000_dragapult/scripts/016_alakazam_exit_attn_large/train.sh" Enter
#     tmux attach -t pkm-train         # watch; Ctrl-b d to detach
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

args=(
  # ---- training style + model + task ----
  --method exit                  # AlphaZero-ish expert iteration (MCTS teaches every move)
  --policy-head attn             # ← NEW transformer-decoder head (the point of this run)
  --n-dec-layers 3               # ← decoder depth, balanced with the large encoder (3 layers)
  --model large                  # same as 015
  --deck alakazam                # played 60-card list (as 015)
  --aux-weight prize_margin=0.25 # aux head (as 015); training-only, stripped from the bundle

  # ---- expert-iteration (MCTS) knobs — exit-only ----
  --exit-value-target tdlambda --exit-lambda 0.9
  --mcts-worlds 4                # determinized worlds / decision (2^2), as 015
  --mcts-simulations 32          # 2^5, as 015
  --mcts-c-puct 1.25 --mcts-temperature 1.0 --determinization sample

  # ---- reward shaping — PPO-only (INERT under exit; kept explicit as 015) ----
  --shaping prize_potential --shaping-coef 1.0

  # ---- optimizer / learn step (rates/coeffs: NOT powers of 2) ----
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5
  --value-coef 0.5 --minibatch-size 64 --epochs 4
  --gamma 0.997 --lam 0.95 --clip-eps 0.2 --entropy-coef 0.01 --seed 0

  # ---- run length + parallelism + device (counts: powers of 2) ----
  --updates 8192                 # cosine T_max; LONG — stop when eval is good (ckpts persist)
  --games 16 --workers 8         # 8 workers (2^3) — box is free (010/015 stopped)
  --device auto                  # this box → cpu (no usable CUDA); MCTS rollout is CPU anyway

  # ---- eval / checkpoint / logging / identity ----
  --eval-every 128 --eval-games 32 --ckpt-every 128
  --experiment 016_alakazam_exit_attn_large
  --run-name 016_alakazam_exit_attn_w4_large
  --tb --engine local-nix --force   # --force: non-interactive (tmux) overwrite guard
)
# `uv run` so the repo venv resolves regardless of the launching shell (fish)
# having it activated; falls back to bare python if uv is absent.
if command -v uv >/dev/null 2>&1; then
  exec uv run python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
else
  exec python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
fi
