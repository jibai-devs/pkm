#!/usr/bin/env bash
#
# 015_alakazam_exit_combo_large/train.sh — run 010's ExIt recipe, but with the
# NEW combination-scoring policy head (--policy-head combo) and MORE MCTS.
#
# What's new vs 010 (alakazam ExIt TD(λ) medium):
#   * --policy-head combo   — the combination-scoring multi-select head (scores
#                             whole enumerated option-sets in one pass; see
#                             model.ComboPolicyHead). Under --method exit it
#                             trains via its MARGINALIZED [B,L] (MCTS visit-π
#                             cross-entropy) — the combo scorer gets full
#                             gradient, but the combination-level ranking rides
#                             on the marginal (not a direct combo target). For a
#                             direct combo-distribution benchmark, train under
#                             --method ppo instead.
#   * --model large         — up from 010's medium (~2–3× cost/forward).
#   * --mcts-simulations 32 — up from 010's 16 (2^5): "more mcts", ×2 per search.
#   * --workers 16          — this box has 23 cores; run 010 uses 8, so 16 here
#                             mildly oversubscribes (chosen deliberately).
#
# Engine: --engine kaggle (the pip kaggle libcg.so, ABI-identical to the
# vendored build; 010 used local-nix but that nix output was GC'd — the backend
# is NOT in the config hash, so this is a pure local-convenience difference and
# a later --resume can switch back to local-nix).
#
# LONG "let it cook" run: --updates 8192 sets the cosine T_max, but large+sims32
# makes each update ~4–6× slower than 010 — realistically stop when the eval
# curve is good (latest.pt + ckpt_N.pt persist; safe to Ctrl-C / kill anytime).
#
# LAUNCH INTO A PERSISTENT TMUX (required convention) — e.g. a new window in the
# shared session so it survives detach and outlives this shell:
#     tmux new-window -t pkm-train -n combo
#     tmux send-keys  -t pkm-train:combo \
#         "cd $(git rev-parse --show-toplevel) && bash pkm/new_agents/agent_000_dragapult/scripts/015_alakazam_exit_combo_large/train.sh" Enter
#     tmux attach -t pkm-train         # watch; Ctrl-b d to detach
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

args=(
  # ---- training style + model + task ----
  --method exit                  # AlphaZero-ish expert iteration (MCTS teaches every move)
  --policy-head combo            # ← NEW combination-scoring multi-select head (the point of this run)
  --model large                  # ↑ from 010's medium
  --deck alakazam                # played 60-card list (as 010)
  --aux-weight prize_margin=0.25 # aux head (as 010); training-only, stripped from the bundle

  # ---- expert-iteration (MCTS) knobs — exit-only ----
  --exit-value-target tdlambda --exit-lambda 0.9
  --mcts-worlds 4                # determinized worlds / decision (2^2), as 010
  --mcts-simulations 32          # ↑ 16→32 (2^5): MORE MCTS, ×2 per search
  --mcts-c-puct 1.25 --mcts-temperature 1.0 --determinization sample

  # ---- reward shaping — PPO-only (INERT under exit; kept explicit as 010) ----
  --shaping prize_potential --shaping-coef 1.0

  # ---- optimizer / learn step (rates/coeffs: NOT powers of 2) ----
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5
  --value-coef 0.5 --minibatch-size 64 --epochs 4
  --gamma 0.997 --lam 0.95 --clip-eps 0.2 --entropy-coef 0.01 --seed 0

  # ---- run length + parallelism + device (counts: powers of 2) ----
  --updates 8192                 # cosine T_max; LONG — stop when eval is good (ckpts persist)
  --games 16 --workers 16        # 16 workers (2^4) — oversubscribes alongside 010's 8 on 23 cores
  --device auto                  # this box → cpu (no usable CUDA); MCTS rollout is CPU anyway

  # ---- eval / checkpoint / logging / identity ----
  --eval-every 128 --eval-games 32 --ckpt-every 128
  --experiment 015_alakazam_exit_combo_large
  --run-name 015_alakazam_exit_combo_w4_large
  --tb --engine kaggle --force   # --force: non-interactive (tmux) overwrite guard
)
# `uv run` so the repo venv resolves regardless of the launching shell (fish)
# having it activated; falls back to bare python if uv is absent.
if command -v uv >/dev/null 2>&1; then
  exec uv run python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
else
  exec python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
fi
