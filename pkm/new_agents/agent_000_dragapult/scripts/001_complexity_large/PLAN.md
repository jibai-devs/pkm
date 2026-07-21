# 001 — Complexity: the LARGE network

Goal: find out whether a **bigger, deeper** policy/value network beats the tiny v1
net that plateaued at Kaggle score **600**. This directory holds the two scripts to
do it (`train.sh`, `sweep.sh`) and this plan.

## What "large" means

`--model large` (see `config.MODEL_PRESETS`):

| dim | small (v1) | large |
|---|---|---|
| `n_layers` (trunk attention layers) | 1 | **3** |
| `d_state` | 128 | **384** |
| `d_entity` | 64 | **192** |
| `n_heads` | 4 | **8** |
| `d_opt` | 64 | **192** |
| `d_card` | 32 | **64** |

The 2 extra trunk layers are **pre-LN transformer blocks** (residual + LayerNorm +
FFN) — the ingredients that make depth trainable. Everything is baked into the
config hash + every checkpoint, so a run is fully reproducible and never collides
with a small-net run.

## ⚠️ The thing to fix FIRST (or this is unfalsifiable)

Three checkpoints — including the tiny net — all scored exactly **600.0**, and
`eval`-vs-random saturates near 100% by ~update 130. **600 is almost certainly an
evaluation ceiling, not a capacity limit.** If we scale the net but keep measuring
only against random, we will very likely still see 600 and learn nothing.

So the honest prerequisite is a **discriminating eval**: self-play vs a pool of past
checkpoints (or a fixed strong opponent), reported alongside win-rate-vs-random.
Until that exists, treat any "it still scores 600" result as *inconclusive*, not
*negative*. (This is tracked as the next infra task, not solved by these scripts.)

## Order of operations

1. **Baseline sanity.** `./train.sh` for a short run first (edit `--updates 512`
   down to e.g. 40) to confirm the large net trains stably: watch `evar` climb,
   `kl` stay small (< ~0.05), `ent` ease down without collapsing, and `t/upd` be
   acceptable. A deep net with too-high `lr` shows up as spiking `kl`/`gnorm` and a
   diverging `val` — if so, lower `--lr`.
2. **Sweep the large arch.** `./sweep.sh` — 40 short trials searching lr / entropy /
   clip / epochs / minibatch / gamma / lam **and** all 16 reward weights, at
   `--model large`. Objective `curve_auc` (rewards fast learning, which still
   separates configs under the win-rate ceiling). Resumable.
3. **Lock in the winners.** The sweep prints the best trial's params (incl. every
   `rw_<term>`). Paste them into `train.sh` (`--lr`, `--entropy-coef`, …, and the
   `--reward-weight` block).
4. **Full run.** `./train.sh` end-to-end (512 updates). Checkpoints land every 64
   updates under the experiment's `checkpoints/`.
5. **Pack + submit** a stable numbered snapshot (never `latest.pt` mid-run):
   ```
   uv run pkm new_agents 000_dragapult pack   -e 001_complexity_large --checkpoint <...>/checkpoints/ckpt_<N>.pt
   uv run pkm new_agents 000_dragapult submit -e 001_complexity_large --message "large ckpt_<N>"
   uv run pkm new_agents 000_dragapult status --watch
   ```

## How to run (tmux)

All training runs and sweeps go in the shared **`pkm-train`** tmux session:

```bash
tmux new-session -d -s pkm-train                # once (skip if it exists)
tmux send-keys -t pkm-train \
  "cd <repo> && ./pkm/new_agents/agent_000_dragapult/scripts/001_complexity_large/sweep.sh" Enter
tmux attach -t pkm-train                        # watch; Ctrl-b d to detach
```

Swap `sweep.sh` for `train.sh` to run the full training instead.

## Hyperparameter notes

- `train.sh` ships a **conservative `lr=3e-4`** on purpose. The Optuna winners
  (`lr≈1.48e-3`) were tuned for the tiny net and tend to be unstable at depth.
  Trust `sweep.sh` to find a large-specific lr rather than reusing the small one.
- `--games 64` (vs the usual 16) keeps the 16 workers busier (4 games/worker →
  lower straggler variance) and gives less noisy per-update stats — worth it for a
  slower, more expensive net.
- Depth/width themselves can be swept later by adding `--n-layers` / `--d-state`
  to the search; this pass fixes the architecture and tunes everything else.
