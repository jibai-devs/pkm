# Training — agent_000_dragapult

How to train the Dragapult ex specialist end to end. This agent is **self-contained**:
it owns its whole stack (encoder, model, PPO self-play loop, CLI) and does **not**
use the `pkm/rl` infrastructure. Design notes live in `README.md`; this file is
how to *run* it correctly.

Everything is driven through one command group:

```bash
pkm new_agents 000_dragapult <command> [options]
```

`pkm new_agents 000_dragapult <command> --help` always prints the authoritative
flag list.

---

## 1. TL;DR — first run in three steps

```bash
# 1. Sanity-check the whole pipeline (writes to a temp-ish output, ~seconds):
pkm new_agents 000_dragapult smoke

# 2. Train for real (see §4 for what these mean):
pkm new_agents 000_dragapult train --games 48 --updates 300 --workers 12 \
    --eval-every 5 --eval-games 100

# 3. Measure the result:
pkm new_agents 000_dragapult eval --games 200
```

You can **stop training any time with Ctrl-C** and resume later — nothing is lost
past the update in progress (see §6).

---

## 2. Commands

| Command  | What it does                                                          |
|----------|-----------------------------------------------------------------------|
| `info`   | Print the default config, engine backend, and where artifacts go.     |
| `smoke`  | Tiny end-to-end run (2 updates × 2 games). Proves the pipeline works. |
| `train`  | PPO self-play training. The main command (§3–4).                      |
| `resume` | Continue from `checkpoints/latest.pt` for more updates.               |
| `eval`   | Win-rate of a checkpoint vs the random baseline.                      |
| `sweep`  | Optuna hyperparameter search (§9).                                    |

There is also a `justfile` in this directory with shorthands: `just smoke`,
`just train updates=300 games=48 workers=12`, `just eval`.

---

## 3. The complete `train` command (every flag at its default)

This is the **fully-expanded default invocation** — every knob spelled out at the
value you'd get if you omitted it. Copy it, then change what you need. (You never
have to pass all of these; defaults apply for anything you omit.)

```bash
pkm new_agents 000_dragapult train \
    --updates 256 \
    --games 16 \
    --workers 8 \
    --lr 0.0003 \
    --gamma 0.997 \
    --lam 0.95 \
    --clip-eps 0.2 \
    --entropy-coef 0.01 \
    --value-coef 0.5 \
    --epochs 4 \
    --minibatch-size 64 \
    --seed 0 \
    --eval-every 16 \
    --eval-games 128 \
    --ckpt-every 64 \
    --output-dir pkm_data/new_agents/agent_000_dragapult \
    --no-resume \
    --tb \
    --wandb-mode offline \
    --run-name my-first-run
```

### Every flag, explained

| Flag                       | Default                                   | What it controls                                                                                         |
|----------------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `--updates`                | `256`                                     | Number of collect→improve cycles = **training length**.                                                  |
| `--games`                  | `16`                                      | Self-play games collected per update = the **batch** (each game ≈ 215 decisions).                        |
| `--workers`                | `8`                                       | Parallel rollout processes (one engine each). **Speed only**, never changes what's learned. Raise toward your core count (~`12`); drop to `1` for the single-process path. |
| `--lr`                     | `0.0003`                                  | Adam learning rate.                                                                                      |
| `--gamma`                  | `0.997`                                   | Reward discount over the ~215-decision game horizon.                                                     |
| `--lam`                    | `0.95`                                    | GAE λ — advantage estimator bias/variance trade-off.                                                     |
| `--clip-eps`               | `0.2`                                     | PPO trust-region clip (how far the policy may move per update).                                          |
| `--entropy-coef`           | `0.01`                                    | Exploration bonus. Raise (→`0.02`–`0.03`) if the policy collapses early.                                 |
| `--value-coef`             | `0.5`                                     | Weight of the value (critic) loss in the total loss.                                                     |
| `--epochs`                 | `4`                                       | Optimizer passes over each collected batch.                                                              |
| `--minibatch-size`         | `64`                                      | SGD minibatch size, in decisions.                                                                        |
| `--seed`                   | `0`                                       | RNG seed (weights, sampling, worker seeding).                                                            |
| `--eval-every`             | `16`                                      | Evaluate vs random every N updates (`0` = never).                                                        |
| `--eval-games`             | `128`                                     | Games per evaluation.                                                                                    |
| `--ckpt-every`             | `64`                                      | Write a numbered `ckpt_<N>.pt` snapshot every N updates. (`latest.pt` is *always* written every update.) |
| `--output-dir` / `-o`      | `pkm_data/new_agents/agent_000_dragapult` | Artifact root (checkpoints, logs, TB, wandb, sweeps).                                                    |
| `--resume` / `--no-resume` | `--no-resume`                             | Continue from `latest.pt` instead of starting fresh (or use the `resume` command).                       |
| `--tb` / `--no-tb`         | `--tb`                                    | Log to TensorBoard under `<output>/runs/`.                                                               |
| `--log-dir`                | *(auto)*                                  | Override the TensorBoard dir (default `<output>/runs/<run-name>`).                                       |
| `--wandb-project`          | *(off)*                                   | Enable Weights & Biases logging to this project. Omitted = wandb off.                                    |
| `--wandb-mode`             | `offline`                                 | `offline` (local, no network), `online` (cloud, needs `wandb login`), or `disabled`.                     |
| `--run-name`               | *(timestamp)*                             | Names the TensorBoard subdir and the wandb run.                                                          |

---

## 4. A good real run (recommended starting point)

The default batch is small (`--games 16`). For an actual run on a multi-core box,
a bigger batch (and more workers if you have the cores) is a better balance of
stability, speed, and feedback:

```bash
pkm new_agents 000_dragapult train \
    --updates 300 \
    --games 48 \
    --workers 12 \
    --eval-every 5 \
    --eval-games 100 \
    --run-name dragapult-v1
```

Why these:
- **`--games 48`** — bigger batch than the default 16 → steadier gradients, still
  cheap (~10k decisions/update). 32–64 is the sweet spot; 256+ is stable but heavy.
- **`--workers 12`** — you have the cores; rollouts are the bottleneck, so this is
  the main wall-clock win. (One engine per worker; see §10.)
- **`--eval-every 5`** — the eval-vs-random number is your *only* real progress
  signal, so get it often.

Then watch it live (§8) and stop when `eval` plateaus.

---

## 5. Where everything is saved

Default output root (override with `--output-dir` / `-o`):

```
pkm_data/new_agents/agent_000_dragapult/        # NOTE: this is a git submodule
├── checkpoints/
│   ├── latest.pt        # rewritten after EVERY update (atomic) — the resume point
│   └── ckpt_<N>.pt      # permanent snapshot every --ckpt-every updates (default 64)
├── logs/
│   └── train.csv        # one row per update (columns = the metrics in §7)
├── runs/                # TensorBoard event files, one subdir per run
├── wandb/               # wandb offline run dirs (only if --wandb-project used)
└── sweeps/              # Optuna SQLite studies (only if you run `sweep`)
```

A checkpoint holds **weights + optimizer state + RNG + update index + config
hash** — everything needed to resume a run exactly. Snapshots are ~1.2 MB each and
are **not auto-pruned**, so they accumulate over long runs.

---

## 6. Stopping & resuming

**Stop any time with Ctrl-C.** `latest.pt` is written after every update via
write-to-temp-then-rename, so it can never be left half-written. The most you lose
is the single update in progress when you interrupt.

Continue where you left off:

```bash
pkm new_agents 000_dragapult resume --updates 100 --workers 12
```

`resume` restores the config from `latest.pt` (so model dims etc. match), picks up
at the saved update index, and **appends** to `train.csv` rather than overwriting.

---

## 7. Reading the metrics

Each console line: `update  games  steps  pol  val  ent  p0/p1  eval`.

| Column    | Meaning                                         | How to read it                                                                                                              |
|-----------|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **steps** | decision-samples in the batch (≈ `games × 215`) | just the batch size; varies with game length                                                                                |
| **pol**   | PPO clipped-surrogate policy loss               | near-zero & noisy **by design** — watch only that it stays small / doesn't explode; not a progress score                    |
| **val**   | value-head MSE (predicted outcome vs actual ±1) | should stay low / trend down as the critic learns                                                                           |
| **ent**   | policy entropy over options                     | starts high (~1.3 = exploring), should **decline slowly**; a fast crash to ~0 = premature collapse (raise `--entropy-coef`) |
| **p0/p1** | self-play seat win split                        | ~50/50 **by construction** (same policy both seats) — *not* a progress signal, only a seat-bias sanity check                |
| **eval**  | win-rate vs the random baseline                 | **the real signal.** `-` on non-eval updates; should climb past 50% toward ~100%                                            |

**A healthy run:** `eval` climbing (e.g. 72% → 82% → 95%), `ent` declining
smoothly, `val` low/stable, `pol` small. Because `eval` is only vs *random* (a weak
opponent) it saturates near 100% — beating stronger opponents is future work
(league / MCTS), so don't over-train against random alone.

---

## 8. Live monitoring (TensorBoard / wandb)

Metrics fan out through an **observer pattern** (`monitor.py`): the train loop
notifies a list of `MetricSink`s each update. Built-in sinks: console, CSV,
TensorBoard, Weights & Biases. A failing sink never crashes training (errors are
isolated and printed to stderr). Add a backend by subclassing `MetricSink`.

**TensorBoard — on by default**, local, no account:
```bash
pkm new_agents 000_dragapult train …          # TB on automatically
tensorboard --logdir pkm_data/new_agents/agent_000_dragapult/runs
```
Scalars are grouped: `loss/policy`, `loss/value`, `policy/entropy`,
`eval/win_rate`, `rollout/*`. Disable with `--no-tb`; custom dir with `--log-dir`.

**Backfill from CSV.** For a run that predates TensorBoard logging (only its
`train.csv` exists), replay the whole history into TensorBoard:
```bash
pkm new_agents 000_dragapult import-csv          # -> runs/csv-import/
```
It reads `<output>/logs/train.csv` (override with `--csv`) and writes the same
grouped scalars, so the full history shows up alongside live runs.

**Weights & Biases — opt in with `--wandb-project`**, and **offline by default**
(fully local, no network; run dirs under `<output>/wandb/`). Sync to the cloud
later with `wandb sync`, or stream live with `--wandb-mode online` (needs
`wandb login`):
```bash
pkm new_agents 000_dragapult train --wandb-project dragapult                 # offline
pkm new_agents 000_dragapult train --wandb-project dragapult --wandb-mode online
```
TensorBoard and wandb can run at the same time. `--run-name` names both.

> For **local, live, zero-setup** viewing, TensorBoard is the simplest option;
> wandb-offline is best as a portable record you sync to the cloud on demand.

---

## 9. Automated tuning (Optuna sweep)

`sweep` searches hyperparameters to **maximize eval win-rate vs random**. Each
trial samples `lr` / `entropy_coef` / `clip_eps` / `epochs` / `minibatch` /
`gamma` / `lam`, runs a short training, and reports its win-rate; weak trials are
pruned early (MedianPruner on intermediate evals).

```bash
pkm new_agents 000_dragapult sweep --trials 30 --updates 15 --games 32 --workers 8
```

| Flag           | Default         | Meaning                                 |
|----------------|-----------------|-----------------------------------------|
| `--trials`     | `30`            | Number of Optuna trials.                |
| `--updates`    | `15`            | PPO updates **per trial** — keep short. |
| `--games`      | `32`            | Games per update within a trial.        |
| `--workers`    | `8`             | Rollout workers per trial.              |
| `--eval-games` | `128`           | Games used to score each trial.         |
| `--objective`  | `curve_auc`     | What each trial's score maximizes (below). |
| `--study`      | `dragapult_ppo` | Study name (SQLite file).               |
| `--seed`       | `0`             | Base seed (offset per trial).           |

- The study is **SQLite-backed** at `<output>/sweeps/<study>.db` → **resumable**
  (rerun the same `--study` to add trials) and inspectable with
  `optuna-dashboard sqlite:///…`.
- **A study is bound to one objective.** Each study records its `--objective`. If
  you resume it with a *different* one (or resume a legacy study created before
  objective-tagging that already holds trials), the sweep **prompts you to delete
  it and start over** — because trials scored under different objectives aren't
  comparable (the sampler would model a mixed target). Decline the prompt and pick
  a **new `--study` name** to keep the old results side by side; pass **`--reset`**
  to delete-and-restart non-interactively (e.g. in scripts). You can also delete
  `<output>/sweeps/<study>.db` by hand.
- Each trial also logs to TensorBoard under `runs/sweep-<study>/trial_<n>/`.
- Trials run **sequentially**, each using `--workers` internally — don't stack
  trial-parallelism on top or you'll oversubscribe cores.
- Keep `--updates` short (10–20): a sweep finds good `lr`/`entropy` fast, then you
  do a full-length `train` with the winning params.

**What each trial optimizes (`--objective`).** The default is **`curve_auc`**, not
just the final win-rate: a short trial can really only measure *how fast and how
steadily* a config learns, so scoring on the whole eval curve is both more
informative and less noisy than a single end snapshot.

| Value           | Trial score = …                                                                                                  |
|-----------------|-------------------------------------------------------------------------------------------------------------------|
| `curve_auc`     | **(default)** mean of the eval learning curve (all intermediate evals + the final one). Rewards fast *and* sustained learning; averages out per-eval noise. |
| `final_winrate` | Final eval win-rate only (the legacy behaviour).                                                                  |
| `peak_winrate`  | Best eval reached — robust to an end-of-run collapse.                                                             |
| `net_winrate`   | Final `win_rate − loss_rate` — credits *not losing*; denser than raw wins once win-rate saturates near random's ceiling. |

Win-rate vs *random* saturates near 100% (§7), so as trials cluster at the top a
final-only objective goes flat exactly where you want discrimination; `curve_auc`
and `net_winrate` both push that back. A true prize-differential margin (scoring by
*how many* prizes you win by) is a further step — it needs the eval harness to
surface per-game prize counts, not just W/L/D.

**Tuning priority if doing it by hand:** `games` + `updates` (biggest levers) →
`lr` → `entropy-coef` (if the policy collapses) → `epochs`/`minibatch` →
`clip-eps`. `gamma`/`lam` usually left alone.

**Config-only knobs** (not CLI flags — changing them invalidates existing
checkpoints via the config-hash guard, forcing a full retrain): the model dims in
`config.py` (`d_card`, `d_state`, `n_heads`, …) and `max_grad_norm` (0.5).

---

## 10. Packaging & submitting to Kaggle

Two commands turn a trained checkpoint into a Kaggle submission:

```bash
pkm new_agents 000_dragapult pack       # latest.pt -> <output>/submissions/submission_<ts>.tar.gz
pkm new_agents 000_dragapult submit     # upload the newest bundle to Kaggle
```

**`pack`** extracts the model weights from the checkpoint into `weights.pt`, adds
the submission entry point (`submit_main.py` → `main.py`, a plain `agent(obs)`
callable) and the `pkm/` package, and writes a timestamped `.tar.gz`. It prints
the size and checks it against the 197.7 MiB limit. Pack a specific checkpoint
with `--checkpoint path/to/ckpt.pt`.

**`submit`** runs `kaggle competitions submit -c pokemon-tcg-ai-battle -f <bundle>`
on the newest bundle (override with `--bundle`, `--competition`, `--message`).
Requires the `kaggle` CLI and credentials (`~/.kaggle/kaggle.json`).

**`status`** polls Kaggle for your submissions' status + score — run it after
`submit` to see whether the agent ran (`complete` + a score) or errored:
```bash
pkm new_agents 000_dragapult status            # snapshot
pkm new_agents 000_dragapult status --watch    # poll until the latest finishes
```
An `error` status here is how the torch-in-sandbox risk shows up; pull the
details with `kaggle competitions logs <competition> <submission>`.

> ⚠️ **Inference uses torch.** Unlike the original `pkm` agent (which exports a
> numpy `policy.npz` for torch-free eval), this agent runs a `PolicyValueModel` at
> inference. Torch is **not** bundled (it exceeds the size limit), so the bundle
> only runs on Kaggle **if the cabt sandbox provides torch**. If it doesn't, the
> submission will fail on import — the fix is to add a numpy-only inference path
> (port the encoder/model forward to numpy, like the old agent's export). Verify
> before relying on a live submission.

---

## 10. Notes / gotchas

- **CPU-first; GPU optional (`--device`).** The bottleneck is the C++ engine
  playing games (multiprocess CPU workers, one engine each), which cannot move to
  GPU. For the *small* net a GPU doesn't help. For the *large* net, `--device cuda`
  puts the learner (PPO update) on GPU while rollout + eval stay on CPU, and does
  help there (big net × 5 epochs makes the update non-trivial). See the changelog
  (§10) for the NixOS `LD_LIBRARY_PATH` requirement and the cu129 torch pin.
- **Batch is measured in games, not samples.** `games × ~215 decisions` = the
  actual batch. `--games 256` ≈ 55k steps/update (very stable but heavy); 32–64 is
  the sweet spot.
- **`--workers > 1` uses `parallel.py`** (spawn, one engine per worker) and spawns
  a `torch_shm_manager` helper to share tensors — normal and harmless, cleaned up
  on exit. `--workers 1` is the single-process path (no manager).
- **`TrainConfig.batch_size` (256) is a dead field** — nothing reads it; the batch
  is whatever the rollout collects.
- **Checkpoints/logs live in the `pkm_data` submodule**, so they never bloat the
  main repo.

---

## 10. Changelog & experiment log (2026-07-19 → 07-20)

Recent work on this agent, newest first, with the *why*. (Full training-loop
mechanics are above; this section is the running record of what changed and what
we learned.)

### The 600 ceiling — root cause found
Every submission scores **exactly 600.0** on Kaggle: small net, large net, more or
fewer updates, heuristic shaping — all 600. **Diagnosis: it's a measurement
ceiling, not a capacity limit.** Training is *pure mirror self-play* (one policy
pilots both seats), but the eval/sweep signal was **vs a random bot**, which any
decent agent beats ~100% → the metric saturates. So we were tuning blind. Nothing
was wrong with training; the scoreboard couldn't tell good from great.

### Discriminating eval — `eval --opponent <ckpt>`
Head-to-head between two checkpoints (greedy vs greedy) — a signal that *isn't*
pinned to the random ceiling. `DragapultAgent.from_checkpoint` now loads any
checkpoint (packed `weights.pt`, training `ckpt_N.pt`, or legacy) and rebuilds the
correct architecture on CPU. Demo: `001 ckpt_448` vs `ckpt_64` = **70% (21-9)** —
proves it ranks policies vs-random couldn't. This is the fix that makes every
"did it help?" answerable.

### Configurable network size/depth — `--model` + overrides
`--model {small,medium,large,xl}` presets plus per-dim overrides (`--n-layers`,
`--d-state`, `--d-entity`, `--n-heads`, `--d-opt`, `--d-card`, `--dropout`).
`small` == the original v1 net and is checkpoint-compatible. Depth stacks
`n_layers-1` extra **pre-LN transformer blocks** (residual + LayerNorm + FFN) —
the pieces that make depth trainable. Dims are in the config hash + every
checkpoint. *Why:* test whether more capacity beats the plateau (it didn't move
the Kaggle score — consistent with the ceiling being the eval, not the net).

### GPU training — `--device {cpu,cuda,auto}`
Learner (model + optimizer + PPO update) runs on the device; **rollout workers +
eval always stay on CPU** (self-play is CPU-bound engine work; `parallel.py`
already ships CPU weights to workers). `resolve_device` fails fast if cuda is
unavailable. Runtime-only (not in the config hash). Gotchas hit & fixed:
- **NixOS:** pip/uv CUDA wheels can't find the driver's `libcuda.so`; export
  `LD_LIBRARY_PATH=/run/opengl-driver/lib` (the `001_/002_` scripts do this).
- **torch pin:** `pyproject` uses the **cu129** index, marker-scoped to
  linux+py3.12 so the universal uv lock still resolves elsewhere (CPU wheels).
- **CUDA OOM fix:** the explained-variance step forwarded *all* update steps as
  one batch (~21 GB → OOM); now chunked by minibatch.

### Heuristic reward shaping — `--shaping heuristic`
Full deck-specific reward stack (16 terms, reused from `pkm/rl`), weighted via
`--reward-weight name=value` or searched with `sweep --tune-rewards`. Denser than
the terminal ±1 signal. Per-run scripts weight them explicitly.

### Large-net instability → tuned run (the current experiments)
`001_complexity_large` (large, `lr 3e-4`) trained **unstable**: `kl ~0.05`,
`grad_norm ~3.5–4.5` against the 0.5 clip (clipped ~8× every step). The small-net
lr was too aggressive for the deeper net. `002_large_tuned` lowers `lr → 1e-4`
(+ `entropy 0.02`, `epochs 3`, `games 96`, `minibatch 128`, `dropout 0.1`) →
`kl` back to `0.005–0.012`, far more stable. Scripts: `scripts/001_complexity_large/`
and `scripts/002_large_tuned/`. Whether 002 is *better* (not just stabler) must be
judged **head-to-head** (`002-final` vs `001-final`), never vs random.

### Inference-time MCTS — `pack --inference mcts -K <sims>` (2026-07-20)
Wraps the trained net in PUCT search at decision time (README §9 item 1) — a
strength upgrade with **no retrain**. The choice is baked into the packed bundle
via an `InferenceConfig` (in `weights.pt`), so one checkpoint can be submitted
two ways:

- `pack ... --inference mcts -K 32` → search bundle (K = simulations/decision).
- `pack ...` (default) or `-K 0` / `--inference policy` → plain policy bundle.
- `scripts/pack_variants.sh [exp] [K]` packs BOTH from the same checkpoint.
- Measure before submitting: `eval --inference mcts -K 32 --opponent <ckpt>`
  (head-to-head vs the policy-only agent is the honest "does search help?" test).

How it works: `agent.DragapultAgent._mcts_pick` calls the existing `mcts.search`
(PUCT over the engine's `Search*` forward model). At deployment the search
symbols come from **Kaggle's own `libcg.so`** — no vendored engine is shipped.
`submit_main.py` picks up the bundle's inference config automatically (via
`from_checkpoint`). **Caveat:** MCTS runs a forward search per decision — mind
Kaggle's per-turn + cumulative 600 s clock; tune K.

**Multi-world IS-MCTS (`-W`, 2026-07-20).** The hidden info (our own deck order,
face-down prizes, the opponent's hand/deck) is unknown, so the search
*determinizes* — guesses one full world — before searching. A single world
(W=1) is **biased**: the search optimizes hard for that one guessed layout and
over-commits to draws it merely assumed; raising K makes that *worse* (searches
the fantasy harder — a likely cause of K=4 < K=1 on the leaderboard). `-W <n>`
re-samples n worlds per decision, runs an independent search in each, and
averages the root visit policies (`mcts.search_worlds`) — a move must be good
across many possible draws. Cost is W× the search time, so trade W against K
under the budget (`W=8 -K 4` ≈ `W=1 -K 32`). Both `pack` and `eval` take `-W`;
W is baked into the bundle. This is the principled lever for draw uncertainty —
`--mcts-sims`/K is not.

Tests: `test_inference_mcts.py` (config toggle + W round-trip, bundle round-trip,
`search_worlds` averaging, engine smoke incl. W=2).

**First head-to-head (2026-07-20, `002_large_tuned/ckpt_512.pt`, 30 games each,
agent-under-test vs the SAME checkpoint played as plain policy):**

| inference | searches/decision | win-rate vs policy |
|---|---|---|
| `W=1 -K 32` | 32 | 60.0% (18-12) |
| `W=8 -K 4`  | 32 | 63.3% (19-11) |

Takeaways: (1) **real search (K=32) clearly beats the raw policy head** (~60%,
well above 50%) — the honest signal the leaderboard's moving average couldn't
give; (2) world-averaging is **directionally** better at matched compute
(63.3 vs 60.0) as the bias argument predicts, but it's a 1-game gap — inside the
±9% noise band at n=30, so not yet proven. Needs ~100+ games to separate W=1
from W=8. Both regimes are a real improvement over policy-only.

### Next lever (not yet built)
**Opponent-pool training** — self-play against *past checkpoints*, not just the
current mirror. Improves the training signal AND gives a non-saturated eval; the
change most likely to actually push past 600.
