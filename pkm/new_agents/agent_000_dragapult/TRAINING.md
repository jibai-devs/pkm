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

| Command | What it does |
|---|---|
| `info`   | Print the default config, engine backend, and where artifacts go. |
| `smoke`  | Tiny end-to-end run (2 updates × 2 games). Proves the pipeline works. |
| `train`  | PPO self-play training. The main command (§3–4). |
| `resume` | Continue from `checkpoints/latest.pt` for more updates. |
| `eval`   | Win-rate of a checkpoint vs the random baseline. |
| `sweep`  | Optuna hyperparameter search (§9). |

There is also a `justfile` in this directory with shorthands: `just smoke`,
`just train updates=300 games=48 workers=12`, `just eval`.

---

## 3. The complete `train` command (every flag at its default)

This is the **fully-expanded default invocation** — every knob spelled out at the
value you'd get if you omitted it. Copy it, then change what you need. (You never
have to pass all of these; defaults apply for anything you omit.)

```bash
pkm new_agents 000_dragapult train \
    --updates 200 \
    --games 16 \
    --workers 1 \
    --lr 0.0003 \
    --gamma 0.997 \
    --lam 0.95 \
    --clip-eps 0.2 \
    --entropy-coef 0.01 \
    --value-coef 0.5 \
    --epochs 4 \
    --minibatch-size 64 \
    --seed 0 \
    --eval-every 10 \
    --eval-games 100 \
    --ckpt-every 50 \
    --output-dir pkm_data/new_agents/agent_000_dragapult \
    --no-resume \
    --tb \
    --wandb-mode offline \
    --run-name my-first-run
```

### Every flag, explained

| Flag | Default | What it controls |
|---|---|---|
| `--updates` | `200` | Number of collect→improve cycles = **training length**. |
| `--games` | `16` | Self-play games collected per update = the **batch** (each game ≈ 215 decisions). |
| `--workers` | `1` | Parallel rollout processes. **Speed only**, never changes what's learned. Try `8`–`12`. |
| `--lr` | `0.0003` | Adam learning rate. |
| `--gamma` | `0.997` | Reward discount over the ~215-decision game horizon. |
| `--lam` | `0.95` | GAE λ — advantage estimator bias/variance trade-off. |
| `--clip-eps` | `0.2` | PPO trust-region clip (how far the policy may move per update). |
| `--entropy-coef` | `0.01` | Exploration bonus. Raise (→`0.02`–`0.03`) if the policy collapses early. |
| `--value-coef` | `0.5` | Weight of the value (critic) loss in the total loss. |
| `--epochs` | `4` | Optimizer passes over each collected batch. |
| `--minibatch-size` | `64` | SGD minibatch size, in decisions. |
| `--seed` | `0` | RNG seed (weights, sampling, worker seeding). |
| `--eval-every` | `10` | Evaluate vs random every N updates (`0` = never). |
| `--eval-games` | `100` | Games per evaluation. |
| `--ckpt-every` | `50` | Write a numbered `ckpt_<N>.pt` snapshot every N updates. (`latest.pt` is *always* written every update.) |
| `--output-dir` / `-o` | `pkm_data/new_agents/agent_000_dragapult` | Artifact root (checkpoints, logs, TB, wandb, sweeps). |
| `--resume` / `--no-resume` | `--no-resume` | Continue from `latest.pt` instead of starting fresh (or use the `resume` command). |
| `--tb` / `--no-tb` | `--tb` | Log to TensorBoard under `<output>/runs/`. |
| `--log-dir` | *(auto)* | Override the TensorBoard dir (default `<output>/runs/<run-name>`). |
| `--wandb-project` | *(off)* | Enable Weights & Biases logging to this project. Omitted = wandb off. |
| `--wandb-mode` | `offline` | `offline` (local, no network), `online` (cloud, needs `wandb login`), or `disabled`. |
| `--run-name` | *(timestamp)* | Names the TensorBoard subdir and the wandb run. |

---

## 4. A good real run (recommended starting point)

The defaults are conservative (`--games 16 --workers 1`). For an actual run on a
multi-core box, this is a better balance of stability, speed, and feedback:

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
│   └── ckpt_<N>.pt      # permanent snapshot every --ckpt-every updates (default 50)
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

| Column | Meaning | How to read it |
|---|---|---|
| **steps** | decision-samples in the batch (≈ `games × 215`) | just the batch size; varies with game length |
| **pol** | PPO clipped-surrogate policy loss | near-zero & noisy **by design** — watch only that it stays small / doesn't explode; not a progress score |
| **val** | value-head MSE (predicted outcome vs actual ±1) | should stay low / trend down as the critic learns |
| **ent** | policy entropy over options | starts high (~1.3 = exploring), should **decline slowly**; a fast crash to ~0 = premature collapse (raise `--entropy-coef`) |
| **p0/p1** | self-play seat win split | ~50/50 **by construction** (same policy both seats) — *not* a progress signal, only a seat-bias sanity check |
| **eval** | win-rate vs the random baseline | **the real signal.** `-` on non-eval updates; should climb past 50% toward ~100% |

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

| Flag | Default | Meaning |
|---|---|---|
| `--trials` | `30` | Number of Optuna trials. |
| `--updates` | `15` | PPO updates **per trial** — keep short. |
| `--games` | `32` | Games per update within a trial. |
| `--workers` | `8` | Rollout workers per trial. |
| `--eval-games` | `100` | Games used to score each trial. |
| `--study` | `dragapult_ppo` | Study name (SQLite file). |
| `--seed` | `0` | Base seed (offset per trial). |

- The study is **SQLite-backed** at `<output>/sweeps/<study>.db` → **resumable**
  (rerun the same `--study` to add trials) and inspectable with
  `optuna-dashboard sqlite:///…`.
- Each trial also logs to TensorBoard under `runs/sweep-<study>/trial_<n>/`.
- Trials run **sequentially**, each using `--workers` internally — don't stack
  trial-parallelism on top or you'll oversubscribe cores.
- Keep `--updates` short (10–20): a sweep finds good `lr`/`entropy` fast, then you
  do a full-length `train` with the winning params.

**Tuning priority if doing it by hand:** `games` + `updates` (biggest levers) →
`lr` → `entropy-coef` (if the policy collapses) → `epochs`/`minibatch` →
`clip-eps`. `gamma`/`lam` usually left alone.

**Config-only knobs** (not CLI flags — changing them invalidates existing
checkpoints via the config-hash guard, forcing a full retrain): the model dims in
`config.py` (`d_card`, `d_state`, `n_heads`, …) and `max_grad_norm` (0.5).

---

## 10. Notes / gotchas

- **CPU by design.** The net is tiny (~79k params) and the bottleneck is the C++
  engine playing games sequentially (one battle per process). Parallelism is
  multiprocess CPU workers (`--workers`), one engine each — a GPU wouldn't help and
  none is required.
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
