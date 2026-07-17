# Training — agent_000_dragapult

Operational guide for the self-contained PPO self-play trainer. This agent does
**not** use the `pkm/rl` infrastructure; it owns its whole stack (encoder, model,
PPO loop, CLI). Design notes live in `README.md`; this file is how to *run* it.

## Quickstart

```bash
# From anywhere (main pkm CLI):
pkm new_agents 000_dragapult info                 # config, engine backend, paths
pkm new_agents 000_dragapult smoke                # 2-update sanity run
pkm new_agents 000_dragapult train --games 48 --updates 300 --workers 12 \
    --eval-every 5 --eval-games 100
pkm new_agents 000_dragapult resume --updates 100 --workers 12
pkm new_agents 000_dragapult eval --games 200

# Or via the justfile in this directory:
just info · just smoke · just train updates=300 games=48 workers=12 · just eval
```

`pkm new_agents 000_dragapult <cmd> --help` lists every flag.

## Where everything is saved

Default output root (override with `--output-dir` / `-o`):

```
pkm_data/new_agents/agent_000_dragapult/          # NOTE: a git submodule
├── checkpoints/
│   ├── latest.pt          # written after EVERY update (atomic)
│   └── ckpt_<N>.pt        # snapshot every --ckpt-every updates (default 50)
└── logs/
    └── train.csv          # one row per update (see columns below)
```

A checkpoint holds **weights + optimizer state + RNG + update index + config
hash** — enough to resume a run exactly.

## Stopping & resuming

**You can stop at any time (Ctrl-C).** `latest.pt` is saved after every update
via a write-to-temp-then-rename, so it can never be left half-written. The most
you lose is the single update in progress when you interrupt.

To continue where you left off:

```bash
pkm new_agents 000_dragapult resume --updates 100 --workers 12
```

`resume` restores the config from `latest.pt` (so model dims etc. match) and
picks up at the saved update index; `train.csv` is appended, not overwritten.

## Reading the metrics

Each line: `update  games  steps  pol  val  ent  p0/p1  eval`.

| Column | Meaning | How to read it |
|---|---|---|
| **steps** | decision-samples in the batch (= `games × ~215`/game) | just the batch size; varies with game length |
| **pol** | PPO clipped-surrogate policy loss | near-zero & noisy **by design**; watch only that it stays small / doesn't explode — not a progress score |
| **val** | value-head MSE (predicted outcome vs actual ±1) | should stay low / trend down as the critic learns |
| **ent** | policy entropy over options | starts high (~1.3 = exploring), should **decline slowly** as it commits; a fast crash to ~0 = premature collapse (raise `--entropy-coef`) |
| **p0/p1** | self-play seat win split | ~50/50 **by construction** (same policy both seats) — *not* a progress signal, only a seat-bias sanity check |
| **eval** | win-rate vs the random baseline | **the real signal.** `-` on non-eval updates; should climb past 50% toward ~100% |

**Healthy run looks like:** `eval` climbing (e.g. 72% → 82% → 95%), `ent`
declining smoothly, `val` stable/low, `pol` small. Because `eval` is only vs
*random* (a weak opponent), it saturates near 100% — beating stronger opponents
is future work (league / MCTS), so don't over-train against random alone.

## Live monitoring (TensorBoard / wandb)

Metrics fan out through an **observer pattern** (`monitor.py`): the train loop
notifies a list of `MetricSink`s each update. Built-in sinks: console, CSV,
TensorBoard, Weights & Biases. Add a backend by subclassing `MetricSink`.

**TensorBoard — on by default**, local, no account. Writes to
`<output>/runs/<run-name>/`:
```bash
pkm new_agents 000_dragapult train …          # TB on automatically
tensorboard --logdir pkm_data/new_agents/agent_000_dragapult/runs
# turn off with --no-tb; custom dir with --log-dir PATH
```
Scalars are grouped: `loss/policy`, `loss/value`, `policy/entropy`,
`eval/win_rate`, `rollout/*`.

**Weights & Biases — opt in with `--wandb-project`.** Default mode is
**`offline`** (fully local, no network; run dirs under `<output>/wandb/`); sync to
the cloud later with `wandb sync`. `--wandb-mode online` streams live (needs
`wandb login`); `disabled` is a no-op.
```bash
pkm new_agents 000_dragapult train --wandb-project dragapult          # offline
pkm new_agents 000_dragapult train --wandb-project dragapult --wandb-mode online
```
Both can run at once. A failing sink never crashes training (errors are isolated,
printed to stderr). `--run-name` names the TB subdir + wandb run.

## Hyperparameters

Exposed as `train` flags (defaults in parens):

| Group | Flags |
|---|---|
| length / data | `--updates` (200), `--games` per update (16), `--workers` (1) |
| PPO | `--lr` (3e-4), `--epochs` (4), `--minibatch-size` (64), `--clip-eps` (0.2), `--entropy-coef` (0.01), `--value-coef` (0.5) |
| credit assignment | `--gamma` (0.997), `--lam` / GAE-λ (0.95) |
| bookkeeping | `--seed`, `--eval-every`, `--eval-games`, `--ckpt-every` |

**Priority to tune:** `games` + `updates` (biggest levers) → `lr` →
`entropy-coef` (if the policy collapses) → `epochs`/`minibatch` → `clip-eps`.
`gamma`/`lam` usually left alone. `workers` only changes **speed**, never what's
learned.

Config-only (not flags, because changing them invalidates checkpoints via the
config-hash guard → full retrain): the model dims in `config.py`
(`d_card`, `d_state`, `n_heads`, …) and `max_grad_norm` (0.5).

## Automated tuning (Optuna sweep)

`sweep` searches hyperparameters to **maximize eval win-rate vs random**. Each
trial samples `lr`/`entropy_coef`/`clip_eps`/`epochs`/`minibatch`/`gamma`/`lam`,
runs a short training, and reports its win-rate; weak trials are pruned early
(MedianPruner on intermediate evals).

```bash
pkm new_agents 000_dragapult sweep --trials 30 --updates 15 --games 32 --workers 8
```

- Study is **SQLite-backed** at `<output>/sweeps/<study>.db` → **resumable** (rerun
  the same `--study` to add trials) and inspectable: `optuna-dashboard sqlite:///…`.
- Each trial also logs to TensorBoard under `runs/sweep-<study>/trial_<n>/`.
- Trials run **sequentially**, each using `--workers` internally — don't stack
  trial-parallelism on top (you'd oversubscribe cores).
- Keep `--updates` short (10–20): sweeps find good `lr`/`entropy` fast; then do a
  full-length `train` with the winning params.

## Notes / gotchas

- **CPU by design.** The net is tiny (~79k params) and the bottleneck is the C++
  engine playing games sequentially (one battle per process). Parallelism is
  multiprocess CPU workers (`--workers`), one engine each — a GPU wouldn't help
  and none is required.
- **Batch is measured in games, not samples.** `games × ~215 decisions` = the
  actual batch. `--games 256` ≈ 55k steps/update (large & stable but heavy);
  32–64 is a good sweet spot.
- **`TrainConfig.batch_size` (256) is a dead field** — nothing reads it; the
  batch is whatever the rollout collects.
- **`--workers > 1` uses `parallel.py`** (spawn, one engine per worker). Verified
  working; `--workers 1` is the simplest single-process path.
