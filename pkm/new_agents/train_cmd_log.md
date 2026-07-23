# new_agents — training command log

Every training command run for anything under `new_agents/` — **`train`,
`sweep`, and `resume`** — gets appended here, fully explicit, with a comment on
each flag and its alternatives. Keeps the run history in the repo (mirrors
`agent_000_dragapult/submission_log.md` for Kaggle bundles).

Commands are written as bash `args=(…)` arrays so **every flag carries an inline
comment** (what it does + other options) and the block still runs. Newest
entries at the bottom. Convention also recorded in the project `CLAUDE.md`.

**Style rules (see CLAUDE.md):** (1) prefer **powers of 2** for count/size params
where sensible — games, workers, updates, minibatch, mcts sims/worlds, eval/ckpt
intervals, trials (NOT rates/coeffs: lr/gamma/lambda/clip/entropy stay as tuned);
(2) prefix each `--experiment` with the next **zero-padded number** (`010`,
`011`, … continuing the `experiments/NNN_*` series; 000–009 already exist).

## How to launch — persistent tmux (required)

All `train` / `sweep` / `resume` runs go into a **persistent tmux** session so
they survive disconnect and **stay alive after the command finishes** (shell
returns to its prompt, scrollback intact). Do NOT run real training as a
foreground or background shell job.

```bash
tmux new-session -d -s pkm-train        # once — a REAL shell (see gotcha below)
tmux attach -t pkm-train                # then paste an args=(…) block + the
                                        # `python -m … "${args[@]}"` line into it.
                                        # Ctrl-b then d to detach (run keeps going).
```

Gotcha: if the `pkm-train` session is the pane where the Claude Code CLI itself
runs, pasting/`send-keys` types into Claude's prompt, not a shell — use a
different shell session/window in that case.

## Where the metrics land

Every run writes, keyed by its `--experiment <name>` (root = `--output-dir`,
default `pkm_data/new_agents/agent_000_dragapult`), under
`…/experiments/<experiment>/`:

- **`logs/train.csv`** — per-update metrics (the main thing to inspect;
  `pol_loss`/`val_loss`/`eval_win_rate`/timing). Read it directly, e.g.:
  ```bash
  cat pkm_data/new_agents/agent_000_dragapult/experiments/<experiment>/logs/train.csv
  ```
- **`runs/<run-name>/`** — TensorBoard event files (`tensorboard --logdir` it).
- **`checkpoints/`** — `latest.pt` + `ckpt_N.pt`.

Each run entry below repeats its own concrete `📊 metrics:` path.

Flag-value legend used below: `alt:` = other accepted values; `PPO-only` = read
only by `--method ppo` (inert under `exit`); `exit-only` = the reverse.

---

## 010 — agent_000 · ALAKAZAM · ExIt (TD(λ)+W-worlds, medium) — MASTER (all flags annotated)

First **alakazam** training with the new expert-iteration levers (TD(λ) value
targets + W-world determinization). This block annotates **every** flag; later
blocks re-annotate only what changes. The learned vocab already spans all decks,
so only the played 60-card list changes vs the dragapult runs.

📊 metrics: `pkm_data/new_agents/agent_000_dragapult/experiments/010_alakazam_exit_tdlambda_medium/logs/train.csv`
(TB: `…/experiments/010_alakazam_exit_tdlambda_medium/runs/010_alakazam_exit_w4_medium/`)

**Cost (this CPU box, extrapolated from the measured ~3.7s/game at medium/w2/
sims16):** with **`--mcts-worlds 4`** (≈2× the w2 search) and **`--updates
8192`**, 16 games / 8 workers ≈ **~3.5–4.5 days on CPU**. This is a "let it cook"
run — stop anytime (`latest.pt` + `ckpt_N.pt` persist; watch the eval curve in
`train.csv`). `--eval-every 128` below → 64 eval points (not 512). ExIt logs
`ent/kl/clip/gnorm/evar` as `0.00` (PPO-only metrics; N/A).

```bash
args=(
  # ---- training style + model + task ----
  --method exit                  # TRAINING STYLE. alt: ppo. exit = MCTS teaches every move (AlphaZero-ish)
  --model medium                 # net size preset. alt: small(=v1)|large|xl. (per-dim overrides: --n-layers/--d-state/--d-entity/--n-heads/--d-opt/--d-card)
  --policy-head marginal         # policy head. alt: autoreg (STOP-token multi-select). NOTE: exit only trains step-0 marginal, so autoreg's conditioning won't learn here — train autoreg under --method ppo
  --deck alakazam                # played 60-card list. alt: dragapult (+ any deck in deck.DECKS)
  --aux-weight prize_margin=0.25 # enable an aux head (repeatable: --aux-weight name=w). deck-agnostic. alt: omit = no aux. Training-only (stripped from Kaggle bundle)

  # ---- expert-iteration (MCTS) knobs — exit-only ----
  --exit-value-target tdlambda   # value target. alt: mc (raw ±1 outcome, v1). tdlambda = blend outcome + MCTS root value
  --exit-lambda 0.9              # tdlambda EMA factor (0..1). higher = trust outcome more. inert unless tdlambda. (coeff → not power-of-2)
  --mcts-worlds 4                # determinized worlds averaged per decision (IS-MCTS, 2^2). alt: 1|2. cost ×W (4 ≈ 2× the search of w2)
  --mcts-simulations 16          # PUCT sims per decision (2^4). more = stronger teacher, linear cost. (large run uses 32)
  --mcts-c-puct 1.25             # PUCT exploration constant. higher = explore more. (coeff → not power-of-2)
  --mcts-temperature 1.0         # visit-count temperature for π. 1.0 = ∝ visits; →0 = argmax
  --determinization sample       # how hidden zones are filled. key into determinize.DETERMINIZERS

  # ---- reward shaping — PPO-only (INERT under --method exit; kept explicit) ----
  --shaping prize_potential      # deck-agnostic. alt: terminal (sparse ±1) | dragapult_heuristic (dragapult-ONLY). exit ignores this
  --shaping-coef 1.0             # scale on the shaping term (0.0 == terminal). PPO-only
  # --reward-weight name=value   # (repeatable) override a heuristic term; only with --shaping dragapult_heuristic (dragapult deck)

  # ---- optimizer / learn step (rates/coeffs: NOT powers of 2) ----
  --lr 1e-4                      # Adam LR (or cosine start). 1e-4 = stable for deeper nets
  --lr-schedule cosine           # alt: constant. cosine anneals lr→lr-min over the run
  --lr-min 1e-5                  # cosine floor (eta_min). inert unless cosine
  --value-coef 0.5               # weight on the value (MSE) loss
  --minibatch-size 64            # decisions per optimizer minibatch (2^6)
  --epochs 4                     # PPO-only: passes per update (2^2). inert for exit
  --gamma 0.997                  # PPO-only: discount (long ~77-decision horizon)
  --lam 0.95                     # PPO-only: GAE lambda
  --clip-eps 0.2                 # PPO-only: PPO clip epsilon
  --entropy-coef 0.01            # PPO-only: entropy bonus
  --seed 0                       # RNG seed (offset per worker)

  # ---- run length + parallelism + device (counts: powers of 2) ----
  --updates 8192                 # number of updates (2^13) — LONG run (~3.5–4.5 days on CPU w4). cosine T_max = this. safe to stop early: latest.pt/ckpt_N.pt persist
  --games 16                     # self-play games per update (2^4)
  --workers 8                    # parallel self-play workers (2^3). exit cost is CPU-bound in workers
  --device auto                  # GPU/CPU. alt: cpu | cuda. auto = cuda if available else cpu. THIS box has no usable CUDA → cpu; --device cuda ERRORS here. (Only the learner update uses GPU; MCTS rollout is always CPU)

  # ---- eval / checkpoint / logging / identity ----
  --eval-every 128               # eval vs RANDOM every N updates (2^7; 0 = never) — loosened for the long run (64 evals, not 512). NB: vs-random saturates → weak signal
  --eval-games 32                # games per eval (2^5)
  --ckpt-every 128               # snapshot ckpt_N.pt every N updates (2^7; latest.pt always written; keep_last=5 prunes)
  --experiment 010_alakazam_exit_tdlambda_medium  # artifact dir <output>/experiments/<name>/ (-e). numbered prefix. config-hash guards resume
  --run-name 010_alakazam_exit_w4_medium           # TB subdir / wandb run name
  --tb                           # TensorBoard scalars to <experiment>/runs/. alt: --no-tb. (CSV always written)
  --engine local-nix             # engine backend. alt: kaggle | local. default local-nix (vendored cg.so)
  # --output-dir PATH            # (-o) artifact root; default is the repo's DATA_DIR
  # --wandb-project NAME         # also log to Weights & Biases (+ --wandb-mode offline|online|disabled)
  # --resume                     # continue from <experiment>/checkpoints/latest.pt (see Resume section)
  # --force                      # (-f) overwrite an existing experiment without prompting (needed for non-interactive/tmux)
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

Smoke to validate + time first (writes only to scratchpad; no TB/eval):

```bash
args=(
  --method exit --exit-value-target tdlambda --exit-lambda 0.9
  --mcts-worlds 4 --mcts-simulations 16 --determinization sample
  --model medium --deck alakazam --aux-weight prize_margin=0.25
  --updates 4 --games 2 --workers 4        # tiny (all powers of 2): just measure seconds/update (now w4, so ~2× the w2 smoke)
  --device auto --eval-every 0 --ckpt-every 128 --no-tb --force
  --output-dir /tmp/exit_smoke --experiment alakazam_smoke_measure
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

---

## 011 — agent_000 · DRAGAPULT · ExIt (TD(λ)+W-worlds, medium)

Same as 010 but the dragapult deck. **Deltas only** (all other flags as 010):

📊 metrics: `pkm_data/new_agents/agent_000_dragapult/experiments/011_dragapult_exit_tdlambda_medium/logs/train.csv`
(TB: `…/experiments/011_dragapult_exit_tdlambda_medium/runs/011_dragapult_exit_w4_medium/`)

```bash
args=(
  --method exit --exit-value-target tdlambda --exit-lambda 0.9 --determinization sample
  --model medium --mcts-worlds 4 --mcts-simulations 16 --mcts-c-puct 1.25 --mcts-temperature 1.0
  --deck dragapult               # ← the only change vs 010
  --aux-weight prize_margin=0.25 --shaping prize_potential --shaping-coef 1.0
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --value-coef 0.5 --minibatch-size 64 --seed 0
  --updates 8192 --games 16 --workers 8     # updates 2^13 (~3.5–4.5 days CPU; see 010 cost note)
  --device auto --eval-every 128 --eval-games 32 --ckpt-every 128
  --experiment 011_dragapult_exit_tdlambda_medium --run-name 011_dragapult_exit_w4_medium --tb
  --engine local-nix
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

---

## 012 — agent_000 · DRAGAPULT · ExIt (large) — the expensive one

Scaled up. **Deltas only** vs 010/011:

📊 metrics: `pkm_data/new_agents/agent_000_dragapult/experiments/012_dragapult_exit_tdlambda_large/logs/train.csv`
(TB: `…/experiments/012_dragapult_exit_tdlambda_large/runs/012_dragapult_exit_w4_large/`)

```bash
args=(
  --method exit --exit-value-target tdlambda --exit-lambda 0.9 --determinization sample
  --model large                  # ↑ from medium: ~2–3× cost/forward
  --mcts-worlds 4                # ↑ 2→4 (2^2): ×2 searches/decision
  --mcts-simulations 32          # ↑ 16→32 (2^5): ×2 per search
  --deck dragapult --aux-weight prize_margin=0.25
  --mcts-c-puct 1.25 --mcts-temperature 1.0
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --value-coef 0.5 --minibatch-size 64 --seed 0
  --updates 256 --games 16 --workers 8      # updates 2^8
  --device auto                  # cpu here → this run is ~tens of hours-to-days; use only with a GPU or lots of patience
  --eval-every 16 --eval-games 128 --ckpt-every 32
  --experiment 012_dragapult_exit_tdlambda_large --run-name 012_dragapult_exit_w4_large --tb
  --engine local-nix
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

---

## 013 — agent_000 · DRAGAPULT · PPO to train the autoregressive head (②)

The autoregressive STOP-token head is only *fully* trained under **PPO** (its
sampler + logprob use the conditioned path; under `exit` only the step-0 marginal
learns). Here the PPO-only knobs (gamma/lam/clip-eps/entropy-coef/epochs) are
LIVE, and the exit/MCTS knobs are absent.

📊 metrics: `pkm_data/new_agents/agent_000_dragapult/experiments/013_dragapult_autoreg_ppo/logs/train.csv`
(TB: `…/experiments/013_dragapult_autoreg_ppo/runs/013_dragapult_autoreg_ppo/`)

```bash
args=(
  --method ppo                   # TRAINING STYLE: policy-gradient self-play (PPO+GAE)
  --policy-head autoreg          # ② STOP-token multi-select head (the point of this run). alt: marginal
  --model large --deck dragapult --aux-weight prize_margin=0.25

  --shaping prize_potential      # LIVE under ppo. alt: terminal | dragapult_heuristic
  --shaping-coef 1.0

  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5
  --gamma 0.997                  # discount (LIVE)
  --lam 0.95                     # GAE lambda (LIVE)
  --clip-eps 0.2                 # PPO clip (LIVE)
  --entropy-coef 0.01            # entropy bonus (LIVE)
  --value-coef 0.5
  --epochs 4                     # PPO passes/update (2^2, LIVE)
  --minibatch-size 64 --seed 0

  --updates 512 --games 32 --workers 8       # powers of 2. PPO rollouts are cheap vs exit → more games/updates
  --device auto                  # cpu here (fine: PPO is much cheaper than exit)
  --eval-every 16 --eval-games 128 --ckpt-every 32
  --experiment 013_dragapult_autoreg_ppo --run-name 013_dragapult_autoreg_ppo --tb --engine local-nix
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

---

## Resume an interrupted run

`--resume` reloads `<experiment>/checkpoints/latest.pt` and continues. Pass the
SAME `--experiment` and repeat every **config-hash** flag (model dims,
`--policy-head`, `--deck`, `--method`, all mcts/exit knobs, `--shaping`,
`--aux-weight`) exactly, or the checkpoint won't load. Example (resumes the
alakazam 010 run):

📊 metrics: appends to the SAME file as the original run —
`pkm_data/new_agents/agent_000_dragapult/experiments/010_alakazam_exit_tdlambda_medium/logs/train.csv`

```bash
args=(
  --resume                       # continue from latest.pt instead of a fresh net
  --experiment 010_alakazam_exit_tdlambda_medium   # MUST match the original run's experiment
  # --- config-hash flags: MUST match the original exactly ---
  --method exit --exit-value-target tdlambda --exit-lambda 0.9
  --mcts-worlds 4 --mcts-simulations 16 --mcts-c-puct 1.25 --mcts-temperature 1.0
  --determinization sample --model medium --deck alakazam --aux-weight prize_margin=0.25
  --shaping prize_potential --shaping-coef 1.0
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --value-coef 0.5 --minibatch-size 64 --seed 0
  # --- run-length / runtime flags: free to change on resume ---
  --updates 8192                 # ADDITIONAL updates from where it stopped (set to what's left)
  --games 16 --workers 8 --device auto
  --eval-every 128 --eval-games 32 --ckpt-every 128
  --run-name 010_alakazam_exit_w4_medium --tb --engine local-nix
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

---

## 014 — agent_000 · DRAGAPULT · Optuna hyperparameter sweep

`sweep` runs many short PPO trials (Optuna samples lr/entropy/clip/epochs/
minibatch/gamma/lam) and keeps the best by eval win-rate. It is PPO-oriented —
there are **no** `--method`/exit/`--policy-head`/`--shaping` flags on `sweep`.

📊 metrics: per-trial TensorBoard under
`pkm_data/new_agents/agent_000_dragapult/experiments/014_dragapult_sweep/runs/sweep-014_dragapult/trial_<N>/`
plus the Optuna study DB (study `014_dragapult`); best trial reported on the console.

```bash
args=(
  --trials 32                    # number of Optuna trials (2^5; each a short training run)
  --updates 16                   # updates PER TRIAL (2^4; short — retrain the winner at full length)
  --games 32                     # self-play games per update (2^5)
  --workers 8                    # rollout workers per trial (2^3)
  --eval-games 128               # games per eval — this IS the Optuna objective (2^7)
  --model large                  # net size for every trial. alt: small|medium|xl (+ per-dim overrides)
  --deck dragapult               # played deck for every trial
  --study 014_dragapult          # Optuna study name (SQLite) — resumable/inspectable
  --experiment 014_dragapult_sweep   # artifact dir
  --device auto                  # cpu here
  --seed 0                       # base seed (offset per trial)
  --engine local-nix
  # --objective win_rate         # what to maximize (default)
  # --tune-rewards               # also sample heuristic reward-term weights
  # --reset                      # start the study fresh instead of resuming it
)
python -m pkm.new_agents.agent_000_dragapult.cli sweep "${args[@]}"
```

---

## 015 — agent_000 · ALAKAZAM · ExIt with the COMBO head + MORE MCTS (large)

Run 010's ExIt recipe, but with the **new combination-scoring policy head**
(`--policy-head combo`, `model.ComboPolicyHead` — scores whole enumerated
option-*sets* in one pass) and a **doubled MCTS budget** (sims 16→32). Script:
`agent_000_dragapult/scripts/015_alakazam_exit_combo_large/train.sh`.

**Method caveat:** under `--method exit` the combo head trains via its
**marginalized `[B,L]`** (softmax of the combo distribution, marginalized to
per-option inclusion logits, trained by cross-entropy vs the MCTS visit-π). The
combo scorer gets full gradient, but the combination-level ranking (what
distinguishes this head from `marginal`) is only *indirectly* supervised. A
direct benchmark of the combination distribution needs `--method ppo` (mirrors
the 013 autoreg template). Chosen here because the ask was "more mcts", which
only applies to ExIt.

**Engine:** `--engine kaggle` (the vendored `local-nix` output was GC'd; the
kaggle `libcg.so` is ABI-identical and the backend is NOT in the config hash, so
a later `--resume` can switch to `local-nix`).

**Cost:** large (~2–3× medium/forward) × sims 32 (2× 010) ≈ 4–6× 010's per-game
cost. `--updates 8192` sets the cosine `T_max` but is a "let it cook" ceiling —
stop when the eval curve is good (`latest.pt`/`ckpt_N.pt` persist). `--workers
16` oversubscribes alongside the running 010 (8 workers) on this 23-core box.

📊 metrics: `pkm_data/new_agents/agent_000_dragapult/experiments/015_alakazam_exit_combo_large/logs/train.csv`
(TB: `…/experiments/015_alakazam_exit_combo_large/runs/015_alakazam_exit_combo_w4_large/`)

```bash
args=(
  --method exit --policy-head combo        # ← combo head is the new thing
  --exit-value-target tdlambda --exit-lambda 0.9 --determinization sample
  --model large --mcts-worlds 4 --mcts-simulations 32   # ← sims 16→32 = "more mcts"
  --mcts-c-puct 1.25 --mcts-temperature 1.0
  --deck alakazam --aux-weight prize_margin=0.25
  --shaping prize_potential --shaping-coef 1.0
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --value-coef 0.5 --minibatch-size 64
  --epochs 4 --gamma 0.997 --lam 0.95 --clip-eps 0.2 --entropy-coef 0.01 --seed 0
  --updates 8192 --games 16 --workers 16   # ← 16 workers (oversubscribes vs 010's 8)
  --device auto --eval-every 128 --eval-games 32 --ckpt-every 128
  --experiment 015_alakazam_exit_combo_large --run-name 015_alakazam_exit_combo_w4_large
  --tb --engine kaggle --force
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```
