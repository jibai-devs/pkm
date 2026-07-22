# new_agents — training command log

Every training command run for anything under `new_agents/` — **`train`,
`sweep`, and `resume`** — gets appended here, fully explicit, with a comment on
each flag and its alternatives. Keeps the run history in the repo (mirrors
`agent_000_dragapult/submission_log.md` for Kaggle bundles).

Commands are written as bash `args=(…)` arrays so **every flag carries an inline
comment** (what it does + other options) and the block still runs. Newest
entries at the bottom. Convention also recorded in the project `CLAUDE.md`.

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

Flag-value legend used below: `alt:` = other accepted values; `PPO-only` = read
only by `--method ppo` (inert under `exit`); `exit-only` = the reverse.

---

## 2026-07-22 — agent_000: cheaper CPU-first ExIt run (medium) — MASTER (all flags annotated)

CPU-friendly first pass of expert iteration with the new TD(λ) + W-world levers.
This block annotates **every** flag; later blocks only re-annotate what changes.

**Measured (this CPU box):** the smoke variant ran clean at **~7.0–7.8s/update
for 2 games (4 workers)** ≈ ~3.7s/game (val-loss 0.0200→0.0158). Extrapolated,
this 12-game/8-worker/100-update run ≈ **~20–40 min**. As expected for ExIt the
`ent/kl/clip/gnorm/evar` metrics log as `0.00` (PPO-only).

```bash
args=(
  # ---- training style + model + task ----
  --method exit                  # TRAINING STYLE. alt: ppo. exit = MCTS teaches every move (AlphaZero-ish)
  --model medium                 # net size preset. alt: small(=v1)|large|xl. (per-dim overrides: --n-layers/--d-state/--d-entity/--n-heads/--d-opt/--d-card)
  --policy-head marginal         # policy head. alt: autoreg (STOP-token multi-select). NOTE: exit only trains step-0 marginal, so autoreg's conditioning won't learn here — train autoreg under --method ppo
  --deck dragapult               # played 60-card list. alt: alakazam (+ any deck in deck.DECKS)
  --aux-weight prize_margin=0.25 # enable an aux head (repeatable: --aux-weight name=w). alt: omit = no aux. Training-only (stripped from Kaggle bundle)

  # ---- expert-iteration (MCTS) knobs — exit-only ----
  --exit-value-target tdlambda   # value target. alt: mc (raw ±1 outcome, v1). tdlambda = blend outcome + MCTS root value
  --exit-lambda 0.9              # tdlambda EMA factor (0..1). higher = trust outcome more. inert unless tdlambda
  --mcts-worlds 2                # determinized worlds averaged per decision (IS-MCTS). alt: 1 (single world, v1). cost ×W
  --mcts-simulations 16          # PUCT sims per decision. more = stronger teacher, linear cost. (large run uses 32)
  --mcts-c-puct 1.25             # PUCT exploration constant. higher = explore more
  --mcts-temperature 1.0         # visit-count temperature for π. 1.0 = ∝ visits; →0 = argmax
  --determinization sample       # how hidden zones are filled. key into determinize.DETERMINIZERS

  # ---- reward shaping — PPO-only (INERT under --method exit; kept explicit) ----
  --shaping prize_potential      # alt: terminal (sparse ±1) | dragapult_heuristic. exit ignores this (its target is outcome/tdlambda)
  --shaping-coef 1.0             # scale on the shaping term (0.0 == terminal). PPO-only
  # --reward-weight name=value   # (repeatable) override a heuristic term; only with --shaping dragapult_heuristic

  # ---- optimizer / learn step ----
  --lr 1e-4                      # Adam LR (or cosine start). 1e-4 = stable for deeper nets
  --lr-schedule cosine           # alt: constant. cosine anneals lr→lr-min over the run
  --lr-min 1e-5                  # cosine floor (eta_min). inert unless cosine
  --value-coef 0.5               # weight on the value (MSE) loss
  --minibatch-size 64            # decisions per optimizer minibatch
  --epochs 4                     # PPO-only: passes per update over the batch (inert for exit)
  --gamma 0.997                  # PPO-only: discount (long ~77-decision horizon)
  --lam 0.95                     # PPO-only: GAE lambda
  --clip-eps 0.2                 # PPO-only: PPO clip epsilon
  --entropy-coef 0.01            # PPO-only: entropy bonus
  --seed 0                       # RNG seed (offset per worker)

  # ---- run length + parallelism + device ----
  --updates 100                  # number of updates (train iterations)
  --games 12                     # self-play games collected per update
  --workers 8                    # parallel self-play workers (1 = single-process). exit cost is CPU-bound in workers
  --device auto                  # GPU/CPU. alt: cpu | cuda. auto = cuda if available else cpu. THIS box has no usable CUDA → cpu; --device cuda ERRORS here. (Only the learner update uses GPU; MCTS rollout is always CPU)

  # ---- eval / checkpoint / logging / identity ----
  --eval-every 20                # eval vs RANDOM every N updates (0 = never). NB: vs-random saturates → weak signal
  --eval-games 64                # games per eval
  --ckpt-every 20                # snapshot ckpt_N.pt every N updates (latest.pt always written)
  --experiment exit_tdlambda_worlds_medium  # artifact dir <output>/<experiment>/ (-e). config-hash guards resume
  --run-name exit_tdlambda_w2_medium         # TB subdir / wandb run name
  --tb                           # TensorBoard scalars to <experiment>/runs/. alt: --no-tb. (CSV at <experiment>/logs/train.csv is always written)
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
  --mcts-worlds 2 --mcts-simulations 16 --determinization sample
  --model medium --deck dragapult --aux-weight prize_margin=0.25
  --updates 3 --games 2 --workers 4        # tiny: just measure seconds/update
  --device auto --eval-every 0 --ckpt-every 100 --no-tb --force
  --output-dir /tmp/exit_smoke --experiment exit_smoke_measure
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

---

## 2026-07-22 — agent_000: full ExIt run (large) — the expensive one

Same as the master block but scaled up. **Deltas only** (all other flags as the
master block above):

```bash
args=(
  --method exit --exit-value-target tdlambda --exit-lambda 0.9 --determinization sample
  --model large                  # ↑ from medium: ~2–3× cost/forward
  --mcts-worlds 4                # ↑ from 2: ×2 searches/decision
  --mcts-simulations 32          # ↑ from 16: ×2 per search
  --deck dragapult --aux-weight prize_margin=0.25
  --mcts-c-puct 1.25 --mcts-temperature 1.0
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --value-coef 0.5 --minibatch-size 64 --seed 0
  --updates 200 --games 16 --workers 8      # ↑ longer run
  --device auto                  # cpu here → this run is ~tens of hours-to-days; use only with a GPU or lots of patience
  --eval-every 16 --eval-games 128 --ckpt-every 32
  --experiment exit_tdlambda_worlds_large --run-name exit_tdlambda_w4_large --tb
  --engine local-nix
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

---

## 2026-07-22 — agent_000: PPO run to train the autoregressive head (②)

The autoregressive STOP-token head is only *fully* trained under **PPO** (its
sampler + logprob use the conditioned path; under `exit` only the step-0 marginal
learns). Here the PPO-only knobs (gamma/lam/clip-eps/entropy-coef/epochs) are
LIVE, and the exit/MCTS knobs are absent.

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
  --epochs 4                     # PPO passes/update (LIVE)
  --minibatch-size 64 --seed 0

  --updates 400 --games 32 --workers 8       # PPO rollouts are cheap vs exit → more games/updates
  --device auto                  # cpu here (fine: PPO is much cheaper than exit)
  --eval-every 16 --eval-games 128 --ckpt-every 32
  --experiment autoreg_large --run-name autoreg_large_ppo --tb --engine local-nix
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

---

## Resume an interrupted run

`--resume` reloads `<experiment>/checkpoints/latest.pt` and continues. Pass the
SAME `--experiment` and repeat every **config-hash** flag (model dims,
`--policy-head`, `--deck`, `--method`, all mcts/exit knobs, `--shaping`,
`--aux-weight`) exactly, or the checkpoint won't load. Example (resumes the
medium ExIt run):

```bash
args=(
  --resume                       # continue from latest.pt instead of a fresh net
  --experiment exit_tdlambda_worlds_medium   # MUST match the original run's experiment
  # --- config-hash flags: MUST match the original exactly ---
  --method exit --exit-value-target tdlambda --exit-lambda 0.9
  --mcts-worlds 2 --mcts-simulations 16 --mcts-c-puct 1.25 --mcts-temperature 1.0
  --determinization sample --model medium --deck dragapult --aux-weight prize_margin=0.25
  --shaping prize_potential --shaping-coef 1.0
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --value-coef 0.5 --minibatch-size 64 --seed 0
  # --- run-length / runtime flags: free to change on resume ---
  --updates 100                  # counts from where it stopped
  --games 12 --workers 8 --device auto
  --eval-every 20 --eval-games 64 --ckpt-every 20
  --run-name exit_tdlambda_w2_medium --tb --engine local-nix
)
python -m pkm.new_agents.agent_000_dragapult.cli train "${args[@]}"
```

---

## Optuna hyperparameter sweep

`sweep` runs many short PPO trials (Optuna samples lr/entropy/clip/epochs/
minibatch/gamma/lam) and keeps the best by eval win-rate. It is PPO-oriented —
there are **no** `--method`/exit/`--policy-head`/`--shaping` flags on `sweep`.

```bash
args=(
  --trials 30                    # number of Optuna trials (each a short training run)
  --updates 15                   # updates PER TRIAL (kept short; retrain the winner at full length)
  --games 32                     # self-play games per update
  --workers 8                    # rollout workers per trial
  --eval-games 128               # games per eval — this IS the Optuna objective
  --model large                  # net size for every trial. alt: small|medium|xl (+ per-dim overrides)
  --deck dragapult               # played deck for every trial
  --study agent000_large_ppo     # Optuna study name (SQLite) — resumable/inspectable
  --experiment sweep_large       # artifact dir
  --device auto                  # cpu here
  --seed 0                       # base seed (offset per trial)
  --engine local-nix
  # --objective win_rate         # what to maximize (default)
  # --tune-rewards               # also sample heuristic reward-term weights
  # --reset                      # start the study fresh instead of resuming it
)
python -m pkm.new_agents.agent_000_dragapult.cli sweep "${args[@]}"
```
