# pkm — project instructions

Full project guide (structure, RL training, decks, submission): @AGENTS.md

## Active Context

- **Multi-deck support shipped (2026-07-21):** agent_000 now plays **more than one
  deck** without a per-deck agent. Two concepts split cleanly: (a) *a deck* — a
  60-card list in the `deck.DECKS` registry (`dragapult` + new `alakazam` = Mega
  Alakazam/Dudunsparce psychic control); (b) *the vocabulary* — the **superset**
  of distinct card IDs over ALL registered decks (now 44 → `VOCAB_SIZE=45`, was
  27), sizing the own-card embedding + hand-histogram. One trained net has learned
  rows for every deck's cards; unused rows are inert (gather, not softmax — zero
  gradient, no learning-difficulty cost). Adding a *new* deck with new cards grows
  the vocab → forces a retrain; reusing known cards is free. Cards resolved by
  name+text vs `replay/cards.json` (source-list set/collector numbers do NOT map to
  engine IDs). `--deck {dragapult,alakazam}` on `train`/`eval` picks the played
  list; it's in `RunConfig.deck` + the config hash (old ckpts backfill to
  `dragapult`). `determinize.py`/`mcts.search` are deck-aware (unblocks future
  cross-deck/mixed self-play). Idempotent `scripts/gen_vocab.py` writes/inspects
  `vocab.json` (drift-guarded by `test_deck_vocab.py`); `test_deck_routing.py`
  locks routing. **Also fixed a pre-existing bug:** `eval` built the MCTS
  `InferenceConfig` but never passed it to the winrate fns, so `eval --inference
  mcts -K` silently ran the plain policy. **Not yet done:** train the alakazam
  deck (`train --deck alakazam …`, generic `prize_potential` shaping) — a fresh
  PPO run; the old 600 ckpt is incompatible (VOCAB_SIZE changed) and retired.
- **agent_000 status (2026-07-20):** stuck at Kaggle **600** across small/large nets —
  it's an **eval ceiling**, not capacity: training is mirror self-play but the metric
  was vs *random* (saturates ~100%). Fixes shipped: head-to-head eval
  (`eval --opponent <ckpt>`), configurable net size/depth (`--model`), GPU
  (`--device cuda`), heuristic shaping. Large net trained unstable (lr too high for
  depth) → `002_large_tuned` (lr 1e-4). **Inference-time MCTS shipped (2026-07-20):**
  wrap the trained net in PUCT search at decision time, no retrain — `pack
  --inference mcts -K <sims>` (K=0 = plain policy) bakes it into the bundle;
  `eval --inference mcts -K <sims>` measures it; `scripts/pack_variants.sh` packs
  both variants. Search rides Kaggle's own `libcg.so`. Next lever: **opponent-pool
  training**. Full log: `pkm/new_agents/agent_000_dragapult/TRAINING.md` §10.
- **Auxiliary losses shipped (2026-07-20) + NEW DEFAULT recipe.** Config-driven
  registry `aux_tasks.py` (mirrors the reward-term registry): each `AuxTask`
  bundles a head factory + per-step labeller + loss; `TrainConfig.aux_weights`
  (name→weight, in the config hash) turns tasks on (weight>0), default all-zero
  = off = v1 behaviour. Enable via `train --aux-weight prize_margin=0.25`
  (repeatable). Heads are **training-only**: built only when a full `Config`
  reaches `build_model`, so inference (bare `ModelConfig`) gets none; `pack`
  strips `aux_heads.*` from the bundle → zero inference/parity/size cost. First
  task: `prize_margin` (predict final prize-count margin, dense −6..+6, terminal
  label). **The new default training = large net + tuned low-LR PPO + heuristic
  rewards + `prize_margin` aux**, packaged as `scripts/003_aux_loss/train.sh`.
  Rationale + menu of future aux tasks (Tier B opponent-belief heads are the real
  ceiling lever): `pkm/new_agents/agent_000_dragapult/docs/00_aux_loss.md`.
- **Submission logging convention:** every time we submit a bundle to Kaggle,
  append a row to `pkm/new_agents/agent_000_dragapult/submission_log.md` (date,
  checkpoint, inference mode/K, bundle filename, message, and the score once it
  lands) plus a short note on what the run was testing. Keep it up to date so the
  leaderboard history stays in the repo.
- **Training/sweep workflow — ALWAYS use a persistent tmux (updated 2026-07-22):**
  EVERY `new_agents` run that trains — **`train`, `sweep`, AND `resume`** — must be
  launched into a tmux session (NOT a foreground/background shell job), so it
  **survives detach and stays alive after the command completes** (shell returns
  to its prompt, scrollback retained for inspection). Pattern:
  `tmux new-session -d -s pkm-train` (once) → `tmux send-keys -t pkm-train "cd
  <repo> && python -m pkm.new_agents.…cli train …" Enter` → watch with
  `tmux attach -t pkm-train` (detach `Ctrl-b d`). One predictable place; long runs
  don't die on disconnect. (Supersedes the earlier "use background Bash jobs, not
  send-keys" memory — the user's standing convention is the tmux.) Per-experiment
  run/sweep scripts live under
  `pkm/new_agents/agent_000_dragapult/scripts/<NNN_name>/` (e.g. `001_complexity_large/`);
  log the exact command in `pkm/new_agents/train_cmd_log.md`.
- **Network size is configurable:** `train`/`sweep` take `--model {small,medium,large,xl}`
  (small = v1, checkpoint-compatible) plus per-dim overrides `--n-layers/--d-state/
  --d-entity/--n-heads/--d-opt/--d-card`. Dims are in the config hash + every checkpoint.
- **Heuristics-integration architecture (Tasks 1-8) merged with the reward-shaping
  heuristics** from `refactor-to-prepare-for-heuristics-integration`, on
  `feature/heuristics-integration` (commits `73356e5`, `dc1e157`, `83265b5`).
  Full architecture write-up: `docs/ARCHITECTURE.md`. Training/export/Kaggle-
  submission runbook: `docs/TRAINING_AND_SUBMISSION.md`. New agent
  `03_pult_munki` (Dragapult ex/Munkidori, **no Dusknoir** — the deck the
  merged reward terms actually target, e.g. Xerosic's Machinations) has a real
  1000-iter PPO checkpoint (eval-vs-random plateaus 85-100% from ~iter 250,
  local only — not yet backed up to HF). Retrain-and-measure ablations for
  Tasks 6/7/8 and Phase 2 expert iteration for this agent are both still
  outstanding — see `AGENTS.md` → "What's Next".
- **Critical, bit us once already:** `pkm/` must stay importable under Python
  **3.11** (Kaggle's actual sandbox runtime) — this repo's own dev env is 3.12
  via `uv`, so PEP 695 generic syntax (`def f[T](...)`, `type X = ...`) passes
  every local check and then hard-`SyntaxError`s on Kaggle with zero local
  signal. `pkm/types/obs.py` had exactly this bug from its first commit
  (every submission before 2026-07-18 would have failed); fixed in commit
  `83265b5`. Use `typing.TypeVar` instead, always. Full details:
  `AGENTS.md` → "Kaggle Submission".
- Kaggle CLI auth (`~/.kaggle/kaggle.json`) is currently 401ing on this
  machine — needs a fresh token (kaggle.com → Account → API → Create New
  Token) before `kaggle competitions submit`/`logs`/`submissions` work again;
  the website upload flow is unaffected.
- Vendored C++ engine in `engine/` (from `ptcg` @ `0a56d34`) builds `engine/build/cg.so`,
  ABI-identical to Kaggle's `libcg.so`. Swap via `PKM_ENGINE=vendored` (default `kaggle`);
  the seam is `pkm/engine/` and all engine imports go through it. `just engine-build` /
  `just engine-parity`. Engine is **nondeterministic** (`random_device` seed, no injection),
  so only initial-obs parity is testable. Full details in AGENTS.md → "Vendored engine".
  Typed API consolidated in `pkm/engine/api.py` (commit `5390696`); 63 tests pass on both backends.
- Human TUI battle shipped on `feature/human-tui-battle`: `just play human neural`.
  Code in `pkm/tui/` (session/labels/widgets/app), typed obs in `pkm/types/obs.py`.
- `select.type` / `select.context` are **0-based on the wire** (the tables in
  `obs_data_structure/OBSERVATION_SCHEMA.md` are 1-based); `OptionType` / `LogType`
  are NOT offset. `example_obs.json` is hand-written and wrong — use
  `tests/fixtures/observations.json` (captured from the live engine).
- Human play must disarm kaggle's cumulative 600s overage clock + `runTimeout`
  (`actTimeout`/`runTimeout` = `1e9`), or the player loses on time.
- kaggle inspects `agent.__code__.co_argcount` — a **bound method** counts `self`
  and gets called with 2 args. Agent callables must be plain functions/lambdas.

- Kaggle submission deck lookup is working-directory independent: `main.py` checks paths relative to its own location.
- `tests/test_main.py` covers resolving bundled `deck.csv` when Kaggle runs from another directory.
- `main(obs)` is the Kaggle callable agent; `run_local_battle()` is separate for local smoke tests.
- RL techniques & improvement ideas: `docs/ideas/rl-improvements.md` (experience replay, offline RL, replay log utilization, interactive training, priority-ranked improvement list).
- Decision & training pipeline reference: `docs/ideas/training-and-decision-pipeline.md` (full decision flow, GAE delta, PPO gradient, modularity analysis).
- Optimizer reference: `docs/ideas/optimizers.md` (SGD, Adagrad, Adam, AdamW explained; why Adam for RL).
- Visualization & HPO tools: `docs/ideas/visualization-and-hpo.md` (TensorBoard, wandb, Optuna, Netron, plotly).
- Agent architecture idea: `docs/ideas/general-agent-architecture.md`.
- Implementation plan: `docs/superpowers/plans/2026-07-12-general-agent-architecture.md`.
- Implementation is in worktree `/home/df/.config/superpowers/worktrees/pkm_new/general-agent-architecture` on branch `feature/general-agent-architecture`.
- Completed there: profile-owned decks/config/checkpoints and policy factory/profile play integration. Latest commit: `c68a4b8`.
- Latest worktree verification: 67 tests passed; final Task 2 review must be rerun after the latest packaging fix.
- Next: implement `AgentProfile.train()`, `train_exit()`, and `build_submit()` with per-profile weights before multi-agent play/opponent-pool work.
- **Training-command log convention (2026-07-22):** for `new_agents`, EVERY
  training command — `train`, `sweep`, AND `resume` — gets appended to
  `pkm/new_agents/train_cmd_log.md`. Each entry is the full command in a ```bash
  source block, a short note on what/why the key values were chosen, AND a direct
  path to where that run's **metrics** land so they can be inspected without
  hunting:
  `pkm_data/new_agents/agent_000_dragapult/experiments/<experiment>/logs/train.csv`
  (per-update CSV), plus `…/runs/<run-name>/` (TensorBoard) and `…/checkpoints/`.
  **Two style rules:** (1) prefer **powers of 2** for count/size params where
  sensible — games, workers, updates, minibatch, mcts sims/worlds, eval/ckpt
  intervals, trials (NOT rates/coeffs like lr/gamma/lambda/clip); (2) prefix each
  `--experiment` name with the next **zero-padded incrementing number** (…009 →
  `010`, `011`, `012` …, continuing the existing `experiments/NNN_*` series).
  Keep it current so the run history lives in the repo (mirrors the
  `submission_log.md` convention for Kaggle bundles).

## Engine functions: kaggle lib vs vendored (IMPORTANT)

**Every one of the 13 C functions ships in Kaggle's `libcg.so` binary.** Nothing in
our API is "missing" from Kaggle — the search API and card data are real exported
symbols in the shipped lib. What differs is that Kaggle's *Python package* only
**wraps 6 of them** (`cg/sim.py` + `cg/game.py`); the other 7 are unwrapped C
symbols we bind ourselves in `pkm/engine/api.py` (recovered from the official
competition `cg/api.py`).

| Function | In Kaggle `libcg.so` (C symbol) | Wrapped by Kaggle Python | Bound in our `api.py` |
|---|:--:|:--:|:--:|
| `GameInitialize` | ✅ | ✅ | ✅ |
| `BattleStart` | ✅ | ✅ | ✅ |
| `BattleFinish` | ✅ | ✅ | ✅ |
| `GetBattleData` | ✅ | ✅ | ✅ |
| `Select` | ✅ | ✅ | ✅ |
| `VisualizeData` | ✅ | ✅ | ✅ |
| `AgentStart` | ✅ | ❌ | ✅ |
| `SearchBegin` | ✅ | ❌ | ✅ |
| `SearchStep` | ✅ | ❌ | ✅ |
| `SearchEnd` | ✅ | ❌ | ✅ |
| `SearchRelease` | ✅ | ❌ | ✅ |
| `AllCard` | ✅ | ❌ | ✅ |
| `AllAttack` | ✅ | ❌ | ✅ |

**Consequences for how we use each backend:**

- **Deployment (Kaggle submission) → always the Kaggle C lib.** The default backend
  is `kaggle` and the submission sandbox has **no** `engine/`. Because the search
  symbols (`AgentStart`, `SearchBegin/Step/End/Release`) live in Kaggle's own
  `libcg.so`, **MCTS works at deployment by calling Kaggle's C implementation** —
  we just bind those symbols via ctypes. We do **not** ship or need our vendored
  build to run MCTS in the sandbox.
- **Local training → optionally the vendored build.** `PKM_ENGINE=vendored` uses our
  own `engine/build/cg.so`, purely a local convenience (rebuild / instrument / speed).
  It is **never** part of a submission. Default must stay `kaggle`.
- Net: the vendored `cg.so` is a *local-training-only* artifact; the search API for
  MCTS rides on Kaggle's C implementation everywhere it matters.

See `docs/ENGINE.md` and `README.md` for build/compile instructions.

## Replay viewer

Step-by-step viewer for match replays. The maintained one is React/TS at
`replay/05_vite_react_app/`.

```bash
just play                            # generate a replay.json + result.html
just replay-react                    # view it at http://localhost:5175
just replay-react file=/foo.json     # view a different replay
```

Load a different replay three ways (precedence: picker > `?replay=` > `VITE_REPLAY`
> default): the in-app **"Load replay…"** button / drag-drop (any local `.json`),
a `?replay=/foo.json` URL param, or the `VITE_REPLAY` env var. `?step=N`
deep-links a step. Controls: **Space** play/pause, **←/→** step, **Home/End**
first/last, scrubber to jump.

**Full usage:** `docs/REPLAY_VIEWER.md` · **data contract & code layout:**
`replay/05_vite_react_app/README.md`.

## Browser play (React GUI vs bot)

`replay/07_vite_react_cards` at `?mode=play` is a live game GUI (not just a
replay viewer): pick opponent + deck, then click options to play a real match
against a bot with real card art. `just play-web-build` builds + serves UI and
API at `:8000`; for hot-reload dev run `just play-web` (Python bridge) +
`just play-web-dev` (Vite) in two terminals. The engine side reuses the TUI's
`ThreadedEnvSession` through a stdlib `http.server` long-poll bridge
(`pkm/web/server.py`) — a blocking `GET /api/event` *is* `next_event`, a `POST
/api/submit` *is* `submit`; no new Python deps. Option labels are rendered
server-side via `pkm/tui/labels.option_label`; play-mode React code is in
`replay/07_vite_react_cards/src/live/`. Full write-up: `AGENTS.md` → "Human Play
(Browser / React GUI)".
