# Agents

**Reminder: Update this file whenever something significant changes** — new training results, new agents, architecture changes, submission status, etc. Stale docs are worse than no docs.

## Current Progress (as of 2026-07-19)

| Phase | Status | Details |
|-------|--------|---------|
| Phase 1 — PPO self-play | **Done (200 iters)** | 80% win rate vs random. Agent: `00_basic` |
| Phase 1 — PPO self-play | **Done (1000 iters)** | Eval-vs-random plateaus 85-100% from ~iter 250. Agent: `03_pult_munki` |
| Phase 2 — Expert iteration | **Started (1 run)** | MCTS self-play + distillation. Agent: `00_basic`. **Not yet run for `03_pult_munki`.** |
| Agent profiles | **Done** | Per-agent directories for checkpoints, metrics, runs. Agents: `00_basic`, `01_psychic`, `02_dragapult`, `03_pult_munki` |
| Heuristics integration | **Done (Tasks 1-8)** | `GameContext`/`DeckTracker`, `FeatureSpec` registry, Tier-1 deterministic features, pooled deck-ledger, opponent-archetype auxiliary head. Task 9 (hard-rule forced picks) deliberately not built. Full write-up: `docs/ARCHITECTURE.md` |
| Reward-shaping merge | **Done** | Deck-specific (Dreepy/Drakloak/Dragapult ex/Budew/Xerosic) reward terms merged from `refactor-to-prepare-for-heuristics-integration`, off by default (`pkm/rl/reward_terms.py`) |
| Opponent-archetype classifier | **Done (Parts 1-2)** | Synthetic-data supervised classifier over 25 `staples.json` archetypes, wired into the encoder (belief feature) + MCTS determinization, both opt-in. Full plan/status: `docs/opponent-archetype-classifier-plan.md`. **Known breakage:** the belief-feature resize invalidated existing policy checkpoints, pending the full retrain Part 3 sets up for. |
| Opponent pool decklists (Part 3a) | **Done (25/25)** | Real, legal 60-card decklist per `staples.json` archetype, scraped from limitlesstcg tournament lists via `pkm/archetype/scrape_decklist.py` and converted with `pkm/archetype/build_pool_deck.py` → `deck/pool_<id>_<slug>.csv`. |
| Pool bots (Part 3b, solo PPO) | **Done (25/25)** | Fallback path (plan §3b/3c "as first written", not simultaneous population training): each `deck/pool_*.csv` trained solo via unmodified `pkm/rl/train.py`, 100 iters/16 games/agent, `agents/pool_<id>_<slug>/`. Eval-vs-random 65-100% across all 25 (worst: `pool_361_metagross_metal_maker` 65%, `pool_362_mega_starmie_ex` 70%). **Confirmed via export attempt (2026-07-19):** the known belief-feature-resize breakage is concrete — `03_pult_munki`'s committed checkpoint is stamped `opponent_archetype_belief` dim=4 (old dormant 3-class+Other trunk head), current registry expects dim=26 (25 real archetypes + Other from the live Part 1 classifier); that checkpoint cannot be exported/used until the full retrain. Part 3c (below) closes the "not yet started" gap this line used to flag. |
| Cross-archetype sampling (Part 3c) | **Done (implementation, 2026-07-19)** | `GameSpec.opponent_deck` (`pkm/rl/rollout.py`) lets a pooled opponent bring its own deck instead of mirroring the trainee's; `pkm/rl/opponent_pool.py:load_pool_bots()` loads all trained `agents/pool_*/` checkpoints; `pkm train --archetype-pool [--archetype-pool-prob 0.2]` opt-in, default off (backward compatible). **Caveat found while wiring it up:** `pkm/cli/__init__.py`'s `train` command is a hand-duplicated shim over `pkm/rl/train.py`'s own typer app (not a passthrough) — any new flag must be added to *both* or it silently no-ops from the actual `pkm train` entry point; hit and fixed this once already for the new flags. Smoke-tested end-to-end against real pool-bot checkpoints (5 iters, throwaway agent). |
| Belief-in-encoder wired into training (Part 2a → train.py) | **Done (2026-07-19)** | Part 2a's `NumpyArchetypeClassifier`/`compute_belief` were only ever exercised at the `TorchPolicy`-unit level (`tests/test_archetype_integration.py`) — `pkm/rl/train.py` never actually constructed a classifier, so every training run to date (including the original `03_pult_munki` 1000-iter run) saw an all-zero belief feature regardless of the dim-4→26 resize. Now: `pkm train --archetype-belief [--archetype-weights pkm/archetype.npz]` loads the classifier once and attaches it to the trainee's `TorchPolicy` only (never the frozen opponent's) via `play_one`/`parallel_rollout.py` (plumbing verified by `tests/test_rl.py::test_play_one_classifier_reaches_trainee_not_opponent`, a monkeypatch spy test). Smoke-tested standalone, combined with `--archetype-pool`, and under `--workers 2` (confirms the classifier — plain numpy arrays — pickles fine across worker processes). Opt-in, off by default. |
| `03_pult_munki` full retrain (Milestone 8) | **Done (2026-07-19)** | Fresh run (old checkpoint incompatible, preserved at `agents/03_pult_munki/checkpoints_pre_belief_resize/`): `pkm train --agent 03_pult_munki --iterations 2000 --games 16 --eval-every 10 --archetype-pool --archetype-pool-prob 0.4 --archetype-belief`. `--archetype-pool-prob` bumped from the 0.2 default and iterations doubled from the original run's 1000 — both user calls, not ablation-derived. Finished all 2000 iterations, 100% eval-vs-random. This checkpoint became Milestone 9's population-training starting point (below) — superseded by that run's iter-2370 checkpoint, but still the reference point for the Verification #5 ablation. |
| Pool bots, 26th addition | **Done (2026-07-19)** | `pool_400_mega_abomasnow_ex` added alongside the original 25: real decklist scraped from a user-supplied limitlesstcg tournament-list URL, solo-trained the same way as Part 3b (100 iters/16 games, 95% eval-vs-random). **Deliberately pool-bot-only, not added to `staples.json`** — no aggregate presence-percentage stats page exists for this archetype (low-representation, ~39% overall win rate on limitlesstcg), and adding it to the classifier would have bumped `BELIEF_DIM` (25→26 archetypes), which is baked into *every* model's input size as a GLOBAL feature (`pkm/rl/features.py:261-262`) — would have invalidated all 26 existing checkpoints simultaneously, including the Milestone 8 retrain that had just finished. Full-archetype/classifier integration for this deck remains a separate, not-yet-done follow-up. |
| Population training (Milestone 9) | **Run (partial), 2026-07-19/20** | `pkm population-train --iterations 3000 --games-per-pairing 2 --workers 8`, anchor `03_pult_munki` vs all 26 pool bots (25 + Mega Abomasnow). Stopped by user request at iteration ~2375/3000 (not a crash). All 27 members' `ppo_latest.pt` confirmed current as of iteration 2370's checkpoint cycle (verified: present, uniform size, valid stamp sidecars). **Snapshot eval** (`pkm eval-vs-pool --agent 03_pult_munki --games 20`, belief-off, since this predates the Finding-1 fix below): 60.1% average win rate across the 26 pool bots; best `pool_320_n_s_zoroark_ex` (100%), worst `pool_400_mega_abomasnow_ex` (5%). **Corrected (belief-on, matching production) re-run: 62.7% overall, 35% vs `pool_400_mega_abomasnow_ex` — see "Abomasnow Matchup Investigation" below, both findings now root-caused, Finding 1 fixed.** This is the raw material for the plan's Verification #5 ablation (population training vs. Milestone 8's frozen-pool design) once the same eval is run against the Milestone 8 checkpoint. Full plan/status: `docs/opponent-archetype-classifier-plan.md` §3b+3c. |
| Metrics & monitoring | **Done** | CSV logging + Plotly notebook |
| Kaggle submission | **Working, verified** | `pkm export --agent <name> pkm/policy.npz` + `bash submit.sh <name>` — see `docs/TRAINING_AND_SUBMISSION.md` for the full runbook |

### What's Working
- Pointer/scoring policy network handles variable-length action spaces
- Submission `main.py` exposes the Kaggle agent protocol; deck-agnostic (bundled `deck.csv` decides which deck plays, `submit.sh` no longer hardcodes `02_dragapult`)
- PPO self-play with checkpoint pool opponent sampling
- Potential-based reward shaping (prize differential) + a registry of deck-specific reward terms (`pkm/rl/reward_terms.py`), each off by default
- Per-game memory (`GameContext`/`DeckTracker`) tracking own-deck card locations and deducing prize-pile contents by elimination
- Declarative `FeatureSpec` registry (`pkm/rl/features.py`) — GLOBAL/PER_SLOT/PER_OPTION float features, ablation via `FeatureConfig`, checkpoint stamping to reject registry/weights mismatches
- Tier-1 deterministic heuristics (`lethal_this_turn`, `type_effectiveness`, `retreat_viable`)
- Pooled deck-ledger feature family (unseen-card-count-weighted sum through the network's own card embedding table, not a flat slot vector)
- Detached opponent-archetype auxiliary head, re-injected as a belief feature one decision later
- IS-MCTS with determinization for imperfect information; reads `GameContext` once at the real root only
- Expert iteration (MCTS targets -> network training)
- Numpy-only inference for Kaggle submission (no torch at eval time); parity with the torch model is a standing CI gate (`tests/test_numpy_torch_parity.py`)
- CSV metric logging for all training runs

### What's Next
1. **Retrain-and-measure ablations still outstanding** — Tasks 6/7/8's win-rate/accuracy lift claims (Tier-1 features, deck ledger, archetype head) have never actually been measured on/vs-ablated. `FeatureConfig` supports this; nobody's run it yet.
2. **Task 9 decision** — hard-rule extension to `forced_picks` was deliberately skipped (every candidate condition considered was either already covered or a policy judgment call in disguise). See `docs/superpowers/plans/2026-07-16-heuristics-integration-architecture.md`.
3. **Phase 2 for `03_pult_munki`** — run expert iteration on top of its 1000-iter PPO checkpoint; never measured whether MCTS actually beats the raw policy head-to-head for this deck.
4. **Kaggle CLI auth is broken on this machine** — `kaggle competitions submit`/`logs`/`submissions` all 401. `~/.kaggle/kaggle.json` needs a fresh token (kaggle.com → Account → API → Create New Token) before the CLI can be used again; the website upload flow still works.
5. **Hyperparameter sweep** — LR, games/iter, pool size, eval frequency
6. **Larger model** — wider MLP, more embedding dims, attention over options
7. **Multi-deck training** — all 26 real opponent decklists are sourced (25 from Part 3a + `pool_400_mega_abomasnow_ex`), 26 solo pool bots are trained, cross-archetype sampling is implemented (Part 3c, `pkm train --archetype-pool`), the classifier's belief is wired into training (`--archetype-belief`), Milestone 8's frozen-pool retrain finished (2000 iters), and population training has now actually run (Milestone 9, partial — 2375/3000 iterations, stopped by user; see Current Progress above). **Next:** resume/extend the population-training run if desired, then run the plan's Verification #5 ablation properly (`pkm eval-vs-pool` against both the Milestone 8 checkpoint and the population-trained one, apples-to-apples) before flipping any defaults (Milestone 10). ~~Root-cause the `pool_400_mega_abomasnow_ex` 5%-win-rate outlier~~ — done, see "Abomasnow Matchup Investigation" below: the number itself was an artifact of `eval-vs-pool`'s methodology, and the real (smaller, ~45%) gap traces to a concrete feature bug. See `docs/opponent-archetype-classifier-plan.md` Part 3.
8. **Bulk-upload the population-trained roster to Hugging Face** — all 27 checkpoints (anchor + 26 pool bots) exist locally only; see "Population-Trained Bot Roster" below for the upload loop and analysis workflow. Not yet done.
9. ~~Fix the variable-damage blind spot in `lethal_this_turn`/`attack_damage`~~ — **Phase 1 done 2026-07-20** (`pkm/rl/attack_damage_estimator.py`, 14 patterns, 170 tests passing). Hammer-lanche's own deck-mill family is Phase 2, not yet done; retraining on the changed feature is Phase 3, also not done. See "Abomasnow Matchup Investigation" below and `docs/superpowers/plans/2026-07-20-attack-damage-estimator.md`.
10. ~~Re-measure `eval-vs-pool` with a wired-in archetype classifier~~ — done 2026-07-20, now defaults on (`--no-archetype-belief` for the old baseline). See "Abomasnow Matchup Investigation" below.
11. **Decide whether `population_train.py` should compute live belief during rollout** — Milestone 8 trained `03_pult_munki` with real `--archetype-belief` values; Milestone 9's `population_train.py` never attaches a classifier, so all 2375 population-training iterations saw belief≡0, likely eroding whatever Milestone 8 learned about that feature before deployment (which does feed it live values). Scoped (Phase 2/3) in `docs/superpowers/plans/2026-07-20-belief-classifier-routing.md`; not yet implemented.

## Build & Run
```bash
uv sync                    # install deps
python main.py             # run a battle
./submit.sh                # create Kaggle submission bundle
pkm deck list                # list decks
```

## Lint & Typecheck
```bash
ruff check .               # lint
ruff format .              # format
pytest tests/              # run tests
```

## Project Structure
- `pkm/data/card_data.py` — card/attack metadata from cabt C library
- `pkm/data/deck.py` — Deck class (CSV/JSON load/save, 60-card validation)
- `pkm/agents/base.py` — `make_agent(deck, strategy_fn)` factory
- `pkm/agents/random_agent.py` — random legal move agent
- `pkm/agents/neural_agent.py` — greedy trained-policy agent (numpy inference, no torch)
- `pkm/agents/singaporean_middleman.py` — decision-routing agent; dispatches each
  turn to sub-agents (heuristics/neural/random). Current Kaggle submission agent
- `pkm/heuristics/` — hand-written strategy helpers (`deck_tracker.py`: deck/prize tracking)
- `pkm/engine/` — the single engine seam: `loader.py` (backend switch, ctypes ABI, capabilities), `api.py` (all 13 typed engine functions incl. SearchBegin/SearchStep)
- `pkm/rl/` — encoders, pointer-style policy/value net, PPO self-play, expert iteration
  - `pkm/rl/features.py` — declarative `FeatureSpec` registry (GLOBAL/PER_SLOT/PER_OPTION), checkpoint stamping
  - `pkm/rl/deterministic_features.py` — Tier-1 heuristics (`lethal_this_turn`, `type_effectiveness`, `retreat_viable`)
  - `pkm/rl/reward_terms.py` — reward-shaping term registry (`POTENTIAL_TERMS`/`DIRECT_TERMS`/`DEFAULT_WEIGHTS`), per-agent weights JSON
  - `pkm/rl/parallel_rollout.py` — `ProcessPoolExecutor` self-play (`pkm train --workers N`)
- `pkm/heuristics/` — `GameContext` (per-game memory) + `DeckTracker` (own-deck card-location tracking, prize deduction)
- `pkm/archetype/` — opponent-archetype classifier (Parts 1-2) + real pool-deck sourcing (Part 3a)
  - `pkm/archetype/archetypes.py`, `pkm/archetype/aliases.py` — `staples.json` name → engine `card_id` resolution (exact match, then hand-maintained alias table for collisions)
  - `pkm/archetype/gen.py`, `pkm/archetype/numpy_model.py` — synthetic training-data generator + numpy-forward classifier twin (Kaggle, no torch)
  - `pkm/archetype/build_pool_deck.py` — converts a real `entries.json` decklist into a legal `deck/pool_<id>_<slug>.csv`; best-effort multi-match resolution (unlike the classifier's strict resolution)
  - `pkm/archetype/scrape_decklist.py` — scrapes a `limitlesstcg.com/decks/list/<id>` tournament decklist page straight into the same pipeline (`python -m pkm.archetype.scrape_decklist <url> <archetype_id> <slug>`)
  - `docs/opponent-archetype-classifier-plan.md` — full plan + status (Parts 1/2/3)
- `pkm/cli_deck.py` — deck management CLI (list, show, convert)
- `docs/ARCHITECTURE.md` — full technical walkthrough of the heuristics-integration architecture (GameContext/registry/trunk/heads/reward-shaping/MCTS boundary)
- `docs/TRAINING_AND_SUBMISSION.md` — runbook: start/background/monitor a training run, export, back up to HF, build + upload a Kaggle submission
- `docs/ideas/` — architecture ideas and future design notes
  - `docs/ideas/agent-composition-and-refactor.md` — code map (net/policy/value/MCTS/training), composition modes (pipeline vs injection vs delegation), and the ranked refactor plan
- `pkm/mcts/` — determinization + IS-MCTS over the search API
- `pkm/strategies/` — future strategy implementations
- `pkm/types/obs.py` — pydantic models for the observation (typed contract for the TUI + RL encoder)
- `pkm/tui/` — Textual human-vs-agent battle UI (`session`, `labels`, `widgets`, `app`)
- `main.py` — battle runner entry point
- `deck/` — deck files (CSV: one card ID per line; JSON: id/name/count)
- `deck/00_basic.csv` — starter 60-card deck
- `deck/01_psychic.csv` — Psychic Toolbox (Slowking + Mega Kangaskhan ex)
- `deck/02_dragapult.csv` — **default deck**: Dragapult ex / Dusknoir (Psychic/Dark)
- `deck/03_pult_munki.csv` — Dragapult ex / Munkidori, **no Dusknoir**, carries Xerosic's Machinations — the deck the merged reward-shaping terms actually target
- `deck/pool_<id>_<slug>.csv` — 26 real, legal 60-card decklists: 25 from `staples.json` archetypes (Part 3a) plus `pool_400_mega_abomasnow_ex` (pool-bot-only, not in `staples.json` — see Current Progress). Opponent pool for population training (Milestone 9, `pkm/rl/population_train.py`)
- `submit.sh` — creates a Kaggle submission bundle for any agent (`bash submit.sh <agent>`); no longer hardcoded to `02_dragapult`; validates `deck/<agent>.csv` exists first
- `docs/RL_PLAN.md` — RL self-play design (Phase 1 PPO, Phase 2 IS-MCTS/ExIt)
- `replay/` — replay viewer + data
  - `replay/05_vite_react_app/` — Bun + Vite + React/TS replay viewer
  - `replay/replay.json` — sample replay log
  - `replay/cards.json` — card database with attack metadata
  - `replay/requirements.md` — viewer requirements & data format reference
  - `replay/ideas_and_recommendations.md` — approach options & design notes

## RL Training
Prefer the `justfile` (run `just` to list recipes): `just train` / `just resume`
(Phase 1 PPO), `just exit-train` / `just exit-resume` (Phase 2), `just export`,
`just play mcts neural`, `just eval mcts neural 30`, `just build_submit`, `just upload`.
Underlying commands:
```bash
python -m pkm.rl.train --iterations 50 --games 16 [--init checkpoints/ppo_latest.pt]
python -m pkm.rl.exit_train --iterations 5 --games 8    # Phase 2: expert iteration (init from ppo_latest.pt)
python -m pkm.rl.export checkpoints/ppo_latest.pt pkm/policy.npz  # export for torch-free inference
python -m pkm.rl.play --p0 mcts --p1 neural             # replay -> result.html + replay.json
```
- Checkpoints land in `checkpoints/`; `pkm/policy.npz` is bundled in the submission (no torch needed at inference).
- `pkm/engine/api.py` search signatures were recovered from the official competition `cg/api.py` (SearchBegin needs `lib.AgentStart()` handle + the observation's `search_begin_input`, returns ApiResult JSON; search ids are int64).

## Replay Viewer
```bash
just replay                          # start vanilla-JS viewer (Bun + Vite)
just replay-react                    # start React/TS viewer on :5175
just replay-react file=/foo.json     # ...loading a different replay
```
The React viewer (`replay/05_vite_react_app/`, the maintained one) can also load a
replay via a `?replay=/foo.json` URL param or an in-app file picker / drag-drop
(any local `.json`). **Full usage: `docs/REPLAY_VIEWER.md`.**

## Deck Management
```bash
just deck                           # list available decks
just deck-show 01_psychic           # show deck contents with card names
just deck-convert 01_psychic json   # convert CSV -> JSON format
```
Use `--deck` to specify a non-default deck for training or play:
```bash
just play neural random deck/01_psychic.csv
just eval neural random 30 deck/01_psychic.csv
just train 200 16 deck/01_psychic.csv
```

## Metrics & Monitoring
Training logs are saved to CSV during training:
- `metrics/ppo_train.csv` — PPO self-play (iter, wins, losses, pi_loss, v_loss, entropy, clip_frac, eval_win_rate)
- `metrics/exit_train.csv` — expert iteration (iter, pi_loss, v_loss)

TensorBoard (live dashboards, run comparison):
```bash
tensorboard --logdir=runs          # opens http://localhost:6006
# Logs: runs/ppo/ (PPO), runs/exit/ (expert iteration)
# Compare runs: python -m pkm.rl.train --log-dir runs/lr_1e-3
```

Plotly notebook (interactive charts):
```bash
jupyter notebook notebooks/training_monitor.ipynb
```

## Human Play (TUI)
Play against a trained agent yourself, in the terminal:
```bash
just play human neural             # you vs the neural agent (both 02_dragapult)
just play human random             # you vs random
just play neural human 01_psychic  # you as player 2
```
`1`-`9` toggle options, `Enter` confirms (attacks and end-turn ask twice — there is
no undo), `q` quits. Input is inert while the agent thinks. The match writes
`result.html` + `replay.json` like any other, so you can rewatch a hand-played game
in the React replay viewer.

Implementation: `pkm/tui/` (Textual), typed observations in `pkm/types/obs.py`.
Two things that are easy to get wrong here, both verified by measurement:
- **Human play must disarm kaggle's timeouts** (`actTimeout`/`runTimeout` = `1e9`).
  Otherwise a *cumulative* 600s "overage clock" ticks down while you think, and the
  first move that overdraws it gives you status `TIMEOUT` and a **loss**.
- **Never `print()` in TUI code.** kaggle wraps each agent call in `redirect_stdout`,
  which is active process-wide while the human agent blocks; prints vanish into its
  buffer. Use `textual.log`.

Design + rationale: `docs/superpowers/specs/2026-07-13-human-tui-battle-design.md`.

## Custom Agents
Agents are plain functions with signature `def agent(obs: dict) -> list[int]`.
To add your own agent:
1. Create `pkm/agents/your_agent.py` with a `make_your_agent(deck, **kwargs)` factory
2. Add a branch in `pkm/rl/play.py:make_agent_by_name()`
3. Run: `just play your_agent neural` or `just eval your_agent neural 30`

The `make_agent(deck, strategy_fn)` base factory in `pkm/agents/base.py` handles deck submission boilerplate — your strategy_fn only needs to handle `obs["select"] is not None`.

## cabt Engine API
- Import the engine through **`pkm.engine`** (the single seam), not directly from
  `kaggle_environments.envs.cabt.cg.*`. `pkm/engine/api.py` collates **all 13**
  engine functions (kaggle's package only wrapped 6): `battle_start/select/finish`,
  `visualize_data`, `search_begin/step/end/release`, `all_cards/all_attacks`, plus
  `to_observation`. All argtypes live in one place (`loader._configure_argtypes`).
  `from pkm.engine import battle_start, battle_select, search_step, all_cards`
- **Return-type convention** (matches `pkm/types/obs.py`'s "dict at the seam,
  pydantic inward" design): `battle_*` return raw obs **dicts** (37 call sites read
  them as dicts; rollouts stay cheap — validate with `to_observation()` at the ML
  boundary). `search_*` return a typed **`SearchState`** whose `.observation`
  validates lazily + caches (one consumer, `pkm/mcts`, so typed with no hot-loop
  cost). card/attack data come back as `list[dict]`.
- Agents must be plain functions (not class instances) for kaggle-env compatibility
- `obs["select"] is None` → return deck (60 card IDs)
- Otherwise return list of option indices from `obs["select"]["option"]`

## Vendored engine (`engine/`) + backend switch
The C++ engine source lives in `engine/` (copied from the standalone `ptcg` repo,
commit `0a56d34`, Competition-Use-Only license). It compiles to `engine/build/cg.so`,
which is ABI-identical (same 13 exported symbols) to the Kaggle-bundled `libcg.so`.

`pkm/engine/loader.py` picks which build backs the process, precedence:
`PKM_ENGINE_LIB=/abs/path` > `PKM_ENGINE=local-nix` (nix build) / `local` (cmake) /
`vendored` (deprecated alias: nix-then-cmake) > default `kaggle`. **Default must
stay `kaggle`** — the submission sandbox has no `engine/`. The switch covers the
direct engine paths (search, card data, RL/MCTS rollouts); `pkm/rl/play.py` and the
TUI still run matches through `kaggle_environments.make()`, which always uses the
bundled engine. The engine loads **lazily on first use** (`loader.get_lib()`), so
the backend can be picked at runtime (`loader.set_backend(...)` / the agent CLIs'
`--engine` flag) and non-engine commands skip the load; only `kaggle` pulls in
`kaggle_environments`, so `local*` starts fast.

```bash
just engine-build         # nix devshell: cmake+ninja (libc++) -> engine/build/cg.so
just engine-build-nix     # hermetic: nix build -> engine/result/lib/cg.so
just engine-build-cc      # NO nix: system cmake + C++20 compiler (libstdc++)
just engine-info          # print backend + capability report (respects PKM_ENGINE)
just engine-parity        # assert vendored initial-obs matches the official engine
PKM_ENGINE=vendored just test        # run the suite against the vendored engine
```

`pkm.engine` also exposes capability detection — `capabilities()`,
`available_backends()`, `kaggle_available()`, `vendored_built()` — so code can
adapt to which backends are present and whether a (future, patched) seeded ABI
exists.

**The engine is nondeterministic by design.** `ApiBattleStart` (`engine/src/api/Api.h`)
seeds `std::mt19937` from `std::random_device()` with no seed injection through the
public `BattleStart(int*)` ABI — even the official lib diverges from itself after the
first `Select`. So full-game byte-parity is impossible; `test_engine_parity.py` only
asserts the deterministic **initial observation** matches. Seed-exact reproducibility
would require patching the C++ to accept an injected seed. Two gotchas that do *not*
break rules-parity but do change draw order: `std::shuffle` is implementation-defined
(libc++ vs libstdc++ differ), and the vendored `flake.nix` builds with libc++ while
the official lib links libstdc++.

**Full write-up (build with/without nix, backend swap, capability detection, the
determinism issue and our compromise): `docs/ENGINE.md`.**

## Weights on Hugging Face
Published (public): **https://huggingface.co/TomatoCream/pkm-cabt-ppo**

Holds `policy.npz` (numpy export), `ppo_latest.pt` for the original three
agents, `00_basic/exit_latest.pt`, each agent's `deck.csv`, and the training
metrics. **`03_pult_munki`'s 1000-iter PPO checkpoint (2026-07-18) is not yet
uploaded here** — it only exists locally in `agents/03_pult_munki/checkpoints/`
(gitignored). Checkpoints are gitignored in general, so HF is the only
durable copy — re-upload after a training run that you want to keep:
```bash
hf upload TomatoCream/pkm-cabt-ppo <local_path> <path_in_repo> --repo-type model
hf download TomatoCream/pkm-cabt-ppo policy.npz                 # fetch back
```
Needs a **write** token (`hf auth login`); a read token 403s on upload.

The per-iteration `ppo_iter*.pt` snapshots are **not** kept — they were deleted
locally (268 MB) and are not on HF. Only `*_latest` checkpoints survive.

Note: the repo has no LICENSE, so the HF model card omits a license field —
public but "all rights reserved" by default.

## Population-Trained Bot Roster (Milestone 9 run, 2026-07-19/20)
27 independently-trained checkpoints from the population-training run above:
the anchor `03_pult_munki` plus all 26 `agents/pool_*/` bots. All checkpoints
are gitignored (local-only) — nothing here is backed up yet.

**Locating the weights:**
- `agents/03_pult_munki/checkpoints/ppo_latest.pt` — the anchor
- `agents/pool_<id>_<slug>/checkpoints/ppo_latest.pt` — each pool bot (one
  per `deck/pool_<id>_<slug>.csv`)
- Per-member training curves: `agents/<name>/metrics/population_train.csv`
  (kept separate from `ppo_train.csv` so Part 3b's original solo-training
  history isn't overwritten)
- `AgentProfile` resolves every name to `deck/<name>.csv` 1:1
  (`pkm/agents/profile.py`) — this holds for the anchor and every pool bot,
  no exceptions, which is what the upload loop below relies on.

**Analyzing bot play:**
```bash
# aggregate per-archetype win rate for any profile against the whole pool
# (torch checkpoints directly, no export needed) -- already run for the
# anchor above, 62.7% overall (belief-on default as of 2026-07-20; pass
# --no-archetype-belief for the old zero-belief baseline)
pkm eval-vs-pool --agent 03_pult_munki --games 20 [--pool-glob "pool_*"]

# a single watchable replay between two specific bots, each on its own deck.
# pkm play's p0/p1 shared one deck+weights pair before 2026-07-20; --p0-agent/
# --p1-agent (pkm/rl/play.py) resolve each side from its own AgentProfile
# instead. Needs each side's checkpoint exported to .npz first (pkm play's
# `neural` agent type is numpy-inference only, same as Kaggle):
pkm export agents/03_pult_munki/checkpoints/ppo_latest.pt agents/03_pult_munki/checkpoints/policy.npz
pkm export agents/pool_400_mega_abomasnow_ex/checkpoints/ppo_latest.pt agents/pool_400_mega_abomasnow_ex/checkpoints/policy.npz
pkm play --p0 neural --p1 neural \
  --p0-agent 03_pult_munki --p1-agent pool_400_mega_abomasnow_ex \
  --html result.html --replay replay.json
just replay-react file=/replay.json    # step through it at localhost:5175
```
Verified end-to-end 2026-07-20 (real exported weights, real cross-deck game,
loadable replay.json). `--p0-agent`/`--p1-agent` fall back to `--agent`/
`--deck`/`--weights` when omitted, so every pre-existing `pkm play` call is
unaffected; the CLI shim in `pkm/cli/__init__.py` was updated in the same
change (see the Part 3c "hand-duplicated shim" gotcha above — still applies
to every new `pkm play` flag, not just `pkm train`'s).

**Bulk-uploading all 27 to Hugging Face** (not yet done — do before any
future cleanup of `agents/`, since checkpoints are gitignored and HF is the
only durable copy, same rationale as the single-agent section above):
```bash
for d in agents/03_pult_munki agents/pool_*; do
  name=$(basename "$d")
  hf upload TomatoCream/pkm-cabt-ppo "$d/checkpoints/ppo_latest.pt" "population_2026-07-20/$name/ppo_latest.pt" --repo-type model
  hf upload TomatoCream/pkm-cabt-ppo "$d/checkpoints/ppo_latest.pt.stamp.json" "population_2026-07-20/$name/ppo_latest.pt.stamp.json" --repo-type model
  hf upload TomatoCream/pkm-cabt-ppo "deck/$name.csv" "population_2026-07-20/$name/deck.csv" --repo-type model
  hf upload TomatoCream/pkm-cabt-ppo "$d/metrics/population_train.csv" "population_2026-07-20/$name/population_train.csv" --repo-type model
done
```
Needs a **write** token (`hf auth login`), same requirement as the section
above. Uploads raw torch checkpoints, not `.npz` exports — export per-bot
only if/when a specific one is actually going to be played with `pkm play`
or bundled for Kaggle.

## Abomasnow Matchup Investigation (2026-07-20)
Started from the "5%, worst matchup" line in the Milestone 9 snapshot eval
(above): `03_pult_munki` vs `pool_400_mega_abomasnow_ex`. Two distinct findings
came out of it — a methodology bug in how that 5% number was measured, and a
real (much smaller) matchup weakness with a concrete root cause. Neither fix
has been applied yet; this is diagnostic only.

**Finding 1 — the 5% figure doesn't reproduce; `eval-vs-pool` measured a
different input distribution than deployment sees. FIXED 2026-07-20.** Ran 20
games through the actual production path instead (`pkm play` with each side's
exported `policy.npz`, alternating who goes first — script:
`scripts/run_matchup_replays.py`): **9/20 (45%)**, not 5%. Root cause:
- `pkm/rl/eval_vs_pool.py` built `TorchPolicy` directly with no archetype
  classifier attached, so the `opponent_archetype_belief` GLOBAL feature was
  always zero for every game it played.
- `pkm/agents/neural_agent.py:make_neural_agent` — what `pkm play`, the above
  script, and **the real Kaggle submission** all use — auto-loads
  `pkm/archetype.npz` by default (it's present in the repo root and gets
  bundled by `submit.sh`) and computes a live, non-zero belief every decision.

Fixed by wiring a `NumpyArchetypeClassifier` into both sides of every
`eval-vs-pool` game (see
`docs/superpowers/plans/2026-07-20-belief-classifier-routing.md` Phase 1 for
the full design/rationale — belief now defaults **on** here, unlike
`pkm train`'s opt-in `--archetype-belief`, specifically because this tool
exists to measure what a checkpoint actually does and production always
computes live belief; `--no-archetype-belief` reproduces the old baseline).
Re-ran the full 26-bot eval after the fix: **62.7% overall** (was 60.1%),
**35% vs `pool_400_mega_abomasnow_ex` specifically** (was 5%; in the same
ballpark as the 45% measured via real `pkm play` games above — different
random sample, same conclusion). The old 5% was a measurement artifact, not
a real result. Phase 2 (wiring the same classifier into
`pkm/rl/population_train.py`'s actual training loop, which has the same gap)
is scoped in that plan file but not yet done.

**Finding 2 — the real, smaller gap traces to a variable-damage blind spot in
two Tier-1 features. Phase 1 FIXED 2026-07-20.** Decoded the 20 replays' structured `logs` (script:
`scripts/analyze_matchup_replays.py`; a player's `observation.logs` is a
since-last-decision buffer, only fresh at that player's `ACTIVE` steps —
concatenating both players' ACTIVE-step logs and de-duping the union
reconstructs the full public event history without the double-counting the
raw per-step buffers would otherwise cause). Across the 20 games, `03_pult_munki`
("A") lost 73 Pokémon total vs. `pool_400_mega_abomasnow_ex` ("B") losing 31 —
A's *whole* fragile support/tech line dies repeatedly (Budew, Dreepy, Drakloak,
Munkidori, Meowth ex, Moltres, Fezandipiti ex), not just its main attacker.

The mechanism: B's signature attack, **Hammer-lanche** (Mega Abomasnow ex,
`attackId 1046`), was the single most-used move in every game (67 uses across
20 games) and the most common finishing blow. Its card data declares
`damage: 0` — the real damage is computed from card text at runtime
("discard the top 6 cards of your deck; 100 damage per Basic {W} Energy
discarded that way", i.e. 0–600 damage, genuinely random). Two features read
the static `damage` field directly and are blind to this:
- `pkm/rl/deterministic_features.py:lethal_this_turn` (line ~40):
  `opp_active.hp - atk.damage <= 0` — for Hammer-lanche `atk.damage` is
  always `0`, so this **never** flags it as lethal no matter how much real
  damage is about to land.
- `pkm/rl/features.py:_attack_damage` (line ~364): same static field, fed to
  the network as the per-option "how hard does this attack hit" input —
  Hammer-lanche always looks like a 0-damage move to the trained policy.

This is a **general limitation**, not an Abomasnow-specific one: any attack
whose real damage is computed from card text rather than a flat `damage`
field (energy-mill, coin-flip, prize-count-based, etc.) defeats both
features the same way. Abomasnow just leans on this attack shape harder than
anything else currently in the pool.

Fixed (Phase 1 of `docs/superpowers/plans/2026-07-20-attack-damage-estimator.md`)
by a new `pkm/rl/attack_damage_estimator.py`: 14 regex patterns over real
card text, verified against all 1556 attacks in the card database (0
exceptions), covering everything computable from the observation alone --
coin flips (expected value), fixed damage-counter placement, energy attached
to the attacker, discard-pile energy counts, benched-Pokémon count, prizes
taken, teammate-has-a-named-attack, opponent hand size, Pokémon Tool count,
and a general fixed "this attack does N damage to ..." constant (the single
biggest coverage win). `_attack_damage` and `lethal_this_turn` both wired in;
the latter uses a separate `min_guaranteed_damage` that excludes the 3
coin-flip patterns, so an attack whose *expected* damage clears the KO
threshold isn't claimed as a guaranteed kill. **Hammer-lanche itself is
still not fixed** — it's a deck-mill effect (damage depends on the
attacker's unrevealed remaining deck order), scoped as Phase 2 in the plan
doc, not yet implemented. Retraining on the changed feature (Phase 3) is
also not done — this only changes what the feature *reports*; the current
checkpoint hasn't learned to respond to the new values yet.

**Artifacts** (not committed, useful for repeating/extending this):
- `scripts/run_matchup_replays.py <agent_a> <agent_b> --games N` — plays N
  games between two agent profiles via their exported `policy.npz`,
  alternating who goes first, saving a full JSON replay per game plus a
  `summary.json` to `runs/<agent_a>_vs_<agent_b>/`.
- `scripts/analyze_matchup_replays.py <run_dir>` — decodes that directory's
  replays into per-game attack/KO/energy-attach event lists plus cross-game
  aggregates (attack usage by side, KO counts by side, finishing-move
  frequency). KO detection keys off `MoveCard` events with
  `fromArea=Active(4), toArea=Trash(3)` (a `Change`/type-9 log event is
  evolution mid-slot replacement, *not* a KO — that assumption was wrong on
  the first pass and silently produced zero KOs everywhere until fixed).
  Note: the printed "last hit" damage per KO is keyed by `cardId` (species),
  not `(cardId, serial)` (specific physical copy) — with multiple copies of
  the same species in play this can attribute a KO to the wrong copy's last
  hit; the qualitative KO-count-by-side finding above doesn't depend on this,
  but don't trust the per-KO damage numbers without fixing that key first.
- `runs/03_pult_munki_vs_pool_400_mega_abomasnow_ex/` — the 20 replays +
  summary from this run.

## Kaggle Submission
- Full runbook (train → keep running → export → back up → build → upload):
  **`docs/TRAINING_AND_SUBMISSION.md`**.
- Bundle: `pkm export --agent <name> pkm/policy.npz && bash submit.sh <name>`
  (`submit.sh <agent>` — the tar's `-C submission .` intentionally flattens
  paths to `./main.py`/`./deck.csv`/`./pkm/...`, not `submission/...`, because
  Kaggle extracts straight into `/kaggle_simulations/agent/` and expects
  `main.py` at that top level.)
- Max size: 197.7 MiB
- Daily limit: 5 submissions
- Only latest 2 are active
- Files land in `/kaggle_simulations/agent/`

### Critical: the sandbox runs Python 3.11, this repo's dev env runs 3.12
Kaggle's submission sandbox executes `main.py` (and everything it imports
under `pkm/`) with **Python 3.11**. This project's own dev environment (`uv
sync`, `just test`, everything in this doc) runs **3.12** — so a Python
3.12-only construct will pass every local check and then hard-`SyntaxError`
the instant Kaggle tries to import it, with **zero local signal**.

This already happened once: `pkm/types/obs.py` used PEP 695 generic-function
syntax (`def _as_enum[E: IntEnum](...)`) from its very first commit, so
**every submission ever built from this codebase failed identically** until
it was caught (2026-07-18, via a failed validation episode's traceback) and
fixed by rewriting it with a plain `typing.TypeVar` instead. Full context:
`docs/ARCHITECTURE.md`, commit `83265b5`.

**Do not use, anywhere under `pkm/`:**
- `def f[T](...)` / `class C[T]:` (PEP 695 generic syntax) — use
  `T = TypeVar("T")` instead.
- `type Alias = ...` (PEP 695 type-alias statement) — use
  `Alias: TypeAlias = ...` or a plain assignment instead.

If you're unsure whether something's 3.11-safe, the cheapest real check on
this machine is running the bare system `python` (3.11, no `uv run`) against
the specific import chain `main.py` pulls in — it reproduces the sandbox's
parse behavior exactly, which `uv run`'s 3.12 venv does not.

### Critical (fixed 2026-07-19): `submit.sh` didn't bundle `staples.json`
Same severity class as the PEP 695 bug above — **every submission built via
`submit.sh` since the opponent-archetype classifier landed (`a836ebd`) failed
identically on Kaggle**, regardless of agent or deck, caught the same way
(a failed validation episode's traceback, saved as `error.txt`).

Root cause: `pkm/rl/features.py` computes `NUM_TRACKED_ARCHETYPES =
len(get_archetypes())` at **module import time** (not lazily), and
`get_archetypes()` reads `staples.json` from `pkm/archetype/archetypes.py`'s
`STAPLES_JSON_PATH = Path(__file__).resolve().parents[2] / "staples.json"`
— two directories above itself, correctly resolving to the sandbox root
(`/kaggle_simulations/agent/`) on Kaggle, same as repo root in dev. The path
logic was always right; `submit.sh` just never copied `staples.json` into
the bundle. Since `pkm.rl.features` is imported transitively by nearly
everything (`pkm.mcts.search` → `pkm.rl.encoder` → `pkm.rl.features`), this
broke *importing `pkm` at all* — not just the archetype classifier
specifically — with `FileNotFoundError:
'/kaggle_simulations/agent/staples.json'`.

Fixed by adding `cp staples.json submission/` to `submit.sh`. Verified for
real (not just by reasoning about it): extracted a rebuilt bundle to an
isolated scratch directory, confirmed `pkm.archetype.archetypes.__file__`
resolved to the *extracted* copy (not the dev repo, which would have given
a false pass), and confirmed the exact import chain from the traceback now
succeeds with `NUM_TRACKED_ARCHETYPES == 25`.

**Any submission tar built before this fix is broken and should be
discarded, not uploaded** — that includes `submissions/submission_03_pult_munki_20260718_223836.tar.gz`
(pre-dates the fix).
