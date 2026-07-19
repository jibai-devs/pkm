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
| Pool bots (Part 3b, solo PPO) | **Done (25/25)** | Fallback path (plan §3b/3c "as first written", not simultaneous population training): each `deck/pool_*.csv` trained solo via unmodified `pkm/rl/train.py`, 100 iters/16 games/agent, `agents/pool_<id>_<slug>/`. Eval-vs-random 65-100% across all 25 (worst: `pool_361_metagross_metal_maker` 65%, `pool_362_mega_starmie_ex` 70%). **Confirmed via export attempt (2026-07-19):** the known belief-feature-resize breakage is concrete — `03_pult_munki`'s committed checkpoint is stamped `opponent_archetype_belief` dim=4 (old dormant 3-class+Other trunk head), current registry expects dim=26 (25 real archetypes + Other from the live Part 1 classifier); that checkpoint cannot be exported/used until the full retrain. **Not yet started:** Part 3c (`GameSpec` deck-field extension + frozen-pool cross-archetype sampling in `train.py`) to actually retrain `03_pult_munki` against this pool. |
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
7. **Multi-deck training** — all 25 real opponent decklists are sourced (Part 3a) and 25 solo pool bots are trained (Part 3b, fallback path, 65-100% eval-vs-random). Next: Part 3c — extend `GameSpec` (`pkm/rl/rollout.py`) with an opponent-deck field and `train.py`'s checkpoint-pool sampling so games can draw a pool bot + its matching deck, then retrain `03_pult_munki` against this frozen pool (closes the belief-feature-resize breakage above). Simultaneous population training (`pkm/rl/population_train.py`, `PopulationMember`/`PopSpec`) remains a documented fallback-of-the-fallback if solo-trained bots prove too weak — see `docs/opponent-archetype-classifier-plan.md` Part 3.

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
- `deck/pool_<id>_<slug>.csv` — 25 real, legal 60-card decklists, one per `staples.json` archetype (Part 3a, done); opponent pool for the not-yet-built population training (Part 3b/3c)
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
