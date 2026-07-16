# Agents

**Reminder: Update this file whenever something significant changes** — new training results, new agents, architecture changes, submission status, etc. Stale docs are worse than no docs.

## Current Progress (as of Jul 2026)

| Phase | Status | Details |
|-------|--------|---------|
| Phase 1 — PPO self-play | **Done (200 iters)** | 80% win rate vs random. Agent: `00_basic` |
| Phase 2 — Expert iteration | **Started (1 run)** | MCTS self-play + distillation. Agent: `00_basic` |
| Agent profiles | **Done** | Per-agent directories for checkpoints, metrics, runs |
| Metrics & monitoring | **Done** | CSV logging + Plotly notebook |
| Kaggle submission | **Ready** | `just build_submit 00_basic` exports weights + bundles |

### What's Working
- Pointer/scoring policy network handles variable-length action spaces
- Submission `main.py` exposes the Kaggle agent protocol and defaults to the neural Dragapult policy
- PPO self-play with checkpoint pool opponent sampling
- Potential-based reward shaping (prize differential)
- IS-MCTS with determinization for imperfect information
- Expert iteration (MCTS targets -> network training)
- Numpy-only inference for Kaggle submission (no torch at eval time)
- CSV metric logging for all training runs

### What's Next
1. **Hyperparameter sweep** — LR, games/iter, pool size, eval frequency
2. **Longer exit training** — run 50+ iters of expert iteration from the PPO baseline
3. **MCTS vs neural eval** — measure if MCTS agent beats raw policy head-to-head
4. **Larger model** — wider MLP, more embedding dims, attention over options
5. **Multi-deck training** — sample opponent decks from a pool for robustness
6. **Submission** — `just build_submit` + `just upload` and check Kaggle leaderboard

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
- `pkm/engine/` — the single engine seam: `loader.py` (backend switch, ctypes ABI, capabilities), `api.py` (all 13 typed engine functions incl. SearchBegin/SearchStep)
- `pkm/rl/` — encoders, pointer-style policy/value net, PPO self-play, expert iteration
- `pkm/cli_deck.py` — deck management CLI (list, show, convert)
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
- `submit.sh` — creates `submission.tar.gz` for Kaggle
- `docs/RL_PLAN.md` — RL self-play design (Phase 1 PPO, Phase 2 IS-MCTS/ExIt)
- `replay/` — replay viewer + data
  - `replay/02_vite_web_app/` — Bun + Vite replay viewer (vanilla JS)
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
`PKM_ENGINE_LIB=/abs/path` > `PKM_ENGINE=vendored` > default `kaggle`. **Default must
stay `kaggle`** — the submission sandbox has no `engine/`. The switch covers the
direct engine paths (search, card data, RL/MCTS rollouts); `pkm/rl/play.py` and the
TUI still run matches through `kaggle_environments.make()`, which always uses the
bundled engine.

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

Holds `policy.npz` (numpy export), `ppo_latest.pt` for all three agents,
`00_basic/exit_latest.pt`, each agent's `deck.csv`, and the training metrics.
Checkpoints are gitignored, so this is the only durable copy — re-upload after a
training run that you want to keep:
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
- Bundle: `tar -czvf submission.tar.gz main.py deck.csv pkm/`
- Max size: 197.7 MiB
- Daily limit: 5 submissions
- Only latest 2 are active
- Files land in `/kaggle_simulations/agent/`
