# pkm — project instructions

Full project guide (structure, RL training, decks, submission): @AGENTS.md

## Active Context

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
- Agent architecture idea: `docs/ideas/general-agent-architecture.md`.
- Implementation plan: `docs/superpowers/plans/2026-07-12-general-agent-architecture.md`.
- Implementation is in worktree `/home/df/.config/superpowers/worktrees/pkm_new/general-agent-architecture` on branch `feature/general-agent-architecture`.
- Completed there: profile-owned decks/config/checkpoints and policy factory/profile play integration. Latest commit: `c68a4b8`.
- Latest worktree verification: 67 tests passed; final Task 2 review must be rerun after the latest packaging fix.
- Next: implement `AgentProfile.train()`, `train_exit()`, and `build_submit()` with per-profile weights before multi-agent play/opponent-pool work.

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
