# pkm — project instructions

Full project guide (structure, RL training, decks, submission): @AGENTS.md

## Active Context

- Kaggle submission deck lookup is working-directory independent: `main.py` checks paths relative to its own location.
- `tests/test_main.py` covers resolving bundled `deck.csv` when Kaggle runs from another directory.
- `main(obs)` is the Kaggle callable agent; `run_local_battle()` is separate for local smoke tests.

### General agent architecture (this branch)

- Plan: `docs/superpowers/plans/2026-07-12-general-agent-architecture.md`. Branch
  `feature/agent-training-facades`, cut from `feature/general-agent-architecture`.
- Tasks 1-3 are implemented, reviewed (spec + code quality), and committed. 91 tests pass, ruff clean.
- Trainers register in `pkm/agents/registry.py` (`TRAINERS`) alongside the policy/strategy
  tables; `AgentSpec` rejects an unknown policy, strategy, or trainer at profile load.
- Profile isolation is enforced by `AgentProfile._own_output_path()`: every caller-supplied
  output path (`checkpoint_dir`, `metrics_path`, `log_dir`) is realpath-checked, so one profile
  cannot write into another's `agents/<name>/` — symlink escapes included.
- PPO-only hyperparameters travel through `**hyperparams`, so a non-PPO trainer is never handed
  `gamma`/`pool_size`.
- `pkm export --agent X` honors an explicit checkpoint and takes `--phase ppo|exit`; without the
  phase flag a submission built after expert iteration would ship stale PPO weights.
- Next: Task 4 (`build_submit()` + generated Kaggle entry point). `submit.sh` still reads
  `deck/<agent>.csv` instead of the profile-owned deck — fix as part of Task 4.
- Follow-up: the `if agent:` profile branch is duplicated between `pkm/cli/__init__.py` and the
  `pkm/rl/train.py` / `exit_train.py` mains; worth collapsing.

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
