# pkm — project instructions

Full project guide (structure, RL training, decks, submission): @AGENTS.md

## Active Context

- Kaggle submission deck lookup is working-directory independent: `main.py` checks paths relative to its own location.
- `tests/test_main.py` covers resolving bundled `deck.csv` when Kaggle runs from another directory.
- `main(obs)` is the Kaggle callable agent; `run_local_battle()` is separate for local smoke tests.
- Agent architecture idea: `docs/ideas/general-agent-architecture.md`.
- Implementation plan: `docs/superpowers/plans/2026-07-12-general-agent-architecture.md`.
- Implementation is in worktree `/home/df/.config/superpowers/worktrees/pkm_new/general-agent-architecture` on branch `feature/general-agent-architecture`.
- Completed there: profile-owned decks/config/checkpoints and policy factory/profile play integration. Latest commit: `c68a4b8`.
- Latest worktree verification: 67 tests passed; final Task 2 review must be rerun after the latest packaging fix.
- Next: implement `AgentProfile.train()`, `train_exit()`, and `build_submit()` with per-profile weights before multi-agent play/opponent-pool work.

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
