# Agents

## Build & Run
```bash
uv sync                    # install deps
python main.py             # run a battle
./submit.sh                # create Kaggle submission bundle
```

## Lint & Typecheck
```bash
ruff check .               # lint
ruff format .              # format
pytest tests/              # run tests
```

## Project Structure
- `pkm/data/card_data.py` — card/attack metadata from cabt C library
- `pkm/data/deck.py` — Deck class (CSV load/save, 60-card validation)
- `pkm/agents/base.py` — `make_agent(deck, strategy_fn)` factory
- `pkm/agents/random_agent.py` — random legal move agent
- `pkm/agents/neural_agent.py` — greedy trained-policy agent (numpy inference, no torch)
- `pkm/search.py` — correct bindings to the engine's SearchBegin/SearchStep API
- `pkm/rl/` — encoders, pointer-style policy/value net, PPO self-play, expert iteration
- `pkm/mcts/` — determinization + IS-MCTS over the search API
- `pkm/strategies/` — future strategy implementations
- `main.py` — battle runner entry point
- `deck.csv` — sample deck (60 card IDs, one per line)
- `submit.sh` — creates `submission.tar.gz` for Kaggle
- `docs/RL_PLAN.md` — RL self-play design (Phase 1 PPO, Phase 2 IS-MCTS/ExIt)

## RL Training
Prefer the `justfile` (run `just` to list recipes): `just train` / `just resume`
(Phase 1 PPO), `just exit-train` / `just exit-resume` (Phase 2), `just export`,
`just play mcts neural`, `just eval mcts neural 30`, `just submit`.
Underlying commands:
```bash
python -m pkm.rl.train --iterations 50 --games 16 [--init checkpoints/ppo_latest.pt]
python -m pkm.rl.exit_train --iterations 5 --games 8    # Phase 2: expert iteration (init from ppo_latest.pt)
python -m pkm.rl.export checkpoints/ppo_latest.pt pkm/policy.npz  # export for torch-free inference
python -m pkm.rl.play --p0 mcts --p1 neural             # replay -> result.html + replay.json
```
- Checkpoints land in `checkpoints/`; `pkm/policy.npz` is bundled in the submission (no torch needed at inference).
- `pkm/search.py` signatures were recovered from the official competition `cg/api.py` (SearchBegin needs `lib.AgentStart()` handle + the observation's `search_begin_input`, returns ApiResult JSON; search ids are int64).

## cabt Engine API
- `from kaggle_environments.envs.cabt.cg.sim import lib` → `lib.AllCard()`, `lib.AllAttack()`
- `from kaggle_environments.envs.cabt.cg.game import battle_start, battle_select, battle_finish`
- Agents must be plain functions (not class instances) for kaggle-env compatibility
- `obs["select"] is None` → return deck (60 card IDs)
- Otherwise return list of option indices from `obs["select"]["option"]`

## Kaggle Submission
- Bundle: `tar -czvf submission.tar.gz main.py deck.csv pkm/`
- Max size: 197.7 MiB
- Daily limit: 5 submissions
- Only latest 2 are active
- Files land in `/kaggle_simulations/agent/`
