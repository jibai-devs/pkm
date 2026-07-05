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
- `pkm/strategies/` — future strategy implementations
- `main.py` — battle runner entry point
- `deck.csv` — sample deck (60 card IDs, one per line)
- `submit.sh` — creates `submission.tar.gz` for Kaggle

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
