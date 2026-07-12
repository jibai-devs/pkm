# LEARNINGS.md

## Architecture Decisions (2026-07-05)

### Agent Profile Design
- Agent profiles are named directories under `agents/<name>/` that bundle deck + strategy + checkpoints + results
- Profile YAML stores: strategy type, deck path, creation time, notes
- Checkpoints named `<profile_name>_MM_DD_HH_MM_SS.pt` with `latest.pt` symlink
- Built-in agent names (`random`, `neural`, `mcts`) still work as legacy/defaults
- Play resolves agent name: check profiles first, fall back to built-in names
- Per-player deck support needed — currently both sides always share the same deck

### CLI Design
- Migrating from argparse to typer (already in pyproject.toml dependencies)
- Rich output for progress, tables, colored status
- Subcommands: `pkm play`, `pkm train`, `pkm exit-train`, `pkm export`, `pkm profile`
- Typer supports `--p0-weights` / `--p0-deck` style flags for overrides

## Environment & Config
- `typer>=0.26.8` already in pyproject.toml dependencies
- `rice>=0.4.0` in dependencies (terminal rendering library, not yet used much)
- Kaggle env needs `make("cabt", configuration={"decks": [deck, deck]})` — both sides get a deck list
- Agents must be plain functions `def agent(obs: dict) -> list[int]` for kaggle compatibility
- Kaggle may execute an agent with a working directory other than `/kaggle_simulations/agent`; bundled resources must be resolved relative to `__file__`, not only the process working directory
- Kaggle simulation submission runs `main.py` as `__main__`; a guarded local smoke-test call can execute instead of the callable `main(obs)` and must not load files absent from the bundle

## Codebase Conventions
- Agent factories: `make_<type>_agent(deck, **kwargs)` pattern
- `pkm/agents/base.py:make_agent(deck, strategy_fn)` is the base factory (closure-based)
- Checkpoints are `torch.save(model.state_dict(), path)` — state dicts only
- Metrics use `csv.DictWriter` with `writeheader()` + per-row `writerow()` + `flush()`
- Justfile is the primary task runner
