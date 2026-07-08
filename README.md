# pkm — Pokémon TCG AI Battle Challenge

RL agent for the [Kaggle Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge). Trains via PPO self-play and AlphaZero-style expert iteration, submits to Kaggle with numpy-only inference (no torch at eval time).

## Quick Start

```bash
uv sync                          # install deps
pkm deck list                    # list available decks
pkm play --p0 random --p1 random # watch a random match
```

## CLI

All commands go through the `pkm` entry point:

```
pkm deck list|show|convert       # deck management
pkm cards dump <out.json>        # dump card database
pkm train [OPTIONS]              # Phase 1: PPO self-play
pkm exit-train [OPTIONS]         # Phase 2: expert iteration (MCTS)
pkm export [OPTIONS] [OUT]       # checkpoint → .npz for Kaggle
pkm play [OPTIONS]               # play/evaluate matches
```

Run `pkm <command> --help` for full option lists.

### Decks

```bash
pkm deck list                          # show all decks
pkm deck show 02_dragapult             # deck contents
pkm deck convert 01_psychic --to json  # CSV ↔ JSON
```

### Training

```bash
# Phase 1 — PPO self-play (default deck: 02_dragapult)
pkm train --agent 00_basic --iterations 200 --games 16

# Resume from latest checkpoint
pkm train --agent 00_basic --iterations 200 --init checkpoints/ppo_latest.pt

# Phase 2 — Expert iteration (MCTS self-play + distillation)
pkm exit-train --agent 00_basic --iterations 20 --games 8 --sims 32
```

Use `--agent <name>` to auto-resolve deck paths, checkpoint dirs, and metrics under `agents/<name>/`.

### Evaluation

```bash
# Single match with HTML replay
pkm play --p0 neural --p1 random --agent 00_basic

# MCTS vs neural, 30-game evaluation
pkm play --p0 mcts --p1 neural --agent 00_basic --games 30

# Open the last replay
xdg-open result.html
```

Agent types: `random` | `neural` (greedy policy from .npz) | `mcts` (IS-MCTS with neural prior).

### Export & Submit

```bash
# Export checkpoint to numpy .npz (torch-free inference)
pkm export --agent 00_basic pkm/policy.npz

# Build Kaggle submission bundle
just submit
```

## Just Recipes

The `justfile` wraps the CLI with sensible defaults:

| Recipe | Description |
|---|---|
| `just train [agent]` | Phase 1 PPO (default: 200 iters, 16 games) |
| `just resume [agent]` | Resume Phase 1 |
| `just exit-train [agent]` | Phase 2 expert iteration |
| `just exit-resume [agent]` | Resume Phase 2 |
| `just export [agent]` | Export weights to .npz |
| `just play [p0] [p1] [agent]` | Single match + replay |
| `just eval [p0] [p1] [games] [agent]` | Win-rate over N games |
| `just submit` | Export + build submission.tar.gz |
| `just deck` | List decks |
| `just test` | Run pytest |
| `just lint` | Ruff check + format |

## Project Structure

```
pkm/
├── cli/              # CLI entry points (Typer)
│   ├── __init__.py   # main `pkm` app — registers all subcommands
│   ├── deck.py       # pkm deck
│   └── cards.py      # pkm cards
├── rl/               # reinforcement learning
│   ├── train.py      # Phase 1: PPO self-play
│   ├── exit_train.py # Phase 2: expert iteration
│   ├── export.py     # checkpoint → .npz
│   ├── play.py       # match runner + evaluation
│   ├── model.py      # PolicyValueNet
│   ├── ppo.py        # PPO update
│   ├── rollout.py    # game rollout
│   ├── encoder.py    # state/action encoding
│   └── numpy_policy.py  # torch-free inference
├── mcts/             # IS-MCTS with determinization
├── agents/           # agent factories + profiles
├── data/             # deck/card data loading
└── policy.npz        # exported weights (bundled in submission)
deck/                 # deck files (CSV/JSON)
main.py               # Kaggle entry point
submit.sh             # builds submission.tar.gz
```

## Kaggle Submission

- Bundle: `tar -czvf submission.tar.gz main.py deck.csv pkm/`
- Max size: 197.7 MiB
- Daily limit: 5 submissions (only latest 2 active)
- Files land in `/kaggle_simulations/agent/`

## Monitoring

```bash
# TensorBoard (live dashboards)
tensorboard --logdir=runs

# CSV metrics
cat metrics/ppo_train.csv
cat metrics/exit_train.csv
```
