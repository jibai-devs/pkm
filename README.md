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
├── engine/           # the ONE engine seam (loader + typed api)
│   ├── loader.py     # backend switch, ctypes ABI, capabilities
│   └── api.py        # all 13 engine functions (typed)
└── policy.npz        # exported weights (bundled in submission)
engine/               # vendored C++ engine source (builds cg.so) — see below
deck/                 # deck files (CSV/JSON)
main.py               # Kaggle entry point
submit.sh             # builds submission.tar.gz
```

## Engine (cabt)

Every battle runs on the **cabt C engine** — a shared library (`libcg.so`) exposing
a 13-function C ABI. There are two builds of it, and you choose which one backs the
process. **You never have to compile anything to train or submit** — the default
uses the engine Kaggle already ships.

### The two backends

| | Kaggle build (default) | Vendored build (`PKM_ENGINE=vendored`) |
|---|---|---|
| Where it comes from | bundled in the `kaggle_environments` pip package | compiled from C++ source in `engine/` |
| Use it for | **everything by default**, and **all deployment/submission** | **local training only** (rebuild / instrument / speed) |
| Need to compile? | **No** | Yes (or `just engine-build`) |

Import the engine only through `pkm.engine` (never `kaggle_environments...cg.*`).
The backend is chosen at load time, precedence: `PKM_ENGINE_LIB=/abs/path` >
`PKM_ENGINE=vendored` > default `kaggle`. Inspect the active backend:

```bash
just engine-info                    # what's loaded + 13/13 ABI symbols present
PKM_ENGINE=vendored just engine-info
```

### What Kaggle actually gives you

**All 13 engine functions ship in Kaggle's own `libcg.so`** — including the search
API (`SearchBegin/Step/End/Release`) that MCTS needs. Kaggle's *Python* wrapper only
covers 6 of them (battle + visualize); we bind the other 7 ourselves in
`pkm/engine/api.py`. The practical upshot:

- **MCTS works at deployment on Kaggle's C engine** — we call their search symbols
  directly via ctypes. No vendored build is shipped or needed in the submission
  sandbox (which has no `engine/`).
- The vendored `cg.so` is a **local-training convenience only**.

(Full table of which functions Kaggle wraps vs. which we bind: `CLAUDE.md` →
"Engine functions". Determinism/parity details: `docs/ENGINE.md`.)

### Compiling the vendored engine (only if you want it)

```bash
just engine-build        # with nix: cmake+ninja in engine/'s flake devshell (libc++)
just engine-build-nix    # with nix, fully hermetic: nix build
just engine-build-cc     # WITHOUT nix: any system cmake + a C++20 compiler (libstdc++)
just engine-clean        # remove build outputs
just engine-parity       # assert the vendored initial-obs matches Kaggle's engine
```

`engine-build-cc` needs only `cmake` and a C++20 compiler (`g++`/`clang++`) — no
Nix. The single translation unit is memory-heavy; add `-j1` if the compile is
OOM-killed. Output lands at `engine/build/cg.so` (gitignored), which
`PKM_ENGINE=vendored` picks up automatically. **If you don't want to compile, do
nothing — the Kaggle build is the default.**

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
