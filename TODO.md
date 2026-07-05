# TODO

## Architecture: Agent Profiles

Currently agents (random/neural/mcts), decks, and weights are all independent
args passed at the CLI. Both players always share the same deck + weights.
There's no link between checkpoints, decks, and results.

**New design: agent profiles** — each agent is a named directory that bundles
deck + strategy + weights + results together.

### Directory layout

```
agents/
  my_agent_v1/
    profile.yaml          # strategy, deck path, creation time, notes
    deck.csv              # the deck this agent was trained on
    checkpoints/
      my_agent_v1_07_05_14_30_22.pt   # mm_dd_hh_mm_ss
      my_agent_v1_07_05_16_45_10.pt
      latest.pt -> <most recent>
    metrics/
      train.csv
      eval.csv
    replays/
      vs_random_2026_07_05.html
```

### Play semantics

```bash
# Profile-based (common case) — agent name decides everything
pkm play my_agent_v1 random

# Override specific things (ad-hoc)
pkm play neural random --p0-weights checkpoints/ppo_iter0200.pt
pkm play neural random --p0-deck other_deck.csv

# Per-player deck support
pkm play agent_a agent_b --p0-deck deck_a.csv --p1-deck deck_b.csv
```

An "agent name" resolves to:
1. Check `agents/<name>/profile.yaml` — use its deck, strategy, latest checkpoint
2. Check built-in names: `random`, `neural`, `mcts` (legacy, use defaults)
3. Error if not found

---

## Tasks

### 1. CLI migration: argparse -> typer + rich
- [x] Migrated train.py, exit_train.py, play.py, export.py from argparse to typer
- [x] All `--help` outputs now use typer+rich formatting
- [x] Justfile commands work unchanged (same `--flag` syntax)
- [ ] `pkm/cli_profile.py` — `profile create/list/info` subcommands
- [ ] Rich output: progress bars for training, tables for eval results, colored status

### 2. Agent profile system
- [ ] `pkm/profile.py` — Profile class (load/save YAML, resolve agent)
- [ ] Profile YAML schema: strategy, deck_path, created_at, notes
- [ ] `agents/` directory creation logic
- [ ] Checkpoint naming: `<profile_name>_MM_DD_HH_MM_SS.pt`
- [ ] Symlink management for `latest.pt`
- [ ] Profile-aware `make_agent_by_name()` that resolves profiles

### 3. Link results to agents
- [ ] Metrics CSV goes into `agents/<name>/metrics/`
- [ ] Replays go into `agents/<name>/replays/`
- [ ] Eval results logged to `agents/<name>/metrics/eval.csv`
- [ ] Training results logged to `agents/<name>/metrics/train.csv`

### 4. Per-player deck support
- [ ] `play.py` accepts per-player deck paths
- [ ] `play_match()` and `win_rate()` support different decks per side
- [ ] Kaggle env gets correct deck per side

### 5. Visualization / replay
- [ ] Verify `result.html` rendering works end-to-end
- [ ] Rich terminal summary after play (colored win/loss/draw, game stats)
- [ ] `pkm watch` subcommand to open latest replay in browser

### 6. TUI (stretch)
- [ ] Investigate textual/rich for interactive agent selection
- [ ] Browse profiles, compare results, select matchups interactively

### 7. Deck system
- [x] `deck/` directory structure (CSV + JSON formats)
- [x] JSON deck format (id/name/count)
- [x] Deck CLI (`pkm/cli_deck.py`): list/show/convert subcommands

### 8. Clean up
- [ ] Update AGENTS.md with new architecture
- [ ] Update justfile recipes
- [ ] Update submit.sh to work with profile system
- [ ] Migration: create profile from existing checkpoints + deck.csv
