# argparse → typer Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all `argparse` usage with `typer` + `rich` for prettier CLI output across the 4 RL entry points.

**Architecture:** Each module's `main()` function (which builds an argparse parser and calls a typed `train()`/`play_match()`/etc. function) gets replaced by a `typer.Typer()` app with `@app.command()` decorators. The underlying typed functions stay unchanged. `if __name__ == "__main__"` calls `app()` for `python -m` compatibility.

**Tech Stack:** typer (already in pyproject.toml), rich (ships with typer)

---

## Files to Modify

| File | Change |
|------|--------|
| `pkm/rl/train.py` | Replace `main()` argparse with typer app |
| `pkm/rl/exit_train.py` | Replace `main()` argparse with typer app |
| `pkm/rl/play.py` | Replace `main()` argparse with typer app |
| `pkm/rl/export.py` | Replace `__main__` argparse with typer app |
| `justfile` | No changes needed — `python -m pkm.rl.train --iterations 50` works the same with typer |

## Task 1: Convert `pkm/rl/export.py`

**Files:**
- Modify: `pkm/rl/export.py:26-34`

Simplest module first — two positional args, good warm-up.

- [ ] **Step 1: Replace argparse block with typer app**

Replace lines 26-34:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to .pt state_dict")
    parser.add_argument("out", help="output .npz path (e.g. pkm/policy.npz)")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.out)
    print(f"exported {args.checkpoint} -> {args.out}")
```

With:

```python
if __name__ == "__main__":
    import typer

    app = typer.Typer(help=__doc__)

    @app.command()
    def main(
        checkpoint: str = typer.Argument(help="path to .pt state_dict"),
        out: str = typer.Argument(help="output .npz path (e.g. pkm/policy.npz)"),
    ) -> None:
        export_checkpoint(checkpoint, out)
        typer.echo(f"exported {checkpoint} -> {out}")

    app()
```

- [ ] **Step 2: Test it runs**

```bash
python -m pkm.rl.export --help
```

Expected: typer-generated help text showing `CHECKPOINT` and `OUT` arguments.

---

## Task 2: Convert `pkm/rl/train.py`

**Files:**
- Modify: `pkm/rl/train.py:7,206-246`

13 options, all with defaults. The `train()` function already has typed params.

- [ ] **Step 1: Replace argparse main with typer app**

Remove `import argparse` (line 7). Replace lines 206-246:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="deck.csv")
    # ... all the args ...
    args = parser.parse_args()
    train(
        deck_path=args.deck,
        # ... all the mappings ...
    )


if __name__ == "__main__":
    main()
```

With:

```python
def main(
    deck: str = typer.Option("deck.csv", help="path to deck CSV"),
    iterations: int = typer.Option(50, help="number of training iterations"),
    games: int = typer.Option(8, help="games per iteration"),
    lr: float = typer.Option(3e-4, help="learning rate"),
    gamma: float = typer.Option(0.99, help="discount factor"),
    shaping: float = typer.Option(0.2, help="reward shaping coefficient"),
    pool_size: int = typer.Option(8, help="opponent checkpoint pool size"),
    eval_every: int = typer.Option(5, help="evaluate every N iterations"),
    eval_games: int = typer.Option(20, help="games for evaluation"),
    checkpoint_dir: str = typer.Option("checkpoints", help="checkpoint directory"),
    metrics: str = typer.Option("metrics/ppo_train.csv", help="metrics CSV path"),
    log_dir: str = typer.Option("runs/ppo", help="TensorBoard log directory"),
    init: str | None = typer.Option(None, help="checkpoint to resume from"),
    seed: int = typer.Option(0, help="random seed"),
) -> None:
    train(
        deck_path=deck,
        iterations=iterations,
        games_per_iter=games,
        lr=lr,
        gamma=gamma,
        shaping_coef=shaping,
        pool_size=pool_size,
        eval_every=eval_every,
        eval_games=eval_games,
        checkpoint_dir=checkpoint_dir,
        metrics_path=metrics,
        log_dir=log_dir,
        init_checkpoint=init,
        seed=seed,
    )


app = typer.Typer(help=__doc__)
app.command()(main)

if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Add `import typer` at the top**

Replace `import argparse` (line 7) with `import typer`.

- [ ] **Step 3: Test it runs**

```bash
python -m pkm.rl.train --help
```

Expected: typer help showing all options with defaults.

---

## Task 3: Convert `pkm/rl/exit_train.py`

**Files:**
- Modify: `pkm/rl/exit_train.py:12,286-316`

11 options. Same pattern as train.py.

- [ ] **Step 1: Replace argparse main with typer app**

Remove `import argparse` (line 12). Replace lines 286-316:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # ... args ...
    args = parser.parse_args()
    train(
        deck_path=args.deck,
        # ... mappings ...
    )


if __name__ == "__main__":
    main()
```

With:

```python
def main(
    deck: str = typer.Option("deck.csv", help="path to deck CSV"),
    iterations: int = typer.Option(3, help="number of training iterations"),
    games: int = typer.Option(4, help="games per iteration"),
    sims: int = typer.Option(24, help="MCTS simulations per move"),
    dets: int = typer.Option(2, help="MCTS determinizations"),
    lr: float = typer.Option(1e-4, help="learning rate"),
    init: str = typer.Option("checkpoints/ppo_latest.pt", help="initial checkpoint"),
    checkpoint_dir: str = typer.Option("checkpoints", help="checkpoint directory"),
    metrics: str = typer.Option("metrics/exit_train.csv", help="metrics CSV path"),
    log_dir: str = typer.Option("runs/exit", help="TensorBoard log directory"),
    seed: int = typer.Option(0, help="random seed"),
) -> None:
    train(
        deck_path=deck,
        iterations=iterations,
        games_per_iter=games,
        n_simulations=sims,
        n_determinizations=dets,
        lr=lr,
        init_checkpoint=init,
        checkpoint_dir=checkpoint_dir,
        metrics_path=metrics,
        log_dir=log_dir,
        seed=seed,
    )


app = typer.Typer(help=__doc__)
app.command()(main)

if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Add `import typer` at the top**

Replace `import argparse` (line 12) with `import typer`.

- [ ] **Step 3: Test it runs**

```bash
python -m pkm.rl.exit_train --help
```

Expected: typer help showing all options.

---

## Task 4: Convert `pkm/rl/play.py`

**Files:**
- Modify: `pkm/rl/play.py:14,101-129`

7 options with conditional logic (games > 1 → win_rate mode).

- [ ] **Step 1: Replace argparse main with typer app**

Remove `import argparse` (line 14). Replace lines 101-129:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # ... args ...
    args = parser.parse_args()
    if args.games > 1:
        win_rate(...)
    else:
        play_match(...)


if __name__ == "__main__":
    main()
```

With:

```python
app = typer.Typer(help=__doc__)


@app.command()
def main(
    p0: str = typer.Option("neural", help="player 0 agent: random|neural|mcts"),
    p1: str = typer.Option("random", help="player 1 agent: random|neural|mcts"),
    deck: str = typer.Option("deck.csv", help="path to deck CSV"),
    weights: str | None = typer.Option(None, help="path to policy .npz"),
    html: str = typer.Option("result.html", help="HTML replay output path"),
    replay: str = typer.Option("replay.json", help="JSON replay output path"),
    games: int = typer.Option(1, help=">1: win-rate mode, no replay"),
) -> None:
    if games > 1:
        win_rate(p0, p1, games, deck_path=deck, weights=weights)
    else:
        play_match(p0, p1, deck_path=deck, weights=weights, html_path=html, replay_path=replay)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Add `import typer` at the top**

Replace `import argparse` (line 14) with `import typer`.

- [ ] **Step 3: Test it runs**

```bash
python -m pkm.rl.play --help
```

Expected: typer help showing all options.

---

## Task 5: Verify justfile compatibility

**Files:**
- None to modify — just verification

- [ ] **Step 1: Run each justfile recipe that invokes these modules**

```bash
just train 1 4        # tiny run: 1 iteration, 4 games
just exit-train 1 2   # tiny run: 1 iteration, 2 games, 8 sims
just play neural random
just eval neural random 2
just export
```

Expected: all commands parse args correctly and run (or fail for non-CLI reasons like missing checkpoints, which is fine).

- [ ] **Step 2: Verify `--help` on all four modules**

```bash
python -m pkm.rl.train --help
python -m pkm.rl.exit_train --help
python -m pkm.rl.play --help
python -m pkm.rl.export --help
```

Expected: all show typer-formatted help with options and defaults.

---

## Task 6: Lint check

- [ ] **Step 1: Run ruff**

```bash
ruff check pkm/rl/train.py pkm/rl/exit_train.py pkm/rl/play.py pkm/rl/export.py
ruff format --check pkm/rl/train.py pkm/rl/exit_train.py pkm/rl/play.py pkm/rl/export.py
```

Expected: no errors. Fix any unused-import warnings for `argparse`.
