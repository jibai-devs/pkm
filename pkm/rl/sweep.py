"""Hyperparameter sweep with Optuna.

Usage:
    pkm sweep --trials 50 --games 8 --iterations 20
    pkm sweep --agent 01_psychic --trials 100
    pkm sweep-exit --trials 30 --iterations 10

Optuna searches over: lr, gamma, lam, shaping_coef, pool_size, pool_prob.
"""

from __future__ import annotations

from pathlib import Path

import typer

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from pkm.agents.profile import AgentProfile
from pkm.data import Deck
from pkm.rl.train import evaluate_vs_random, train as ppo_train


def _ppo_objective(
    trial: optuna.Trial,
    *,
    deck_path: str,
    iterations: int,
    games: int,
    seed: int,
    checkpoint_dir: str,
    metrics_dir: str,
    runs_dir: str,
) -> float:
    """Objective function for PPO hyperparameter search."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    lam = trial.suggest_float("lam", 0.9, 0.99)
    shaping_coef = trial.suggest_float("shaping_coef", 0.0, 0.5)
    pool_size = trial.suggest_int("pool_size", 4, 16)
    pool_prob = trial.suggest_float("pool_prob", 0.2, 0.8)

    model = ppo_train(
        deck_path=deck_path,
        iterations=iterations,
        games_per_iter=games,
        lr=lr,
        gamma=gamma,
        lam=lam,
        weights={"shaping": shaping_coef},
        pool_size=pool_size,
        pool_prob=pool_prob,
        eval_every=iterations,  # only eval at the end
        eval_games=30,
        checkpoint_dir=checkpoint_dir,
        metrics_path=str(Path(metrics_dir) / f"ppo_sweep_{trial.number}.csv"),
        log_dir=str(Path(runs_dir) / f"ppo_sweep_{trial.number}"),
        seed=seed + trial.number,
    )

    deck = Deck.from_csv(deck_path).card_ids
    win_rate = evaluate_vs_random(model, deck, games=30)
    return win_rate


def _exit_objective(
    trial: optuna.Trial,
    *,
    deck_path: str,
    iterations: int,
    games: int,
    seed: int,
    init_checkpoint: str,
    checkpoint_dir: str,
    metrics_dir: str,
    runs_dir: str,
) -> float:
    """Objective function for expert iteration hyperparameter search."""
    from pkm.rl.exit_train import train as exit_train

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    n_simulations = trial.suggest_int("n_simulations", 16, 64, step=8)
    n_determinizations = trial.suggest_int("n_determinizations", 1, 4)

    model = exit_train(
        deck_path=deck_path,
        iterations=iterations,
        games_per_iter=games,
        n_simulations=n_simulations,
        n_determinizations=n_determinizations,
        lr=lr,
        init_checkpoint=init_checkpoint,
        checkpoint_dir=checkpoint_dir,
        metrics_path=str(Path(metrics_dir) / f"exit_sweep_{trial.number}.csv"),
        log_dir=str(Path(runs_dir) / f"exit_sweep_{trial.number}"),
        seed=seed + trial.number,
    )

    deck = Deck.from_csv(deck_path).card_ids
    win_rate = evaluate_vs_random(model, deck, games=30)
    return win_rate


sweep_app = typer.Typer(help=__doc__)


@sweep_app.command()
def sweep(
    agent: str | None = typer.Option(None, help="agent profile name"),
    deck: str = typer.Option("deck/02_dragapult.csv", help="path to deck CSV"),
    trials: int = typer.Option(50, help="number of Optuna trials"),
    iterations: int = typer.Option(20, help="training iterations per trial"),
    games: int = typer.Option(8, help="games per iteration"),
    seed: int = typer.Option(0, help="base random seed"),
    study_name: str = typer.Option("ppo_sweep", help="Optuna study name"),
    storage: str | None = typer.Option(None, help="Optuna storage URL (e.g. sqlite:///sweep.db)"),
) -> None:
    """PPO hyperparameter sweep."""
    if not HAS_OPTUNA:
        print("optuna not installed. Run: uv add --group dev optuna")
        raise typer.Exit(1)

    if agent:
        profile = AgentProfile(agent)
        profile.ensure_dirs()
        deck = str(profile.deck_path)
        checkpoint_dir = str(profile.checkpoint_dir)
        metrics_dir = str(profile.metrics_dir)
        runs_dir = str(profile.runs_dir)
    else:
        checkpoint_dir = "checkpoints"
        metrics_dir = "metrics"
        runs_dir = "runs"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: _ppo_objective(
            trial,
            deck_path=deck,
            iterations=iterations,
            games=games,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            metrics_dir=metrics_dir,
            runs_dir=runs_dir,
        ),
        n_trials=trials,
    )

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best win rate: {study.best_value:.1%}")
    print(f"Best params: {study.best_params}")


@sweep_app.command(name="exit")
def exit_sweep(
    agent: str | None = typer.Option(None, help="agent profile name"),
    deck: str = typer.Option("deck/02_dragapult.csv", help="path to deck CSV"),
    trials: int = typer.Option(30, help="number of Optuna trials"),
    iterations: int = typer.Option(10, help="training iterations per trial"),
    games: int = typer.Option(4, help="games per iteration"),
    init: str = typer.Option("checkpoints/ppo_latest.pt", help="initial checkpoint"),
    seed: int = typer.Option(0, help="base random seed"),
    study_name: str = typer.Option("exit_sweep", help="Optuna study name"),
    storage: str | None = typer.Option(None, help="Optuna storage URL"),
) -> None:
    """Expert iteration hyperparameter sweep."""
    if not HAS_OPTUNA:
        print("optuna not installed. Run: uv add --group dev optuna")
        raise typer.Exit(1)

    if agent:
        profile = AgentProfile(agent)
        profile.ensure_dirs()
        deck = str(profile.deck_path)
        checkpoint_dir = str(profile.checkpoint_dir)
        metrics_dir = str(profile.metrics_dir)
        runs_dir = str(profile.runs_dir)
        if init == "checkpoints/ppo_latest.pt":
            init = profile.exit_init()
    else:
        checkpoint_dir = "checkpoints"
        metrics_dir = "metrics"
        runs_dir = "runs"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: _exit_objective(
            trial,
            deck_path=deck,
            iterations=iterations,
            games=games,
            seed=seed,
            init_checkpoint=init,
            checkpoint_dir=checkpoint_dir,
            metrics_dir=metrics_dir,
            runs_dir=runs_dir,
        ),
        n_trials=trials,
    )

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best win rate: {study.best_value:.1%}")
    print(f"Best params: {study.best_params}")
