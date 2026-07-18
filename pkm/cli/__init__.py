"""pkm — Pokémon TCG AI CLI.

Usage:
    pkm deck list
    pkm deck show 00_basic
    pkm cards dump cards.json
    pkm train --agent 00_basic --iterations 50
    pkm exit-train --agent 00_basic
    pkm export --agent 00_basic
    pkm play --p0 mcts --p1 neural
"""

import typer

from pkm.cli.deck import app as deck_app
from pkm.cli.cards import app as cards_app
from pkm.rl.sweep import sweep_app

app = typer.Typer(help="pkm — Pokémon TCG AI CLI")

app.add_typer(deck_app, name="deck", help="Deck management")
app.add_typer(cards_app, name="cards", help="Card data")
app.add_typer(sweep_app, name="sweep", help="Hyperparameter sweeps")


# Single-command modules: register their main functions directly so
# `pkm train --help` works (not `pkm train main --help`).


@app.command()
def train(
    agent: str | None = typer.Option(
        None, help="agent profile name (e.g. 00_basic, 01_psychic)"
    ),
    deck: str = typer.Option("deck/02_dragapult.csv", help="path to deck CSV"),
    iterations: int = typer.Option(50, help="number of training iterations"),
    games: int = typer.Option(8, help="games per iteration"),
    lr: float = typer.Option(3e-4, help="learning rate"),
    gamma: float = typer.Option(0.99, help="discount factor"),
    weights: str | None = typer.Option(
        None,
        "--weights",
        help="path to a JSON file of {term: weight} overrides — see "
        "pkm/rl/reward_terms.py for term names and defaults. Defaults "
        "to the agent's own reward_weights.json when --agent is given.",
    ),
    pool_size: int = typer.Option(8, help="opponent checkpoint pool size"),
    eval_every: int = typer.Option(5, help="evaluate every N iterations"),
    eval_games: int = typer.Option(20, help="games for evaluation"),
    checkpoint_dir: str = typer.Option("checkpoints", help="checkpoint directory"),
    metrics: str = typer.Option("metrics/ppo_train.csv", help="metrics CSV path"),
    log_dir: str = typer.Option("runs/ppo", help="TensorBoard log directory"),
    init: str | None = typer.Option(None, help="checkpoint to resume from"),
    seed: int = typer.Option(0, help="random seed"),
    wandb_project: str | None = typer.Option(None, help="wandb project name (enables wandb logging)"),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name"),
) -> None:
    """Phase 1: PPO self-play training."""
    from pkm.rl.train import main as _train_main

    _train_main(
        agent=agent,
        deck=deck,
        iterations=iterations,
        games=games,
        lr=lr,
        gamma=gamma,
        weights=weights,
        pool_size=pool_size,
        eval_every=eval_every,
        eval_games=eval_games,
        checkpoint_dir=checkpoint_dir,
        metrics=metrics,
        log_dir=log_dir,
        init=init,
        seed=seed,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


@app.command(name="exit-train")
def exit_train(
    agent: str | None = typer.Option(
        None, help="agent profile name (e.g. 00_basic, 01_psychic)"
    ),
    deck: str = typer.Option("deck/02_dragapult.csv", help="path to deck CSV"),
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
    wandb_project: str | None = typer.Option(None, help="wandb project name (enables wandb logging)"),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name"),
) -> None:
    """Phase 2: expert iteration (AlphaZero-style)."""
    from pkm.rl.exit_train import main as _exit_train_main

    _exit_train_main(
        agent=agent,
        deck=deck,
        iterations=iterations,
        games=games,
        sims=sims,
        dets=dets,
        lr=lr,
        init=init,
        checkpoint_dir=checkpoint_dir,
        metrics=metrics,
        log_dir=log_dir,
        seed=seed,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


@app.command()
def export(
    checkpoint: str = typer.Argument(
        "", help="path to .pt state_dict (omit with --agent)"
    ),
    out: str = typer.Argument("pkm/policy.npz", help="output .npz path"),
    agent: str | None = typer.Option(None, help="agent profile name"),
) -> None:
    """Export checkpoint to .npz for torch-free inference."""
    from pkm.rl.export import main as _export_main

    _export_main(checkpoint=checkpoint, out=out, agent=agent)


@app.command()
def play(
    p0: str = typer.Option("neural", help="player 0 agent: random|neural|mcts|human"),
    p1: str = typer.Option("random", help="player 1 agent: random|neural|mcts|human"),
    agent: str | None = typer.Option(
        None, help="agent profile name (resolves deck + weights)"
    ),
    deck: str = typer.Option("deck/02_dragapult.csv", help="path to deck CSV"),
    weights: str | None = typer.Option(None, help="path to policy .npz"),
    html: str = typer.Option("result.html", help="HTML replay output path"),
    replay: str = typer.Option("replay.json", help="JSON replay output path"),
    games: int = typer.Option(1, help=">1: win-rate mode, no replay"),
) -> None:
    """Play/evaluate matches."""
    from pkm.rl.play import main as _play_main

    _play_main(
        p0=p0,
        p1=p1,
        agent=agent,
        deck=deck,
        weights=weights,
        html=html,
        replay=replay,
        games=games,
    )


if __name__ == "__main__":
    app()
