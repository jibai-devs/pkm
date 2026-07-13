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

from pathlib import Path

import typer

from pkm.cli.deck import app as deck_app
from pkm.cli.cards import app as cards_app

app = typer.Typer(help="pkm — Pokémon TCG AI CLI")

app.add_typer(deck_app, name="deck", help="Deck management")
app.add_typer(cards_app, name="cards", help="Card data")


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
    shaping: float = typer.Option(0.2, help="reward shaping coefficient"),
    pool_size: int = typer.Option(8, help="opponent checkpoint pool size"),
    eval_every: int = typer.Option(5, help="evaluate every N iterations"),
    eval_games: int = typer.Option(20, help="games for evaluation"),
    checkpoint_dir: str | None = typer.Option(None, help="checkpoint directory"),
    metrics: str | None = typer.Option(None, help="metrics CSV path"),
    log_dir: str | None = typer.Option(None, help="TensorBoard log directory"),
    init: str | None = typer.Option(None, help="checkpoint to resume from"),
    seed: int = typer.Option(0, help="random seed"),
) -> None:
    """Phase 1: PPO self-play training."""
    if agent:
        from pkm.agents.profile import AgentProfile

        AgentProfile.load(agent).train(
            iterations=iterations,
            games=games,
            lr=lr,
            gamma=gamma,
            shaping_coef=shaping,
            pool_size=pool_size,
            eval_every=eval_every,
            eval_games=eval_games,
            seed=seed,
            resume_path=Path(init) if init else None,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            metrics_path=Path(metrics) if metrics else None,
            log_dir=Path(log_dir) if log_dir else None,
        )
        return
    from pkm.rl.train import main as _train_main

    _train_main(
        agent=agent,
        deck=deck,
        iterations=iterations,
        games=games,
        lr=lr,
        gamma=gamma,
        shaping=shaping,
        pool_size=pool_size,
        eval_every=eval_every,
        eval_games=eval_games,
        checkpoint_dir=checkpoint_dir or "checkpoints",
        metrics=metrics or "metrics/ppo_train.csv",
        log_dir=log_dir or "runs/ppo",
        init=init,
        seed=seed,
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
    init: str | None = typer.Option(None, help="initial checkpoint"),
    checkpoint_dir: str | None = typer.Option(None, help="checkpoint directory"),
    metrics: str | None = typer.Option(None, help="metrics CSV path"),
    log_dir: str | None = typer.Option(None, help="TensorBoard log directory"),
    seed: int = typer.Option(0, help="random seed"),
    resume: bool = typer.Option(False, help="resume expert iteration"),
) -> None:
    """Phase 2: expert iteration (AlphaZero-style)."""
    if agent:
        from pkm.agents.profile import AgentProfile

        AgentProfile.load(agent).train_exit(
            iterations=iterations,
            games=games,
            n_simulations=sims,
            n_determinizations=dets,
            lr=lr,
            seed=seed,
            resume=resume,
            resume_path=Path(init) if init else None,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            metrics_path=Path(metrics) if metrics else None,
            log_dir=Path(log_dir) if log_dir else None,
        )
        return
    from pkm.rl.exit_train import main as _exit_train_main

    _exit_train_main(
        agent=agent,
        deck=deck,
        iterations=iterations,
        games=games,
        sims=sims,
        dets=dets,
        lr=lr,
        init=init or "checkpoints/ppo_latest.pt",
        checkpoint_dir=checkpoint_dir or "checkpoints",
        metrics=metrics or "metrics/exit_train.csv",
        log_dir=log_dir or "runs/exit",
        seed=seed,
    )


@app.command()
def export(
    checkpoint: str = typer.Argument(
        "", help="path to .pt state_dict (omit with --agent)"
    ),
    out: str | None = typer.Argument(None, help="output .npz path"),
    agent: str | None = typer.Option(None, help="agent profile name"),
    phase: str = typer.Option(
        "ppo", help="which profile checkpoint to export: ppo or exit"
    ),
) -> None:
    """Export checkpoint to .npz for torch-free inference."""
    from pkm.rl.export import main as _export_main

    _export_main(checkpoint=checkpoint, out=out, agent=agent, phase=phase)


@app.command()
def play(
    p0: str = typer.Option("neural", help="player 0 agent: random|neural|mcts"),
    p1: str = typer.Option("random", help="player 1 agent: random|neural|mcts"),
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
