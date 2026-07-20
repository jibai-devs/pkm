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

from pkm.cli.archetype import app as archetype_app
from pkm.cli.deck import app as deck_app
from pkm.cli.cards import app as cards_app
from pkm.new_agents.cli import app as new_agents_app
from pkm.rl.sweep import sweep_app

app = typer.Typer(help="pkm — Pokémon TCG AI CLI")

app.add_typer(deck_app, name="deck", help="Deck management")
app.add_typer(cards_app, name="cards", help="Card data")
app.add_typer(sweep_app, name="sweep", help="Hyperparameter sweeps")
app.add_typer(new_agents_app, name="new_agents", help="Standalone next-gen agents")
app.add_typer(archetype_app, name="archetype", help="Opponent-archetype classifier")


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
        "pkm/rl/reward_terms.py for term names and defaults. Defaults to "
        "the agent's own reward_weights.json (auto-created there on first "
        "use) when --agent is given, otherwise the built-in defaults.",
    ),
    pool_size: int = typer.Option(8, help="opponent checkpoint pool size"),
    use_archetype_pool: bool = typer.Option(
        False,
        "--archetype-pool",
        help="sample cross-archetype opponents from trained agents/pool_*/ "
        "bots (Part 3c) in addition to the self-checkpoint pool above; "
        "requires deck/pool_*.csv + agents/pool_*/checkpoints/ppo_latest.pt "
        "(see pkm/rl/opponent_pool.py)",
    ),
    archetype_pool_prob: float = typer.Option(
        0.2,
        help="fraction of games played against a random pool bot on its own "
        "deck, when --archetype-pool is set",
    ),
    use_archetype_belief: bool = typer.Option(
        False,
        "--archetype-belief",
        help="inject the standalone opponent-archetype classifier's belief "
        "into the encoder for the trainee's decisions (Part 2a) -- loads "
        "--archetype-weights once and attaches it to the trainee's "
        "TorchPolicy only, never the frozen opponent's",
    ),
    archetype_weights: str = typer.Option(
        "pkm/archetype.npz",
        help="path to the exported NumpyArchetypeClassifier weights, used "
        "when --archetype-belief is set",
    ),
    eval_every: int = typer.Option(5, help="evaluate every N iterations"),
    eval_games: int = typer.Option(20, help="games for evaluation"),
    checkpoint_dir: str = typer.Option("checkpoints", help="checkpoint directory"),
    metrics: str = typer.Option("metrics/ppo_train.csv", help="metrics CSV path"),
    log_dir: str = typer.Option("runs/ppo", help="TensorBoard log directory"),
    init: str | None = typer.Option(None, help="checkpoint to resume from"),
    seed: int = typer.Option(0, help="random seed"),
    wandb_project: str | None = typer.Option(
        None, help="wandb project name (enables wandb logging)"
    ),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name"),
    workers: int = typer.Option(
        1, help="parallel worker processes for self-play rollout (1 = sequential)"
    ),
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
        use_archetype_pool=use_archetype_pool,
        archetype_pool_prob=archetype_pool_prob,
        use_archetype_belief=use_archetype_belief,
        archetype_weights=archetype_weights,
        eval_every=eval_every,
        eval_games=eval_games,
        checkpoint_dir=checkpoint_dir,
        metrics=metrics,
        log_dir=log_dir,
        init=init,
        seed=seed,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        workers=workers,
    )


@app.command(name="population-train")
def population_train_cmd(
    iterations: int = typer.Option(100, help="number of iterations"),
    games_per_pairing: int = typer.Option(
        2, help="anchor games per pool-bot pairing per iteration"
    ),
    lr: float = typer.Option(3e-4, help="learning rate"),
    gamma: float = typer.Option(0.99, help="discount factor"),
    lam: float = typer.Option(0.95, help="GAE lambda"),
    update_every: int = typer.Option(
        3,
        help="max iterations a member's trajectories buffer before a forced "
        "PPO update, even under min-samples",
    ),
    min_samples: int = typer.Option(
        512, help="min buffered samples before a member's PPO update fires early"
    ),
    anchor: str = typer.Option("03_pult_munki", help="anchor agent profile name"),
    pool_glob: str = typer.Option(
        "pool_*", help="glob under agents/ for pool-bot profiles"
    ),
    eval_every: int = typer.Option(10, help="evaluate + checkpoint every N iterations"),
    eval_games: int = typer.Option(20, help="games for evaluation"),
    workers: int = typer.Option(
        1, help="parallel worker processes for self-play rollout (1 = sequential)"
    ),
    seed: int = typer.Option(0, help="random seed"),
) -> None:
    """Milestone 9: simultaneous population training (anchor + pool bots)."""
    from pkm.rl.population_train import population_train as _population_train

    _population_train(
        iterations=iterations,
        games_per_pairing=games_per_pairing,
        lr=lr,
        gamma=gamma,
        lam=lam,
        update_every=update_every,
        min_samples=min_samples,
        anchor=anchor,
        pool_glob=pool_glob,
        eval_every=eval_every,
        eval_games=eval_games,
        workers=workers,
        seed=seed,
    )


@app.command(name="eval-vs-pool")
def eval_vs_pool_cmd(
    agent: str = typer.Option("03_pult_munki", help="agent profile name to evaluate"),
    games: int = typer.Option(20, help="games per pool bot, alternating sides"),
    pool_glob: str = typer.Option(
        "pool_*", help="glob under agents/ for pool-bot profiles"
    ),
    use_archetype_belief: bool = typer.Option(
        True,
        "--archetype-belief/--no-archetype-belief",
        help="compute live opponent-archetype belief for both sides, "
        "matching pkm play/Kaggle (default: on). --no-archetype-belief "
        "reproduces the old always-zero-belief baseline.",
    ),
    archetype_weights: str = typer.Option(
        "pkm/archetype.npz",
        help="path to the exported NumpyArchetypeClassifier weights, used "
        "when --archetype-belief is set",
    ),
) -> None:
    """Per-archetype win rate against every trained agents/pool_*/ bot, on
    each bot's own deck (evaluate_vs_random only covers vs the random agent)."""
    from pkm.rl.eval_vs_pool import _load_archetype_classifier
    from pkm.rl.eval_vs_pool import eval_vs_pool as _eval_vs_pool

    archetype_classifier = (
        _load_archetype_classifier(archetype_weights) if use_archetype_belief else None
    )
    _eval_vs_pool(
        agent=agent,
        games=games,
        pool_glob=pool_glob,
        archetype_classifier=archetype_classifier,
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
    wandb_project: str | None = typer.Option(
        None, help="wandb project name (enables wandb logging)"
    ),
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
    p0_agent: str | None = typer.Option(
        None,
        "--p0-agent",
        help="agent profile name for player 0 only (different deck/weights per side)",
    ),
    p1_agent: str | None = typer.Option(
        None, "--p1-agent", help="agent profile name for player 1 only"
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
        p0_agent=p0_agent,
        p1_agent=p1_agent,
        deck=deck,
        weights=weights,
        html=html,
        replay=replay,
        games=games,
    )


if __name__ == "__main__":
    app()
