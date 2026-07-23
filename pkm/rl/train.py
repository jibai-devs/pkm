"""Phase 1 training entry point: PPO self-play with an opponent checkpoint pool.

Usage:
    pkm train --iterations 50 --games 8
    pkm train --agent 01_psychic --iterations 100
    pkm train --agent 01_psychic --weights agents/01_psychic/reward_weights.json
"""

import typer
import copy
import csv
import random
import time
from pathlib import Path

import torch

from pkm.agents.profile import AgentProfile
from pkm.data import Deck

from .features import archetype_index, write_stamp_sidecar
from .model import PolicyValueNet
from .ppo import ppo_update
from .reward_terms import DEFAULT_WEIGHTS, load_weights, write_default_weights_file
from .rollout import (
    FirstTurnDelegatingPolicy,
    RandomPolicy,
    TorchPolicy,
    aggregate_result,
    make_game_specs,
    make_training_first_turn_agent,
    play_game,
    play_one,
)
from .logging import MetricLog


def evaluate_vs_random(
    model: PolicyValueNet,
    deck: list[int],
    games: int = 20,
    first_turn_agent=None,
) -> float:
    """Win rate of the greedy policy vs. the random agent, alternating sides.

    When `first_turn_agent` is given, the policy side delegates its own first
    turn to it -- so the metric reflects how the agent actually deploys."""
    policy = TorchPolicy(model, greedy=True)
    if first_turn_agent is not None:
        policy = FirstTurnDelegatingPolicy(policy, first_turn_agent)
    rand = RandomPolicy()
    wins = 0.0
    for g in range(games):
        side = g % 2
        policies = (policy, rand) if side == 0 else (rand, policy)
        result = play_game(policies, (deck, deck), collect=(False, False))
        r = result.rewards[side]
        wins += 1.0 if r > 0 else 0.5 if r == 0 else 0.0
    return wins / games


CSV_FIELDS = [
    "iter",
    "games",
    "wins",
    "losses",
    "draws",
    "decisions",
    "samples",
    "pi_loss",
    "v_loss",
    "entropy",
    "clip_frac",
    "archetype_loss",
    "time_s",
    "eval_win_rate",
    "eval_games",
]


def train(
    deck_path: str = "deck/02_dragapult.csv",
    iterations: int = 50,
    games_per_iter: int = 8,
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    temperature: float = 1.0,
    gamma: float = 0.99,
    lam: float = 0.95,
    weights: dict[str, float] | None = None,
    pool_size: int = 8,
    pool_prob: float = 0.4,
    eval_every: int = 5,
    eval_games: int = 20,
    checkpoint_dir: str = "checkpoints",
    metrics_path: str = "metrics/ppo_train.csv",
    log_dir: str = "runs/ppo",
    init_checkpoint: str | None = None,
    seed: int = 0,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    workers: int = 1,
    first_turn_delegate: bool = False,
    max_seconds: float | None = None,
    stop_file: str | None = None,
) -> PolicyValueNet:
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    effective_weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    deck = Deck.from_csv(deck_path).card_ids

    # First-turn delegation: the learner's own first turn is played by the
    # scripted first-turn agent (never collected), so the policy only trains
    # on turn 2+ -- exactly how singaporean_middleman deploys it. Built once
    # and shared across games (stateless per decision).
    ft_agent = make_training_first_turn_agent(deck) if first_turn_delegate else None
    if ft_agent is not None and workers > 1:
        # The parallel worker path (parallel_rollout.py) doesn't thread the
        # first-turn agent through; fall back to sequential rather than
        # silently training without delegation.
        print(
            "first-turn delegation is sequential-only; forcing workers=1",
            flush=True,
        )
        workers = 1
    # Task 8: self-play here always mirrors deck_path against itself (no
    # multi-deck opponent pool yet -- AGENTS.md "What's Next" #5), so the
    # opponent's archetype label is constant for the whole run. The aux
    # loss is real machinery, but degenerately single-class until that
    # roadmap item lands.
    archetype_label = archetype_index(deck_path)
    model = PolicyValueNet()
    if init_checkpoint:
        model.load_state_dict(
            torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        )
        print(f"resumed from {init_checkpoint}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(metrics_file, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    csv_w.writeheader()

    log = MetricLog()
    log.add_tensorboard(log_dir)
    if wandb_project:
        log.add_wandb(
            project=wandb_project,
            run_name=wandb_run_name,
            log_dir=log_dir,
            config={
                "algo": "ppo",
                "lr": lr,
                "gamma": gamma,
                "lam": lam,
                "weights": effective_weights,
                "pool_size": pool_size,
                "pool_prob": pool_prob,
                "games_per_iter": games_per_iter,
                "seed": seed,
                "deck": deck_path,
            },
        )

    # opponent pool of past parameters (state_dicts on CPU)
    pool: list[dict] = [copy.deepcopy(model.state_dict())]
    opponent_model = PolicyValueNet()

    executor = None
    if workers > 1:
        from .parallel_rollout import collect_parallel, make_pool

        executor = make_pool(workers)
        print(f"parallel rollout: {workers} worker processes", flush=True)

    run_start = time.time()
    try:
        for it in range(1, iterations + 1):
            if max_seconds is not None and time.time() - run_start > max_seconds:
                print(
                    f"reached max_seconds={max_seconds:.0f}s after {it - 1} "
                    "iterations; stopping",
                    flush=True,
                )
                break
            if stop_file is not None and Path(stop_file).exists():
                # Checked once per iteration (like max_seconds above), so this
                # isn't instant -- it finishes the in-flight iteration first,
                # which is what lets the unconditional save below still run.
                print(
                    f"stop file {stop_file} found after {it - 1} iterations; stopping",
                    flush=True,
                )
                break
            t0 = time.time()
            model.eval()
            data = []
            w = losses = d = 0
            total_decisions = 0

            specs = make_game_specs(games_per_iter, pool, pool_prob, rng)
            if executor is not None:
                results = collect_parallel(
                    executor, workers, model.state_dict(), deck, specs
                )
            else:
                results = [
                    play_one(
                        model, opponent_model, deck, spec, ft_agent,
                        temperature=temperature,
                    )
                    for spec in specs
                ]

            for spec, result in zip(specs, results):
                total_decisions += result.decisions
                gw, gl, gd = aggregate_result(
                    spec, result, data, gamma, lam, weights=effective_weights
                )
                w, losses, d = w + gw, losses + gl, d + gd

            for dec in data:
                dec.true_archetype = archetype_label

            model.train()
            stats = ppo_update(model, optimizer, data, entropy_coef=entropy_coef)
            model.eval()

            pool.append(copy.deepcopy(model.state_dict()))
            if len(pool) > pool_size:
                pool.pop(0)

            dt = time.time() - t0
            row = {
                "iter": it,
                "games": games_per_iter,
                "wins": w,
                "losses": losses,
                "draws": d,
                "decisions": total_decisions,
                "samples": len(data),
                "pi_loss": f"{stats['policy_loss']:.6f}",
                "v_loss": f"{stats['value_loss']:.6f}",
                "entropy": f"{stats['entropy']:.6f}",
                "clip_frac": f"{stats['clip_frac']:.4f}",
                "archetype_loss": f"{stats['archetype_loss']:.6f}",
                "time_s": f"{dt:.2f}",
                "eval_win_rate": "",
                "eval_games": "",
            }
            print(
                f"iter {it:3d} | games {games_per_iter} (W/L/D {w}/{losses}/{d}) "
                f"| decisions {total_decisions} | samples {len(data)} "
                f"| pi_loss {stats['policy_loss']:.4f} | v_loss {stats['value_loss']:.4f} "
                f"| ent {stats['entropy']:.3f} | clip {stats['clip_frac']:.2f} | {dt:.1f}s",
                flush=True,
            )

            if it % eval_every == 0:
                wr = evaluate_vs_random(
                    model, deck, games=eval_games, first_turn_agent=ft_agent
                )
                row["eval_win_rate"] = f"{wr:.4f}"
                row["eval_games"] = eval_games
                print(
                    f"iter {it:3d} | eval vs random: {wr:.1%} ({eval_games} games)",
                    flush=True,
                )
                torch.save(model.state_dict(), ckpt_dir / f"ppo_iter{it:04d}.pt")
                torch.save(model.state_dict(), ckpt_dir / "ppo_latest.pt")
                write_stamp_sidecar(ckpt_dir / "ppo_latest.pt")

            csv_w.writerow(row)
            csv_f.flush()

            log.scalar("loss/policy", stats["policy_loss"], it)
            log.scalar("loss/value", stats["value_loss"], it)
            log.scalar("loss/archetype", stats["archetype_loss"], it)
            log.scalar("policy/entropy", stats["entropy"], it)
            log.scalar("policy/clip_frac", stats["clip_frac"], it)
            log.scalar(
                "game/win_rate", w / (w + losses + d) if (w + losses + d) else 0, it
            )
            log.scalar("game/decisions", total_decisions, it)
            log.scalar("time/iter_s", dt, it)
            if row["eval_win_rate"]:
                log.scalar("eval/win_rate_vs_random", float(row["eval_win_rate"]), it)
    finally:
        if executor is not None:
            executor.shutdown()

    torch.save(model.state_dict(), ckpt_dir / "ppo_latest.pt")
    write_stamp_sidecar(ckpt_dir / "ppo_latest.pt")
    csv_f.close()
    log.close()
    print(f"metrics saved to {metrics_file}", flush=True)
    return model


app = typer.Typer(help=__doc__)


@app.command()
def main(
    agent: str | None = typer.Option(
        None, help="agent profile name (e.g. 00_basic, 01_psychic)"
    ),
    deck: str = typer.Option("deck/02_dragapult.csv", help="path to deck CSV"),
    iterations: int = typer.Option(50, help="number of training iterations"),
    games: int = typer.Option(8, help="games per iteration"),
    lr: float = typer.Option(3e-4, help="learning rate"),
    entropy_coef: float = typer.Option(
        0.01, help="exploration bonus; higher keeps the policy less certain"
    ),
    temperature: float = typer.Option(
        1.0, help="sampling temperature during rollout; >1 flattens the policy"
    ),
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
    first_turn_delegate: bool = typer.Option(
        False,
        "--first-turn-delegate/--no-first-turn-delegate",
        help="play the learner's own first turn with the scripted first-turn "
        "agent (never collected); trains the policy on turn 2+ only, matching "
        "how singaporean_middleman deploys. Sequential-only.",
    ),
    max_seconds: float | None = typer.Option(
        None,
        help="wall-clock budget in seconds; stop before the next iteration "
        "once exceeded (checkpoints are still saved). E.g. 21600 = 6 hours.",
    ),
    stop_file: str | None = typer.Option(
        None,
        help="if this path exists at the start of an iteration, finish that "
        "iteration, save a final checkpoint, and stop. Checked once per "
        "iteration (not instant). See ian_tools/train.sh for a wrapper that "
        "creates/removes this from a 'stop' typed at the console.",
    ),
) -> None:
    profile = None
    if agent:
        profile = AgentProfile(agent)
        profile.ensure_dirs()
        deck = str(profile.deck_path)
        checkpoint_dir = str(profile.checkpoint_dir)
        metrics = str(profile.metrics_dir / "ppo_train.csv")
        log_dir = str(profile.runs_dir / "ppo")
        if init is None:
            init = profile.ppo_init()
        if weights is None:
            write_default_weights_file(profile.reward_weights_path)
            weights = str(profile.reward_weights_path)
    train(
        deck_path=deck,
        iterations=iterations,
        games_per_iter=games,
        lr=lr,
        gamma=gamma,
        weights=load_weights(weights),
        pool_size=pool_size,
        eval_every=eval_every,
        eval_games=eval_games,
        checkpoint_dir=checkpoint_dir,
        metrics_path=metrics,
        log_dir=log_dir,
        init_checkpoint=init,
        seed=seed,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        workers=workers,
        first_turn_delegate=first_turn_delegate,
        max_seconds=max_seconds,
        stop_file=stop_file,
    )


if __name__ == "__main__":
    app()
