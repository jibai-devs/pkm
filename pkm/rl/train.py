"""Phase 1 training entry point: PPO self-play with an opponent checkpoint pool.

Usage:
    pkm train --iterations 50 --games 8
    pkm train --agent 01_psychic --iterations 100
"""

import typer
import copy
import csv
import random
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from pkm.agents.profile import AgentProfile
from pkm.data import Deck

from .model import PolicyValueNet
from .ppo import ppo_update
from .rollout import (
    RandomPolicy,
    TorchPolicy,
    aggregate_result,
    make_game_specs,
    play_game,
    play_one,
)


def evaluate_vs_random(
    model: PolicyValueNet, deck: list[int], games: int = 20
) -> float:
    """Win rate of the greedy policy vs. the random agent, alternating sides."""
    policy = TorchPolicy(model, greedy=True)
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
    "time_s",
    "eval_win_rate",
    "eval_games",
]


def train(
    deck_path: str = "deck/02_dragapult.csv",
    iterations: int = 50,
    games_per_iter: int = 8,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    shaping_coef: float = 0.2,
    energy_penalty_coef: float = 0.0,
    pool_size: int = 8,
    pool_prob: float = 0.4,
    eval_every: int = 5,
    eval_games: int = 20,
    checkpoint_dir: str = "checkpoints",
    metrics_path: str = "metrics/ppo_train.csv",
    log_dir: str = "runs/ppo",
    init_checkpoint: str | None = None,
    seed: int = 0,
    workers: int = 1,
) -> PolicyValueNet:
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    deck = Deck.from_csv(deck_path).card_ids
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

    tb = SummaryWriter(log_dir)

    # opponent pool of past parameters (state_dicts on CPU)
    pool: list[dict] = [copy.deepcopy(model.state_dict())]
    opponent_model = PolicyValueNet()

    executor = None
    if workers > 1:
        from .parallel_rollout import collect_parallel, make_pool

        executor = make_pool(workers)
        print(f"parallel rollout: {workers} worker processes", flush=True)

    try:
        for it in range(1, iterations + 1):
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
                    play_one(model, opponent_model, deck, spec) for spec in specs
                ]

            for spec, result in zip(specs, results):
                total_decisions += result.decisions
                gw, gl, gd = aggregate_result(
                    spec, result, data, gamma, lam, shaping_coef, energy_penalty_coef
                )
                w, losses, d = w + gw, losses + gl, d + gd

            model.train()
            stats = ppo_update(model, optimizer, data)
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
                wr = evaluate_vs_random(model, deck, games=eval_games)
                row["eval_win_rate"] = f"{wr:.4f}"
                row["eval_games"] = eval_games
                print(
                    f"iter {it:3d} | eval vs random: {wr:.1%} ({eval_games} games)",
                    flush=True,
                )
                torch.save(model.state_dict(), ckpt_dir / f"ppo_iter{it:04d}.pt")
                torch.save(model.state_dict(), ckpt_dir / "ppo_latest.pt")

            csv_w.writerow(row)
            csv_f.flush()

            tb.add_scalar("loss/policy", stats["policy_loss"], it)
            tb.add_scalar("loss/value", stats["value_loss"], it)
            tb.add_scalar("policy/entropy", stats["entropy"], it)
            tb.add_scalar("policy/clip_frac", stats["clip_frac"], it)
            tb.add_scalar(
                "game/win_rate", w / (w + losses + d) if (w + losses + d) else 0, it
            )
            tb.add_scalar("game/decisions", total_decisions, it)
            tb.add_scalar("time/iter_s", dt, it)
            if row["eval_win_rate"]:
                tb.add_scalar(
                    "eval/win_rate_vs_random", float(row["eval_win_rate"]), it
                )
    finally:
        if executor is not None:
            executor.shutdown()

    torch.save(model.state_dict(), ckpt_dir / "ppo_latest.pt")
    csv_f.close()
    tb.close()
    print(f"metrics saved to {metrics_file}", flush=True)
    return model


app = typer.Typer(help=__doc__)


@app.command()
def main(
    agent: str | None = typer.Option(None, help="agent profile name (e.g. 00_basic, 01_psychic)"),
    deck: str = typer.Option("deck/02_dragapult.csv", help="path to deck CSV"),
    iterations: int = typer.Option(50, help="number of training iterations"),
    games: int = typer.Option(8, help="games per iteration"),
    lr: float = typer.Option(3e-4, help="learning rate"),
    gamma: float = typer.Option(0.99, help="discount factor"),
    shaping: float = typer.Option(0.2, help="reward shaping coefficient"),
    energy_penalty: float = typer.Option(
        0.0,
        help="penalty for attaching energy to the active Pokemon when it "
        "can already use every attack and retreat (0 = off)",
    ),
    pool_size: int = typer.Option(8, help="opponent checkpoint pool size"),
    eval_every: int = typer.Option(5, help="evaluate every N iterations"),
    eval_games: int = typer.Option(20, help="games for evaluation"),
    checkpoint_dir: str = typer.Option("checkpoints", help="checkpoint directory"),
    metrics: str = typer.Option("metrics/ppo_train.csv", help="metrics CSV path"),
    log_dir: str = typer.Option("runs/ppo", help="TensorBoard log directory"),
    init: str | None = typer.Option(None, help="checkpoint to resume from"),
    seed: int = typer.Option(0, help="random seed"),
    workers: int = typer.Option(
        1, help="parallel worker processes for self-play rollout (1 = sequential)"
    ),
) -> None:
    if agent:
        profile = AgentProfile(agent)
        profile.ensure_dirs()
        deck = str(profile.deck_path)
        checkpoint_dir = str(profile.checkpoint_dir)
        metrics = str(profile.metrics_dir / "ppo_train.csv")
        log_dir = str(profile.runs_dir / "ppo")
        if init is None:
            init = profile.ppo_init()
    train(
        deck_path=deck,
        iterations=iterations,
        games_per_iter=games,
        lr=lr,
        gamma=gamma,
        shaping_coef=shaping,
        energy_penalty_coef=energy_penalty,
        pool_size=pool_size,
        eval_every=eval_every,
        eval_games=eval_games,
        checkpoint_dir=checkpoint_dir,
        metrics_path=metrics,
        log_dir=log_dir,
        init_checkpoint=init,
        seed=seed,
        workers=workers,
    )


if __name__ == "__main__":
    app()
