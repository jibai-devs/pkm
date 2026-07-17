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
from torch.utils.tensorboard import SummaryWriter

from pkm.agents.profile import AgentProfile
from pkm.data import Deck

from .model import PolicyValueNet
from .ppo import compute_returns, ppo_update
from .reward_terms import DEFAULT_WEIGHTS, load_weights, write_default_weights_file
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


def evaluate_vs_agent(
    model: PolicyValueNet,
    deck: list[int],
    opponent_deck: list[int],
    opponent_checkpoint: str,
    games: int = 20,
) -> float:
    """Win rate of the greedy policy vs. another trained agent's greedy
    policy, each playing its own deck, alternating sides. Reloads the
    opponent checkpoint fresh every call, so this stays current if that
    agent is itself still training.
    """
    opponent_model = PolicyValueNet()
    opponent_model.load_state_dict(
        torch.load(opponent_checkpoint, map_location="cpu", weights_only=True)
    )
    opponent_model.eval()
    policy = TorchPolicy(model, greedy=True)
    opp_policy = TorchPolicy(opponent_model, greedy=True)
    wins = 0.0
    for g in range(games):
        side = g % 2
        policies = (policy, opp_policy) if side == 0 else (opp_policy, policy)
        decks = (deck, opponent_deck) if side == 0 else (opponent_deck, deck)
        result = play_game(policies, decks, collect=(False, False))
        r = result.rewards[side]
        wins += 1.0 if r > 0 else 0.5 if r == 0 else 0.0
    return wins / games


def play_vs_fixed_opponent(
    model: PolicyValueNet,
    deck: list[int],
    opponent_model: PolicyValueNet,
    opponent_deck: list[int],
    games: int,
    gamma: float,
    lam: float,
    weights: dict[str, float],
    win_reward: float = 1.0,
) -> tuple[list, int, int, int, int]:
    """Generate one iteration's training data entirely from games against a
    frozen opponent -- no self-play. The opponent plays its own deck
    greedily and is never trained on; only the trainee's trajectory is
    collected. Alternates sides each game. Returns (data, wins, losses,
    draws, total_decisions) in the same shape the self-play path produces,
    so the rest of the training loop doesn't need to know which mode ran.
    """
    trainee = TorchPolicy(model)
    opponent = TorchPolicy(opponent_model, greedy=True)
    data: list = []
    w = losses = d = 0
    total_decisions = 0
    for g in range(games):
        side = g % 2
        policies = (trainee, opponent) if side == 0 else (opponent, trainee)
        decks = (deck, opponent_deck) if side == 0 else (opponent_deck, deck)
        result = play_game(
            policies, decks, collect=(True, False) if side == 0 else (False, True)
        )
        total_decisions += result.decisions
        traj = result.trajectories[side]
        r = result.rewards[side]
        compute_returns(
            traj, r, gamma=gamma, lam=lam, weights=weights, win_reward=win_reward
        )
        data.extend(traj)
        w += 1 if r > 0 else 0
        losses += 1 if r < 0 else 0
        d += 1 if r == 0 else 0
    return data, w, losses, d, total_decisions


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
    "eval_vs_agent_win_rate",
]


def train(
    deck_path: str = "deck/02_dragapult.csv",
    iterations: int = 50,
    games_per_iter: int = 8,
    lr: float = 3e-4,
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
    workers: int = 1,
    eval_vs_agent_name: str | None = None,
    eval_vs_agent_deck: list[int] | None = None,
    eval_vs_agent_checkpoint: str | None = None,
    vs_agent_name: str | None = None,
    vs_agent_deck: list[int] | None = None,
    vs_agent_checkpoint: str | None = None,
    win_reward: float = 1.0,
) -> PolicyValueNet:
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    effective_weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    deck = Deck.from_csv(deck_path).card_ids
    model = PolicyValueNet()
    if init_checkpoint:
        model.load_state_dict(
            torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        )
        print(f"resumed from {init_checkpoint}", flush=True)
    if vs_agent_deck is not None:
        print(
            f"training vs fixed opponent '{vs_agent_name}' only -- no self-play",
            flush=True,
        )
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
    if workers > 1 and vs_agent_deck is None:
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

            if vs_agent_deck is not None and vs_agent_checkpoint:
                opponent_model.load_state_dict(
                    torch.load(
                        vs_agent_checkpoint, map_location="cpu", weights_only=True
                    )
                )
                opponent_model.eval()
                data, w, losses, d, total_decisions = play_vs_fixed_opponent(
                    model,
                    deck,
                    opponent_model,
                    vs_agent_deck,
                    games_per_iter,
                    gamma,
                    lam,
                    effective_weights,
                    win_reward=win_reward,
                )
            else:
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
                        spec,
                        result,
                        data,
                        gamma,
                        lam,
                        weights=effective_weights,
                        win_reward=win_reward,
                    )
                    w, losses, d = w + gw, losses + gl, d + gd

            model.train()
            stats = ppo_update(model, optimizer, data)
            model.eval()

            if vs_agent_deck is None:
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
                "eval_vs_agent_win_rate": "",
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

                if eval_vs_agent_deck is not None and eval_vs_agent_checkpoint:
                    wr2 = evaluate_vs_agent(
                        model,
                        deck,
                        eval_vs_agent_deck,
                        eval_vs_agent_checkpoint,
                        games=eval_games,
                    )
                    row["eval_vs_agent_win_rate"] = f"{wr2:.4f}"
                    print(
                        f"iter {it:3d} | eval vs {eval_vs_agent_name}: "
                        f"{wr2:.1%} ({eval_games} games)",
                        flush=True,
                    )
                    tb.add_scalar(f"eval/win_rate_vs_{eval_vs_agent_name}", wr2, it)

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
        help="path to a JSON file of {term: weight} overrides -- see "
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
    workers: int = typer.Option(
        1, help="parallel worker processes for self-play rollout (1 = sequential)"
    ),
    eval_vs: str | None = typer.Option(
        None,
        "--eval-vs",
        help="another agent profile name -- every --eval-every iterations, also "
        "report (not train on) the greedy win rate against that agent's latest "
        "checkpoint, each side playing its own deck. Reloaded fresh each time, "
        "so it stays current if that agent is itself still training.",
    ),
    vs_agent: str | None = typer.Option(
        None,
        "--vs-agent",
        help="another agent profile name -- train entirely against that agent's "
        "latest checkpoint (its own deck, played greedily) instead of self-play. "
        "Reloaded fresh every iteration, so it tracks that agent's progress if "
        "it's training concurrently. Mutually exclusive with the self-play pool "
        "(--pool-size/--pool-prob are ignored) and --workers > 1.",
    ),
    win_reward: float = typer.Option(
        1.0,
        "--win-reward",
        help="scales the terminal reward for a win only (losses/draws untouched) "
        "-- makes winning matter more relative to shaping and losing.",
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
        if weights is None:
            write_default_weights_file(profile.reward_weights_path)
            weights = str(profile.reward_weights_path)

    eval_vs_agent_deck = None
    eval_vs_agent_checkpoint = None
    if eval_vs:
        eval_profile = AgentProfile(eval_vs)
        eval_vs_agent_deck = Deck.from_csv(eval_profile.deck_path).card_ids
        eval_vs_agent_checkpoint = eval_profile.ppo_init()
        if eval_vs_agent_checkpoint is None:
            raise typer.BadParameter(
                f"--eval-vs {eval_vs!r} has no checkpoints/ppo_latest.pt yet"
            )

    vs_agent_deck = None
    vs_agent_checkpoint = None
    if vs_agent:
        if workers > 1:
            raise typer.BadParameter("--vs-agent doesn't support --workers > 1 yet")
        vs_profile = AgentProfile(vs_agent)
        vs_agent_deck = Deck.from_csv(vs_profile.deck_path).card_ids
        vs_agent_checkpoint = vs_profile.ppo_init()
        if vs_agent_checkpoint is None:
            raise typer.BadParameter(
                f"--vs-agent {vs_agent!r} has no checkpoints/ppo_latest.pt yet"
            )

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
        eval_vs_agent_name=eval_vs,
        eval_vs_agent_deck=eval_vs_agent_deck,
        eval_vs_agent_checkpoint=eval_vs_agent_checkpoint,
        vs_agent_name=vs_agent,
        vs_agent_deck=vs_agent_deck,
        vs_agent_checkpoint=vs_agent_checkpoint,
        seed=seed,
        workers=workers,
        win_reward=win_reward,
    )


if __name__ == "__main__":
    app()
