"""Phase 1 training entry point: PPO self-play with an opponent checkpoint pool.

Usage:
    python -m pkm.rl.train --iterations 50 --games 8
"""

import argparse
import copy
import csv
import random
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from pkm.data import Deck

from .model import PolicyValueNet
from .ppo import compute_returns, ppo_update
from .rollout import RandomPolicy, TorchPolicy, play_game


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
    deck_path: str = "deck.csv",
    iterations: int = 50,
    games_per_iter: int = 8,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    shaping_coef: float = 0.2,
    pool_size: int = 8,
    pool_prob: float = 0.4,
    eval_every: int = 5,
    eval_games: int = 20,
    checkpoint_dir: str = "checkpoints",
    metrics_path: str = "metrics/ppo_train.csv",
    log_dir: str = "runs/ppo",
    init_checkpoint: str | None = None,
    seed: int = 0,
) -> PolicyValueNet:
    random.seed(seed)
    torch.manual_seed(seed)

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

    for it in range(1, iterations + 1):
        t0 = time.time()
        model.eval()
        current = TorchPolicy(model)
        data = []
        w = losses = d = 0
        total_decisions = 0

        for _ in range(games_per_iter):
            if random.random() < pool_prob and len(pool) > 1:
                opponent_model.load_state_dict(random.choice(pool[:-1]))
                opp = TorchPolicy(opponent_model)
                side = random.randint(0, 1)
                policies = (current, opp) if side == 0 else (opp, current)
                collect = (side == 0, side == 1)
            else:
                side = -1  # mirror match: collect both sides
                policies = (current, current)
                collect = (True, True)

            result = play_game(policies, (deck, deck), collect=collect)
            total_decisions += result.decisions
            for p in range(2):
                if not collect[p]:
                    continue
                compute_returns(
                    result.trajectories[p],
                    result.rewards[p],
                    gamma=gamma,
                    lam=lam,
                    shaping_coef=shaping_coef,
                )
                data.extend(result.trajectories[p])
                if side == -1 and p == 1:
                    continue  # count mirror games once
                r = result.rewards[p if side == -1 else side]
                w, losses, d = w + (r > 0), losses + (r < 0), d + (r == 0)

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
            tb.add_scalar("eval/win_rate_vs_random", float(row["eval_win_rate"]), it)

    torch.save(model.state_dict(), ckpt_dir / "ppo_latest.pt")
    csv_f.close()
    tb.close()
    print(f"metrics saved to {metrics_file}", flush=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="deck.csv")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--shaping", type=float, default=0.2)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--metrics", default="metrics/ppo_train.csv")
    parser.add_argument("--log-dir", default="runs/ppo")
    parser.add_argument(
        "--init",
        default=None,
        help="checkpoint to resume from (e.g. checkpoints/ppo_latest.pt)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(
        deck_path=args.deck,
        iterations=args.iterations,
        games_per_iter=args.games,
        lr=args.lr,
        gamma=args.gamma,
        shaping_coef=args.shaping,
        pool_size=args.pool_size,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=args.metrics,
        log_dir=args.log_dir,
        init_checkpoint=args.init,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
