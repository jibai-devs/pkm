"""Phase 2 training: expert iteration (AlphaZero-style).

Self-play where both players choose moves by IS-MCTS; the network is then
trained toward the search's visit distribution (policy) and the game outcome
(value). Stronger net -> better priors/leaf evals -> stronger search -> better
targets.

Usage:
    pkm exit-train --iterations 3 --games 4 --sims 24 --dets 2
    pkm exit-train --agent 01_psychic --iterations 5
"""

import typer
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from kaggle_environments.envs.cabt.cg.game import (
    battle_finish,
    battle_select,
    battle_start,
)

from pkm.agents.profile import AgentProfile, TrainingResult
from pkm.data import Deck
from pkm.mcts.search import MCTS, _forced_picks
from pkm.rl.encoder import EncodedDecision, encode_decision
from pkm.rl.model import OPT_ENC, PolicyValueNet
from pkm.rl.numpy_policy import NumpyPolicy
from pkm.rl.rollout import MAX_DECISIONS


class ExitSample:
    __slots__ = ("decision", "target_dist", "z")

    def __init__(self, decision: EncodedDecision, target_dist: np.ndarray | None):
        self.decision = decision  # picks/stopped hold the MCTS-chosen sequence
        self.target_dist = target_dist  # (n+1,) over options + STOP, or None
        self.z = 0.0  # outcome from the mover's perspective


def _numpy_policy_from(model: PolicyValueNet) -> NumpyPolicy:
    return NumpyPolicy({k: v.detach().numpy() for k, v in model.state_dict().items()})


def _visit_target(agg: dict, n: int) -> np.ndarray | None:
    """Visit counts -> distribution over (n options + STOP); None for multi-pick."""
    if not agg or any(len(a) > 1 for a in agg):
        return None
    t = np.zeros(n + 1, dtype=np.float32)
    for a, count in agg.items():
        t[a[0] if a else n] += count
    s = t.sum()
    return t / s if s > 0 else None


def play_exit_game(
    mcts_pair: tuple[MCTS, MCTS],
    decks: tuple[list[int], list[int]],
    temp_decisions: int = 30,
    rng: random.Random | None = None,
) -> tuple[tuple[list[ExitSample], list[ExitSample]], tuple[float, float]]:
    """One MCTS-vs-MCTS game; returns per-player samples and rewards."""
    rng = rng or random.Random()
    obs, start = battle_start(list(decks[0]), list(decks[1]))
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorPlayer={start.errorPlayer}")

    samples: tuple[list[ExitSample], list[ExitSample]] = ([], [])
    count = 0
    try:
        while obs["current"]["result"] < 0 and count < MAX_DECISIONS:
            p = obs["current"]["yourIndex"]
            forced = _forced_picks(obs["select"])
            if forced is not None:
                obs = battle_select(forced)
                count += 1
                continue

            mcts = mcts_pair[p]
            picks, agg = mcts.choose(obs, decks[p], decks[1 - p])
            n = len(obs["select"]["option"])
            target = _visit_target(agg, n)

            # temperature: sample proportional to visits early in the game
            if agg and count < temp_decisions:
                actions = list(agg.keys())
                weights = [agg[a] for a in actions]
                picks = list(rng.choices(actions, weights=weights)[0])

            d = encode_decision(obs)
            d.picks = list(picks)
            d.stopped = len(picks) < d.max_count
            samples[p].append(ExitSample(d, target))

            obs = battle_select(picks)
            count += 1
        result = obs["current"]["result"]
    finally:
        battle_finish()

    rewards = (1.0, -1.0) if result == 0 else (-1.0, 1.0) if result == 1 else (0.0, 0.0)
    for p in range(2):
        for s in samples[p]:
            s.z = rewards[p]
    return samples, rewards


def _first_step_logprobs(model: PolicyValueNet, d: EncodedDecision) -> torch.Tensor:
    """Log-probs over (options + STOP) for the first pick of a decision."""
    board = torch.from_numpy(d.board_cards).unsqueeze(0)
    hand = torch.from_numpy(d.hand_cards).unsqueeze(0)
    feats = torch.from_numpy(d.state_feats).unsqueeze(0)
    h = model.encode_state(board, hand, feats)
    opts = model.encode_options(
        torch.from_numpy(d.opt_type).unsqueeze(0),
        torch.from_numpy(d.opt_card).unsqueeze(0),
        torch.from_numpy(d.opt_card2).unsqueeze(0),
        torch.from_numpy(d.opt_attack).unsqueeze(0),
        torch.from_numpy(d.opt_feats).unsqueeze(0),
    )
    n = len(d.opt_type)
    mask = torch.ones(1, n + 1, dtype=torch.bool)
    mask[0, n] = d.min_count == 0
    logits = model.option_logits(h, opts, torch.zeros(1, OPT_ENC), mask)
    value = model.value(h)
    return F.log_softmax(logits, dim=-1)[0], value[0]


def exit_update(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    data: list[ExitSample],
    epochs: int = 2,
    minibatch: int = 64,
    value_coef: float = 1.0,
) -> dict[str, float]:
    stats = {"policy_loss": 0.0, "value_loss": 0.0}
    n_batches = 0
    for _ in range(epochs):
        order = np.random.permutation(len(data))
        for start in range(0, len(data), minibatch):
            batch = [data[i] for i in order[start : start + minibatch]]
            policy_loss = torch.zeros(())
            value_loss = torch.zeros(())
            n_pol = 0
            seq_samples = [s for s in batch if s.target_dist is None]
            for s in batch:
                if s.target_dist is not None:
                    logp, v = _first_step_logprobs(model, s.decision)
                    target = torch.from_numpy(s.target_dist)
                    policy_loss = policy_loss - (target * logp).sum()
                    n_pol += 1
                    value_loss = value_loss + (v - s.z) ** 2
            if seq_samples:
                # multi-pick: behavior-clone the MCTS-chosen sequence
                lp, _, vals = model.evaluate([s.decision for s in seq_samples])
                policy_loss = policy_loss - lp.sum()
                n_pol += len(seq_samples)
                zs = torch.tensor([s.z for s in seq_samples], dtype=torch.float32)
                value_loss = value_loss + ((vals - zs) ** 2).sum()

            policy_loss = policy_loss / max(n_pol, 1)
            value_loss = value_loss / len(batch)
            loss = policy_loss + value_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in stats.items()}


EXIT_CSV_FIELDS = [
    "iter",
    "games",
    "p0_wins",
    "p0_losses",
    "samples",
    "pi_loss",
    "v_loss",
    "time_s",
]


def train(
    deck_path: str = "deck/02_dragapult.csv",
    iterations: int = 3,
    games_per_iter: int = 4,
    n_simulations: int = 24,
    n_determinizations: int = 2,
    lr: float = 1e-4,
    init_checkpoint: str = "checkpoints/ppo_latest.pt",
    checkpoint_dir: str = "checkpoints",
    metrics_path: str = "metrics/exit_train.csv",
    log_dir: str = "runs/exit",
    seed: int = 0,
    checkpoint_path: str | None = None,
) -> PolicyValueNet:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    deck = Deck.from_csv(deck_path).card_ids
    model = PolicyValueNet()
    if Path(init_checkpoint).is_file():
        model.load_state_dict(
            torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        )
        print(f"initialized from {init_checkpoint}")
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    output_checkpoint = (
        Path(checkpoint_path) if checkpoint_path else ckpt_dir / "exit_latest.pt"
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(metrics_file, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=EXIT_CSV_FIELDS)
    csv_w.writeheader()

    tb = SummaryWriter(log_dir)

    for it in range(1, iterations + 1):
        t0 = time.time()
        policy = _numpy_policy_from(model)
        data: list[ExitSample] = []
        w = losses = 0
        for g in range(games_per_iter):
            mcts_pair = tuple(
                MCTS(
                    policy,
                    n_determinizations=n_determinizations,
                    n_simulations=n_simulations,
                    dirichlet_eps=0.25,
                    rng=random.Random(seed * 10_000 + it * 100 + g * 2 + i),
                )
                for i in range(2)
            )
            samples, rewards = play_exit_game(mcts_pair, (deck, deck))
            data.extend(samples[0])
            data.extend(samples[1])
            w += rewards[0] > 0
            losses += rewards[0] < 0

        model.train()
        stats = exit_update(model, optimizer, data)
        model.eval()
        dt = time.time() - t0
        csv_w.writerow(
            {
                "iter": it,
                "games": games_per_iter,
                "p0_wins": w,
                "p0_losses": losses,
                "samples": len(data),
                "pi_loss": f"{stats['policy_loss']:.6f}",
                "v_loss": f"{stats['value_loss']:.6f}",
                "time_s": f"{dt:.2f}",
            }
        )
        csv_f.flush()
        tb.add_scalar("loss/policy", stats["policy_loss"], it)
        tb.add_scalar("loss/value", stats["value_loss"], it)
        tb.add_scalar("game/p0_win_rate", w / (w + losses) if (w + losses) else 0, it)
        tb.add_scalar("time/iter_s", dt, it)
        print(
            f"exit iter {it} | games {games_per_iter} (p0 W/L {w}/{losses}) "
            f"| samples {len(data)} | pi_loss {stats['policy_loss']:.4f} "
            f"| v_loss {stats['value_loss']:.4f} | {dt:.1f}s",
            flush=True,
        )
        torch.save(model.state_dict(), output_checkpoint)

    csv_f.close()
    tb.close()
    print(f"metrics saved to {metrics_file}", flush=True)
    return model


def train_profile(
    *,
    deck_path: Path,
    checkpoint_path: Path,
    checkpoint_dir: Path,
    metrics_dir: Path,
    runs_dir: Path,
    resume_path: Path | None,
    iterations: int = 3,
    games_per_iter: int = 4,
    n_simulations: int = 24,
    n_determinizations: int = 2,
    lr: float = 1e-4,
    seed: int = 0,
    **kwargs: object,
) -> TrainingResult:
    """Profile-facing expert-iteration facade; the legacy ``train`` is unchanged."""
    train(
        deck_path=str(deck_path),
        iterations=iterations,
        games_per_iter=games_per_iter,
        n_simulations=n_simulations,
        n_determinizations=n_determinizations,
        lr=lr,
        init_checkpoint=str(resume_path) if resume_path else "",
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_dir / "exit_train.csv"),
        log_dir=str(runs_dir / "exit"),
        seed=seed,
        **kwargs,
    )
    return TrainingResult(
        checkpoint=checkpoint_path,
        metrics=metrics_dir / "exit_train.csv",
        iterations=iterations,
    )


app = typer.Typer(help=__doc__)


@app.command()
def main(
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
) -> None:
    if agent:
        AgentProfile.load(agent).train_exit(
            iterations=iterations,
            games=games,
            n_simulations=sims,
            n_determinizations=dets,
            lr=lr,
            seed=seed,
            resume_path=Path(init) if init != "checkpoints/ppo_latest.pt" else None,
        )
        return
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


if __name__ == "__main__":
    app()
