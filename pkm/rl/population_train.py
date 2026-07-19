"""Simultaneous population training (Milestone 9,
docs/opponent-archetype-classifier-plan.md Part 3 SS3b+3c).

The anchor (e.g. 03_pult_munki) and every Part 3b pool bot train together
from shared games, each side updating its own live policy from that game's
outcome -- not a frozen-checkpoint opponent pool (that's pkm/rl/train.py's
`--archetype-pool`, which stays exactly as-is; this is additive, not a
replacement).

Usage:
    pkm population-train --iterations 100 --games-per-pairing 2
"""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import typer

from pkm.agents.profile import AgentProfile
from pkm.data import Deck

from .encoder import EncodedDecision
from .features import archetype_index, write_stamp_sidecar
from .model import PolicyValueNet
from .ppo import compute_returns, ppo_update
from .reward_terms import load_weights, write_default_weights_file
from .rollout import GameResult, TorchPolicy, play_game
from .train import evaluate_vs_random

CSV_FIELDS = [
    "iter",
    "games",
    "wins",
    "losses",
    "draws",
    "samples",
    "pi_loss",
    "v_loss",
    "entropy",
    "clip_frac",
    "archetype_loss",
    "eval_win_rate",
    "eval_games",
]


@dataclass
class PopulationMember:
    name: str
    deck: list[int]
    model: PolicyValueNet
    optimizer: torch.optim.Optimizer
    weights: dict[str, float]
    archetype_label: int
    profile: AgentProfile
    buffer: list[EncodedDecision] = field(default_factory=list)
    iters_since_update: int = 0


@dataclass
class PopSpec:
    """One game's matchup by roster index -- population training's own
    concept, deliberately not GameSpec (that stays exactly as-is for
    train.py's frozen-checkpoint-pool path). Never built with
    member_a_idx == member_b_idx; bot-vs-bot games are out of scope for v1
    (see docs/opponent-archetype-classifier-plan.md SS3b+3c)."""

    member_a_idx: int
    member_b_idx: int
    collect: tuple[bool, bool]


def _load_member(name: str, lr: float) -> PopulationMember:
    profile = AgentProfile(name)
    profile.ensure_dirs()
    deck = Deck.from_csv(str(profile.deck_path)).card_ids
    model = PolicyValueNet()
    init = profile.ppo_init()
    if init:
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=True))
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    write_default_weights_file(profile.reward_weights_path)
    weights = load_weights(str(profile.reward_weights_path))
    archetype_label = archetype_index(str(profile.deck_path))
    return PopulationMember(
        name=name,
        deck=deck,
        model=model,
        optimizer=optimizer,
        weights=weights,
        archetype_label=archetype_label,
        profile=profile,
    )


def load_roster(
    anchor: str = "03_pult_munki",
    pool_glob: str = "pool_*",
    agents_dir: str = "agents",
    lr: float = 3e-4,
) -> list[PopulationMember]:
    """Roster[0] is always the anchor; the rest are every `agents_dir`
    profile matching `pool_glob` that has a matching `deck/<name>.csv` --
    same "skip untrained/unresolvable, don't error" convention as
    `pkm.rl.opponent_pool.load_pool_bots`. Each member resumes from its own
    `ppo_latest.pt` if one exists (Part 3b's solo-trained checkpoint for
    pool bots), continuing that training rather than discarding it."""
    roster = [_load_member(anchor, lr)]
    for profile_dir in sorted(Path(agents_dir).glob(pool_glob)):
        name = profile_dir.name
        if not (Path("deck") / f"{name}.csv").is_file():
            continue
        roster.append(_load_member(name, lr))
    return roster


def make_pop_specs(games_per_pairing: int, num_members: int) -> list[PopSpec]:
    """The anchor (roster index 0) plays `games_per_pairing` games against
    EVERY other roster member each iteration, sides alternated
    deterministically (not randomly) so every pairing is exactly balanced.
    Both sides are always collected -- the whole point of population
    training vs. train.py's frozen pool is that both sides learn."""
    specs = []
    for bot_idx in range(1, num_members):
        for g in range(games_per_pairing):
            if g % 2 == 0:
                specs.append(PopSpec(0, bot_idx, (True, True)))
            else:
                specs.append(PopSpec(bot_idx, 0, (True, True)))
    return specs


def _bucket_result(
    roster: list[PopulationMember],
    spec: PopSpec,
    result: GameResult,
    gamma: float,
    lam: float,
    stats: dict[str, list[int]],
) -> None:
    """Route each side's trajectory into ITS OWN member's buffer -- never
    the opponent's -- with that member's own reward weights and archetype
    label, and tally that member's win/loss/draw for this game."""
    for member_idx, side in ((spec.member_a_idx, 0), (spec.member_b_idx, 1)):
        if not spec.collect[side]:
            continue
        member = roster[member_idx]
        traj = result.trajectories[side]
        compute_returns(
            traj, result.rewards[side], gamma=gamma, lam=lam, weights=member.weights
        )
        for dec in traj:
            dec.true_archetype = member.archetype_label
        member.buffer.extend(traj)
        r = result.rewards[side]
        row = stats.setdefault(member.name, [0, 0, 0])
        row[0 if r > 0 else 1 if r < 0 else 2] += 1


def run_pop_iteration(
    roster: list[PopulationMember],
    specs: list[PopSpec],
    gamma: float,
    lam: float,
    executor=None,
    workers: int = 1,
) -> dict[str, tuple[int, int, int]]:
    """Play every spec (sequentially, or across `executor` if given) and
    bucket trajectories into each member's own buffer. Returns per-member
    (wins, losses, draws) for this iteration -- a member absent from the
    dict didn't play (shouldn't happen for a well-formed spec list, but
    callers shouldn't assume every roster member is a key)."""
    for m in roster:
        m.model.eval()

    if executor is not None:
        from .parallel_rollout import collect_pop_parallel

        games = [
            (
                roster[s.member_a_idx].deck,
                roster[s.member_a_idx].model.state_dict(),
                roster[s.member_b_idx].deck,
                roster[s.member_b_idx].model.state_dict(),
                s.collect,
            )
            for s in specs
        ]
        results = collect_pop_parallel(executor, workers, games)
    else:
        results = [
            play_game(
                (
                    TorchPolicy(roster[s.member_a_idx].model),
                    TorchPolicy(roster[s.member_b_idx].model),
                ),
                (roster[s.member_a_idx].deck, roster[s.member_b_idx].deck),
                collect=s.collect,
            )
            for s in specs
        ]

    stats: dict[str, list[int]] = {}
    for spec, result in zip(specs, results):
        _bucket_result(roster, spec, result, gamma, lam, stats)
    return {name: tuple(v) for name, v in stats.items()}


def population_train(
    iterations: int = 100,
    games_per_pairing: int = 2,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    update_every: int = 3,
    min_samples: int = 512,
    anchor: str = "03_pult_munki",
    pool_glob: str = "pool_*",
    agents_dir: str = "agents",
    eval_every: int = 10,
    eval_games: int = 20,
    workers: int = 1,
    seed: int = 0,
) -> list[PopulationMember]:
    """Milestone 9's orchestration loop. Reuses `play_game`/`ppo_update`
    unchanged (`pkm/rl/rollout.py`, `pkm/rl/ppo.py`) and each member's own
    `AgentProfile` dirs for checkpoints -- no new directory convention.
    Metrics land in a *separate* `population_train.csv` per member
    (`profile.metrics_dir`), so a pool bot's Part 3b solo-training history
    in `ppo_train.csv` is never overwritten by this run."""
    random.seed(seed)
    torch.manual_seed(seed)

    roster = load_roster(
        anchor=anchor, pool_glob=pool_glob, agents_dir=agents_dir, lr=lr
    )
    if len(roster) < 2:
        raise ValueError(
            f"population_train needs at least one pool bot besides the anchor "
            f"(found {len(roster) - 1} matching {agents_dir}/{pool_glob})"
        )

    csv_files = {}
    csv_writers = {}
    for m in roster:
        f = open(m.profile.metrics_dir / "population_train.csv", "w", newline="")
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        csv_files[m.name] = f
        csv_writers[m.name] = w

    executor = None
    if workers > 1:
        from .parallel_rollout import make_pool

        executor = make_pool(workers)
        print(f"parallel rollout: {workers} worker processes", flush=True)

    try:
        for it in range(1, iterations + 1):
            specs = make_pop_specs(games_per_pairing, len(roster))
            game_stats = run_pop_iteration(
                roster, specs, gamma, lam, executor=executor, workers=workers
            )

            summary_parts = []
            for m in roster:
                m.iters_since_update += 1
                w, losses, d = game_stats.get(m.name, (0, 0, 0))
                row = {f: "" for f in CSV_FIELDS}
                row["iter"] = it
                row["games"] = w + losses + d
                row["wins"], row["losses"], row["draws"] = w, losses, d

                ready = bool(m.buffer) and (
                    len(m.buffer) >= min_samples or m.iters_since_update >= update_every
                )
                if ready:
                    m.model.train()
                    pstats = ppo_update(m.model, m.optimizer, m.buffer)
                    m.model.eval()
                    row["samples"] = len(m.buffer)
                    row["pi_loss"] = f"{pstats['policy_loss']:.6f}"
                    row["v_loss"] = f"{pstats['value_loss']:.6f}"
                    row["entropy"] = f"{pstats['entropy']:.6f}"
                    row["clip_frac"] = f"{pstats['clip_frac']:.4f}"
                    row["archetype_loss"] = f"{pstats['archetype_loss']:.6f}"
                    m.buffer = []
                    m.iters_since_update = 0

                if it % eval_every == 0:
                    wr = evaluate_vs_random(m.model, m.deck, games=eval_games)
                    row["eval_win_rate"] = f"{wr:.4f}"
                    row["eval_games"] = eval_games
                    torch.save(
                        m.model.state_dict(),
                        m.profile.checkpoint_dir / f"ppo_iter{it:04d}.pt",
                    )
                    torch.save(
                        m.model.state_dict(), m.profile.checkpoint_dir / "ppo_latest.pt"
                    )
                    write_stamp_sidecar(m.profile.checkpoint_dir / "ppo_latest.pt")
                    summary_parts.append(f"{m.name}={wr:.0%}")

                csv_writers[m.name].writerow(row)
                csv_files[m.name].flush()

            anchor_w, anchor_l, anchor_d = game_stats.get(roster[0].name, (0, 0, 0))
            line = (
                f"iter {it:4d} | anchor {roster[0].name} W/L/D "
                f"{anchor_w}/{anchor_l}/{anchor_d}"
            )
            if summary_parts:
                line += " | eval: " + ", ".join(summary_parts)
            print(line, flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
        for m in roster:
            torch.save(m.model.state_dict(), m.profile.checkpoint_dir / "ppo_latest.pt")
            write_stamp_sidecar(m.profile.checkpoint_dir / "ppo_latest.pt")
        for f in csv_files.values():
            f.close()

    return roster


app = typer.Typer(help=__doc__)


@app.command()
def main(
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
    population_train(
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


if __name__ == "__main__":
    app()
