"""Evolution loop for `default_dragapult_darwinian`.

Run it:
    python -m darwinian_ml.evolve --generations 50
    darwinian_ml/evolve.sh                     # background, stoppable
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import typer

from pkm.data import Deck
from pkm.rl.model import PolicyValueNet

from .config import DarwinConfig
from .fitness import Running, evaluate, flatten, genome_shapes, load_genome
from .opponent import BundleOpponent, extract_bundle
from .population import next_generation

CSV_FIELDS = [
    "generation",
    "best_score",
    "mean_score",
    "best_win_rate",
    "pop_win_rate",
    "best_games",
    "best_prize_margin",
    "best_shaping",
    "sigma",
    "seconds",
]


def _seed_population(cfg: DarwinConfig, model: PolicyValueNet, shapes, rng):
    """Start from a trained policy, jittered -- not from noise.

    342k parameters is far past what a genetic algorithm can discover from
    scratch in any realistic number of games. Seeding from the existing
    agent turns this into *directed variation around a working policy*,
    which is the only form of the experiment that can produce a result.
    """
    base = flatten(model)
    if cfg.seed_weights and Path(cfg.seed_weights).is_file():
        try:
            npz = np.load(cfg.seed_weights)
            sd = model.state_dict()
            loaded = 0
            for name, shape, _ in shapes:
                if name in npz.files and tuple(npz[name].shape) == shape:
                    sd[name].copy_(torch.from_numpy(npz[name]))
                    loaded += 1
            model.load_state_dict(sd)
            base = flatten(model)
            print(f"seeded from {cfg.seed_weights} ({loaded}/{len(shapes)} tensors)")
        except Exception as exc:
            print(f"could not seed from {cfg.seed_weights}: {exc}; using fresh init")
    else:
        print("no seed weights found -- starting from a fresh random network")

    from .population import per_tensor_scale

    scale = per_tensor_scale(base, shapes)
    pop = [base.copy()]  # generation 0 contains the unmodified seed
    while len(pop) < cfg.population:
        noise = rng.normal(0.0, 1.0, base.shape).astype(np.float32)
        pop.append((base + noise * cfg.sigma * scale).astype(np.float32))
    return pop


def evolve(cfg: DarwinConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    out = Path(cfg.out_dir)
    bundle_dir = extract_bundle(cfg.bundle, out / "opponent_bundle")
    deck = Deck.from_csv(cfg.deck_path).card_ids

    model = PolicyValueNet()
    shapes = genome_shapes(model)
    pop = _seed_population(cfg, model, shapes, rng)
    # One accumulator per genome, carried forward for survivors so their
    # estimates sharpen instead of being re-rolled every generation.
    running = [Running() for _ in pop]
    scratch = PolicyValueNet()

    csv_path = out / "evolution.csv"
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    writer.writeheader()

    best_ever = -1e9
    sigma = cfg.sigma
    stop_file = out / "STOP"
    stop_file.unlink(missing_ok=True)
    deadline = time.time() + cfg.max_hours * 3600 if cfg.max_hours else None

    print(
        f"darwinian: population={cfg.population} elites={cfg.elites} "
        f"games/genome={cfg.games_per_genome} vs {Path(cfg.bundle).name}"
    )
    with BundleOpponent(bundle_dir) as opponent:
        print(f"opponent bundle live, deck={len(opponent.deck)} cards")
        for gen in range(1, cfg.generations + 1):
            if stop_file.exists():
                print(f"stop file found after {gen - 1} generations")
                break
            if deadline is not None and time.time() > deadline:
                print(
                    f"time budget of {cfg.max_hours}h reached at generation {gen - 1}"
                )
                break
            t0 = time.time()
            cfg.sigma = sigma
            results = [evaluate(g, scratch, opponent, deck, cfg, shapes) for g in pop]
            for acc, res in zip(running, results):
                acc.add(res)
            # Rank on the *cumulative* estimate, not this batch alone.
            scores = np.array([acc.score(cfg) for acc in running], dtype=np.float64)
            best_i = int(np.argmax(scores))
            best = results[best_i]
            best_acc = running[best_i]

            row = {
                "generation": gen,
                "best_score": float(scores[best_i]),
                "mean_score": float(scores.mean()),
                "best_win_rate": best_acc.win_rate,
                "pop_win_rate": float(np.mean([r.win_rate for r in results])),
                "best_games": best_acc.games,
                "best_prize_margin": best.prize_margin,
                "best_shaping": best.shaping,
                "sigma": sigma,
                "seconds": time.time() - t0,
            }
            writer.writerow(row)
            fh.flush()
            print(
                f"gen {gen:>4} | best {best.score:+7.2f} (mean {scores.mean():+7.2f}) "
                f"| win {best_acc.win_rate:.0%} over {best_acc.games}g "
                f"(pop {row['pop_win_rate']:.0%}) "
                f"| prizes {best.prize_margin:+.2f} | sigma {sigma:.4f} "
                f"| {row['seconds']:.0f}s",
                flush=True,
            )

            if best.score > best_ever:
                best_ever = best.score
                load_genome(model, pop[best_i], shapes)
                torch.save(model.state_dict(), out / "best.pt")
                (out / "best.json").write_text(
                    json.dumps({"generation": gen, **row}, indent=2)
                )

            ranked = list(np.argsort(-scores))
            pop = next_generation(pop, scores, shapes, cfg, rng)
            running = [running[i] for i in ranked[: cfg.elites]] + [
                Running() for _ in range(len(pop) - cfg.elites)
            ]
            sigma = max(cfg.sigma_min, sigma * cfg.sigma_decay)

    fh.close()
    print(f"done. best checkpoint: {out / 'best.pt'}")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    generations: int = typer.Option(200),
    population: int = typer.Option(12),
    games: int = typer.Option(4, help="games per genome (sides alternate)"),
    elites: int = typer.Option(3),
    sigma: float = typer.Option(0.02, help="mutation scale, relative to weight std"),
    bundle: str = typer.Option(DarwinConfig.bundle),
    deck: str = typer.Option(DarwinConfig.deck_path),
    seed_weights: str = typer.Option(
        DarwinConfig.seed_weights, help="policy .npz to seed the population from"
    ),
    out_dir: str = typer.Option(DarwinConfig.out_dir),
    max_hours: float = typer.Option(
        0.0, help="stop after this many hours (0 = run until generations/STOP)"
    ),
    seed: int = typer.Option(0),
) -> None:
    evolve(
        DarwinConfig(
            bundle=bundle,
            deck_path=deck,
            population=population,
            elites=elites,
            games_per_genome=games,
            generations=generations,
            sigma=sigma,
            seed_weights=seed_weights or None,
            out_dir=out_dir,
            max_hours=max_hours,
            seed=seed,
        )
    )


if __name__ == "__main__":
    app()
