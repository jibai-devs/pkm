"""The evolutionary operators: who survives, who breeds, what mutates.

Deliberately the plainest possible generational GA -- truncation elitism,
tournament selection, uniform crossover, gaussian mutation -- because the
point of the experiment is the *training signal* (beat a real opponent), not
a clever optimiser. Anything more elaborate would confound the comparison
against the gradient-based agent.
"""

from __future__ import annotations

import numpy as np


def tournament_pick(
    ranked: list[int], scores: np.ndarray, k: int, rng: np.random.Generator
) -> int:
    """Best of `k` random candidates -- mild pressure, keeps diversity."""
    cand = rng.choice(len(ranked), size=min(k, len(ranked)), replace=False)
    return int(max(cand, key=lambda i: scores[ranked[i]]))


def crossover(
    a: np.ndarray, b: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Uniform crossover: each gene independently from A or B.

    Per-gene rather than per-layer on purpose. Layer-wise splicing of two
    networks tends to produce a non-functional child, because a layer's
    weights are only meaningful next to the layer that fed them.
    """
    mask = rng.random(a.shape) < rate
    child = a.copy()
    child[mask] = b[mask]
    return child


def mutate(
    genome: np.ndarray, sigma: float, scale: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Add gaussian noise proportional to each tensor's own weight scale.

    A flat sigma would be devastating to small-magnitude layers and
    imperceptible to large ones; scaling by the parameter block's own standard
    deviation keeps the perturbation meaningful everywhere.
    """
    return genome + rng.normal(0.0, 1.0, genome.shape).astype(np.float32) * (
        sigma * scale
    )


def per_tensor_scale(genome: np.ndarray, shapes: list) -> np.ndarray:
    """A per-gene multiplier equal to its own tensor's std (min-clamped)."""
    scale = np.empty_like(genome)
    i = 0
    for _name, _shape, numel in shapes:
        block = genome[i : i + numel]
        s = float(block.std())
        scale[i : i + numel] = max(s, 1e-3)
        i += numel
    return scale


def next_generation(
    genomes: list[np.ndarray],
    scores: np.ndarray,
    shapes: list,
    cfg,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Elites survive verbatim; the rest are bred from them and mutated."""
    ranked = list(np.argsort(-scores))
    survivors = [genomes[i].copy() for i in ranked[: cfg.elites]]

    children: list[np.ndarray] = []
    while len(survivors) + len(children) < cfg.population:
        ia = tournament_pick(ranked, scores, cfg.tournament, rng)
        ib = tournament_pick(ranked, scores, cfg.tournament, rng)
        a, b = genomes[ranked[ia]], genomes[ranked[ib]]
        child = crossover(a, b, cfg.crossover_rate, rng)
        child = mutate(child, cfg.sigma, per_tensor_scale(child, shapes), rng)
        children.append(child.astype(np.float32))
    return survivors + children
