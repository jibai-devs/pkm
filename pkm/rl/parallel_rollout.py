"""Multiprocess self-play rollout.

The cabt engine keeps its live battle as process-global state
(``kaggle_environments...cg.sim.Battle.battle_ptr``, plus the loaded native
``lib`` handle) — concurrent games on different *threads* of one process
would clobber each other's battle pointer. Separate OS processes each get
their own copy of that state on import, so this uses a
``ProcessPoolExecutor``, not threads.

The pool is meant to be created once (outside the training loop) and reused
across every iteration — spawning a process is expensive (Windows has no
fork; each worker re-imports the whole package, including
``kaggle_environments``' noisy import chain), so paying that cost every
iteration would likely erase the speedup this exists to provide.
"""

from concurrent.futures import ProcessPoolExecutor

import torch

from .model import PolicyValueNet
from .rollout import GameResult, GameSpec, TorchPolicy, play_game, play_one


def init_worker() -> None:
    """Pool initializer, run once per worker process at startup.

    Pins each worker to a single torch thread — N worker processes each
    spawning torch's own (default multi-core) BLAS threads would massively
    oversubscribe the machine's cores.
    """
    torch.set_num_threads(1)


def make_pool(num_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker)


def _play_chunk(
    current_state: dict,
    deck: list[int],
    indexed_specs: list[tuple[int, GameSpec]],
    archetype_classifier=None,
) -> list[tuple[int, GameResult]]:
    """Runs inside a worker process: rebuild the model(s) once, then play
    every spec in this chunk sequentially — a single process still only
    ever runs one game at a time, same as the non-parallel path.

    `archetype_classifier` (see pkm/rl/rollout.py:play_one) is a plain numpy
    object (NumpyArchetypeClassifier), picklable by default -- passed through
    to each worker like `current_state`/`deck`."""
    current_model = PolicyValueNet()
    current_model.load_state_dict(current_state)
    current_model.eval()
    opponent_model = PolicyValueNet()
    return [
        (idx, play_one(current_model, opponent_model, deck, spec, archetype_classifier))
        for idx, spec in indexed_specs
    ]


def _chunk_specs(
    specs: list[GameSpec], num_workers: int
) -> list[list[tuple[int, GameSpec]]]:
    """Round-robin `specs` into up to `num_workers` roughly-even chunks,
    each entry tagged with its original index so results can be placed back
    in order regardless of which worker finishes first."""
    if not specs:
        return []
    n = min(num_workers, len(specs))
    chunks: list[list[tuple[int, GameSpec]]] = [[] for _ in range(n)]
    for i, spec in enumerate(specs):
        chunks[i % n].append((i, spec))
    return chunks


def collect_parallel(
    executor: ProcessPoolExecutor,
    num_workers: int,
    current_state: dict,
    deck: list[int],
    specs: list[GameSpec],
    archetype_classifier=None,
) -> list[GameResult]:
    """Play every spec in `specs` across the pool; returns results in the
    same order as `specs`."""
    chunks = _chunk_specs(specs, num_workers)
    futures = [
        executor.submit(_play_chunk, current_state, deck, chunk, archetype_classifier)
        for chunk in chunks
    ]
    results: list[GameResult | None] = [None] * len(specs)
    for future in futures:
        for idx, result in future.result():
            results[idx] = result
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


PopGame = tuple[list[int], dict, list[int], dict, tuple[bool, bool]]


def _play_pop_chunk(
    indexed_games: list[
        tuple[int, list[int], dict, list[int], dict, tuple[bool, bool]]
    ],
    archetype_classifier=None,
) -> list[tuple[int, GameResult]]:
    """Population-training counterpart to `_play_chunk`. Population training
    has no single shared "current model" opponent pair -- every roster
    member is its own live model -- so each entry carries *both* sides' own
    deck + state_dict directly, and a worker rebuilds exactly the two models
    it needs per game (deliberately decoupled from PopulationMember/PopSpec
    in population_train.py, which import from here, not the reverse).

    `archetype_classifier`, when given, is attached to **both** sides --
    unlike `_play_chunk`'s trainee-only convention, population training has
    no frozen opponent: every roster member is simultaneously being trained,
    so each one computes its own live belief about whoever it's facing (see
    docs/superpowers/plans/2026-07-20-belief-classifier-routing.md)."""
    model_a = PolicyValueNet()
    model_b = PolicyValueNet()
    results = []
    for idx, deck_a, state_a, deck_b, state_b, collect in indexed_games:
        model_a.load_state_dict(state_a)
        model_a.eval()
        model_b.load_state_dict(state_b)
        model_b.eval()
        result = play_game(
            (
                TorchPolicy(model_a, archetype_classifier=archetype_classifier),
                TorchPolicy(model_b, archetype_classifier=archetype_classifier),
            ),
            (deck_a, deck_b),
            collect=collect,
        )
        results.append((idx, result))
    return results


def _chunk_pop_games(
    games: list[PopGame], num_workers: int
) -> list[list[tuple[int, list[int], dict, list[int], dict, tuple[bool, bool]]]]:
    """Same round-robin chunking as `_chunk_specs`, generalized to the plain
    (deck_a, state_a, deck_b, state_b, collect) tuples population training
    hands in instead of a `GameSpec`."""
    if not games:
        return []
    n = min(num_workers, len(games))
    chunks: list[
        list[tuple[int, list[int], dict, list[int], dict, tuple[bool, bool]]]
    ] = [[] for _ in range(n)]
    for i, g in enumerate(games):
        chunks[i % n].append((i, *g))
    return chunks


def collect_pop_parallel(
    executor: ProcessPoolExecutor,
    num_workers: int,
    games: list[PopGame],
    archetype_classifier=None,
) -> list[GameResult]:
    """Play every (deck_a, state_a, deck_b, state_b, collect) tuple in
    `games` across the pool; returns results in the same order as `games`.

    `archetype_classifier` is the same object for every chunk (unlike the
    per-game state dicts), passed once per `executor.submit` rather than
    duplicated into every `PopGame` tuple -- it's a plain numpy object
    (NumpyArchetypeClassifier), already confirmed to pickle cleanly across
    worker processes (see `_play_chunk`'s equivalent, train.py's
    `--archetype-belief` path, smoke-tested under `--workers 2`)."""
    chunks = _chunk_pop_games(games, num_workers)
    futures = [
        executor.submit(_play_pop_chunk, chunk, archetype_classifier)
        for chunk in chunks
    ]
    results: list[GameResult | None] = [None] * len(games)
    for future in futures:
        for idx, result in future.result():
            results[idx] = result
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]
