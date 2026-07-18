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
from .rollout import GameResult, GameSpec, play_one


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
) -> list[tuple[int, GameResult]]:
    """Runs inside a worker process: rebuild the model(s) once, then play
    every spec in this chunk sequentially — a single process still only
    ever runs one game at a time, same as the non-parallel path."""
    current_model = PolicyValueNet()
    current_model.load_state_dict(current_state)
    current_model.eval()
    opponent_model = PolicyValueNet()
    return [
        (idx, play_one(current_model, opponent_model, deck, spec))
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
) -> list[GameResult]:
    """Play every spec in `specs` across the pool; returns results in the
    same order as `specs`."""
    chunks = _chunk_specs(specs, num_workers)
    futures = [
        executor.submit(_play_chunk, current_state, deck, chunk) for chunk in chunks
    ]
    results: list[GameResult | None] = [None] * len(specs)
    for future in futures:
        for idx, result in future.result():
            results[idx] = result
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]
