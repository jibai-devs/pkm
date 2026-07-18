"""Multiprocess self-play rollout collection (synchronous vectorized).

Design & rationale: ``research/infra-todo.md §8a``. The engine has process-global
state (one battle per process), so parallelism is multiprocessing with one engine
per worker — never threads. This is a **synchronous** collector (correct for
on-policy PPO): every update, the learner broadcasts the current weights, each
worker plays its share of games with those weights, and returns its trajectories;
the learner then runs one PPO update and repeats.

Workers are persistent (spawned once, reused) to avoid re-importing the engine /
rebuilding the model every update. Must use the ``spawn`` start method: ``fork``
would copy the parent's already-initialized native engine (shared ``battle_ptr``)
and corrupt it, whereas ``spawn`` re-imports ``pkm.cabt`` fresh per worker so each
gets its own ``GameInitialize()`` and ``battle_ptr``.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.multiprocessing as tmp


def _worker(rank: int, cfg_dict: dict, cmd_q, res_q, base_seed: int) -> None:
    """Persistent worker: load weights on command, play games, return Steps.

    Top-level function (required by the ``spawn`` start method — cannot be a
    closure). Importing here (inside the fresh process) triggers this worker's
    own engine load + ``GameInitialize()``.
    """
    try:
        torch.set_num_threads(1)  # avoid N x BLAS oversubscription
        torch.manual_seed(base_seed + rank)  # distinct sampling per worker
        from pkm.new_agents.agent_000_dragapult.config import Config, build_model
        from pkm.new_agents.agent_000_dragapult import trainers

        cfg = Config.from_dict(cfg_dict)
        model = build_model(cfg)
        trainer = trainers.get_trainer(cfg)
        while True:
            cmd = cmd_q.get()
            if cmd is None:  # shutdown sentinel
                break
            state_dict, n_games = cmd
            model.load_state_dict(state_dict)
            t0 = time.perf_counter()
            steps, stats = trainer.collect(model, n_games, cfg)
            stats["t_worker"] = time.perf_counter() - t0  # busy time (excl. barrier wait)
            res_q.put((rank, steps, stats, None))
    except Exception:  # propagate so the learner doesn't hang on res_q.get()
        import traceback

        res_q.put((rank, [], {}, traceback.format_exc()))


class ParallelRollout:
    """Manages persistent spawn workers for synchronous self-play collection."""

    def __init__(
        self,
        cfg,
        num_workers: int,
        base_seed: int = 0,
        model: torch.nn.Module | None = None,
    ):
        # `model` should be the caller's live TrainState.model: PPO/ExIt update
        # steps mutate its parameters in place (never reassign the object), so
        # holding this one reference is enough for `collect` to always broadcast
        # the current weights without the model being re-passed on every call.
        # Callers that don't have one yet (e.g. this module's own test) get a
        # freshly built model instead.
        from pkm.new_agents.agent_000_dragapult.config import build_model

        self.model = model if model is not None else build_model(cfg)
        self.num_workers = num_workers
        self.ctx = tmp.get_context("spawn")
        self.cmd_qs = [self.ctx.Queue() for _ in range(num_workers)]
        self.res_q = self.ctx.Queue()
        cfg_dict = cfg.to_dict()
        self.procs = []
        for rank in range(num_workers):
            p = self.ctx.Process(
                target=_worker,
                args=(rank, cfg_dict, self.cmd_qs[rank], self.res_q, base_seed),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def _split(self, total_games: int) -> list[int]:
        per = [total_games // self.num_workers] * self.num_workers
        for i in range(total_games % self.num_workers):
            per[i] += 1
        return per

    def collect(
        self, trainer: Any, total_games: int
    ) -> tuple[list, dict[str, Any]]:
        """Broadcast weights, gather trajectories from all workers, aggregate stats.

        ``trainer`` is unused here: each worker already self-selects its own
        trainer from ``cfg.train.method`` (see ``_worker``). It is kept as a
        parameter purely for interface symmetry with the single-process caller
        (``trainer.collect(ts.model, ...)`` in ``train.train``).
        """
        per = self._split(total_games)
        state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        for rank in range(self.num_workers):
            self.cmd_qs[rank].put((state_dict, per[rank]))

        steps: list = []
        stats_list: list[dict] = []
        for _ in range(self.num_workers):
            rank, s, st, err = self.res_q.get()
            if err is not None:
                raise RuntimeError(f"rollout worker {rank} failed:\n{err}")
            steps.extend(s)
            stats_list.append(st)

        games = sum(st.get("games", 0) for st in stats_list)
        d = max(games, 1)
        out = {
            "games": games,
            "steps": len(steps),
            "p0_win": sum(
                st.get("p0_win", 0.0) * st.get("games", 0) for st in stats_list
            )
            / d,
            "p1_win": sum(
                st.get("p1_win", 0.0) * st.get("games", 0) for st in stats_list
            )
            / d,
        }

        # Per-worker load-balance diagnostics. Because collection is synchronous
        # (a barrier at res_q), the learner waits for the SLOWEST worker; every
        # worker that finished earlier is idle until then. worker_busy_* expose
        # that spread so `train` can turn it into a utilization %.
        busy = [st.get("t_worker", 0.0) for st in stats_list]
        wgames = [st.get("games", 0) for st in stats_list]
        if busy:
            out.update(
                {
                    "num_workers": self.num_workers,
                    "worker_busy_min": min(busy),
                    "worker_busy_max": max(busy),  # the straggler that gates rollout
                    "worker_busy_mean": sum(busy) / len(busy),
                    "worker_busy_sum": sum(busy),  # total core-seconds of real work
                    "worker_games_min": min(wgames),
                    "worker_games_max": max(wgames),
                }
            )
        return steps, out

    def close(self) -> None:
        for q in self.cmd_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=5)
        for p in self.procs:
            if p.is_alive():
                p.terminate()
