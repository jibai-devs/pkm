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
        from pkm.new_agents.agent_000_dragapult.train import collect_rollout

        cfg = Config.from_dict(cfg_dict)
        model = build_model(cfg)
        while True:
            cmd = cmd_q.get()
            if cmd is None:  # shutdown sentinel
                break
            state_dict, n_games = cmd
            model.load_state_dict(state_dict)
            steps, stats = collect_rollout(model, n_games, cfg)
            res_q.put((rank, steps, stats, None))
    except Exception:  # propagate so the learner doesn't hang on res_q.get()
        import traceback

        res_q.put((rank, [], {}, traceback.format_exc()))


class ParallelRollout:
    """Manages persistent spawn workers for synchronous self-play collection."""

    def __init__(self, cfg, num_workers: int, base_seed: int = 0):
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
        self, model: torch.nn.Module, total_games: int
    ) -> tuple[list, dict[str, Any]]:
        """Broadcast weights, gather trajectories from all workers, aggregate stats."""
        per = self._split(total_games)
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
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
        return steps, {
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

    def close(self) -> None:
        for q in self.cmd_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=5)
        for p in self.procs:
            if p.is_alive():
                p.terminate()
