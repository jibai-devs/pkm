"""Auxiliary-task registry for agent_000_dragapult.

An **auxiliary task** hangs an extra prediction head off the shared trunk state
(the ``[B, d_state]`` encoder summary in :mod:`.model`) and trains it with its
own loss, purely to shape the trunk's representation — the policy and value that
ride the same trunk get a richer gradient for free.

Design (mirrors the reward-term / shaper registries): each :class:`AuxTask`
bundles the three things a task needs, and which tasks are *active* is chosen by
config (``TrainConfig.aux_weights`` maps ``name -> weight``; ``weight > 0`` = on,
default all-zero = every task off = v1 behaviour bit-for-bit). Adding a new
auxiliary is one ``register(...)`` call; nothing else in the codebase changes.

An :class:`AuxTask` bundles:
  * ``make_head(d_state) -> nn.Module`` — the head, mapping trunk state to a
    prediction (``[B]`` for a scalar task).
  * ``assign(steps, terminal_obs)`` — fills ``step.aux[name]`` with the training
    label for every recorded step, from the finished game.
  * ``loss(pred, target) -> Tensor`` — how predictions are scored (e.g. MSE).

**Training-only.** Aux heads are never consulted by policy/value at inference or
MCTS, and are stripped from the Kaggle bundle at pack time (see ``cli.py:pack``),
so they add zero inference cost and cannot affect the torch<->numpy parity gate.
See ``docs/00_aux_loss.md`` for the rationale and the menu of candidate tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:  # avoid an import cycle / heavy import at module load
    from pkm.new_agents.agent_000_dragapult.trainers.ppo import Step

# Prize piles start at 6; the margin lives in [-6, 6]. Normalise labels into
# roughly [-1, 1] so the aux loss is on the same scale as the value loss.
_MAX_PRIZES = 6.0


@dataclass(frozen=True)
class AuxTask:
    """One auxiliary prediction task (head factory + labeller + loss)."""

    name: str
    make_head: Callable[[int], nn.Module]
    assign: Callable[[list["Step"], dict], None]
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


AUX_TASKS: dict[str, AuxTask] = {}


def register(task: AuxTask) -> AuxTask:
    if task.name in AUX_TASKS:
        raise ValueError(f"duplicate aux task {task.name!r}")
    AUX_TASKS[task.name] = task
    return task


def task_names() -> list[str]:
    return sorted(AUX_TASKS)


def default_weights() -> dict[str, float]:
    """Every registered task at 0.0 — the default (all aux off)."""
    return {name: 0.0 for name in AUX_TASKS}


def active_tasks(weights: dict[str, float]) -> list[str]:
    """Names of tasks with a positive weight, in a stable order.

    This is the single definition of *which* heads a config builds, used both by
    the model (to construct the heads) and the trainer (to add the losses), so
    the two never disagree.
    """
    return [name for name in task_names() if weights.get(name, 0.0) > 0.0]


def _scalar_head(d_state: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_state, d_state),
        nn.ReLU(),
        nn.Linear(d_state, 1),
    )


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean()


# --------------------------------------------------------------------------- #
# Tier A — prize_margin: predict the final prize-count margin (dense outcome).
# Terminal label only, so no opponent-observation plumbing needed. This is the
# recommended first experiment (see docs/00_aux_loss.md).
# --------------------------------------------------------------------------- #


def _assign_prize_margin(steps: list["Step"], terminal_obs: dict) -> None:
    """Label each step with the seat-signed final prize margin, normalised.

    Prizes *taken* by a player = 6 - prizes remaining in their pile. The margin
    (yours - opponent's) is read once off the terminal observation and signed to
    each step's own seat, so a step always predicts "how well did *I* end up".
    """
    cur = terminal_obs.get("current") if isinstance(terminal_obs, dict) else None
    if not cur:
        return  # no terminal board (shouldn't happen); leave labels at default
    players = cur.get("players") or []
    if len(players) < 2:
        return

    def _prizes_left(p: dict) -> int:
        return len([c for c in (p.get("prize") or []) if c is not None])

    left0, left1 = _prizes_left(players[0]), _prizes_left(players[1])
    # prizes taken by p0 minus prizes taken by p1 == left1 - left0.
    margin_p0 = (left1 - left0) / _MAX_PRIZES
    for s in steps:
        s.aux["prize_margin"] = margin_p0 if s.seat == 0 else -margin_p0


register(
    AuxTask(
        name="prize_margin",
        make_head=_scalar_head,
        assign=_assign_prize_margin,
        loss=_mse,
    )
)
