"""Tests for pluggable reward shaping + advantage estimation (shaping.py)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import pytest

from pkm.new_agents.agent_000_dragapult.config import Config
from pkm.new_agents.agent_000_dragapult.shaping import (
    ESTIMATORS,
    SHAPERS,
    assign_targets,
    gae,
    terminal_shaper,
)


@dataclass
class _Feat:
    globals: np.ndarray  # only [7]=own_prize, [8]=opp_prize are read


@dataclass
class _Step:
    """Minimal duck-typed stand-in for train.Step."""

    seat: int
    value: float
    features: _Feat
    reward: float = 0.0
    adv: float = 0.0
    ret: float = 0.0


def _mk_step(seat: int, value: float, own_prize: float, opp_prize: float) -> _Step:
    g = np.zeros(16, dtype=np.float32)
    g[7], g[8] = own_prize, opp_prize
    return _Step(seat=seat, value=value, features=_Feat(globals=g))


def _cfg(**train_kw) -> Config:
    base = Config()
    return replace(base, train=replace(base.train, **train_kw))


# --------------------------------------------------------------------------- #
# terminal shaper
# --------------------------------------------------------------------------- #


def test_terminal_shaper_sparse_pm1():
    traj = [_mk_step(0, 0.0, 1.0, 1.0) for _ in range(3)]
    terminal_shaper(traj, seat=0, result=0, cfg=_cfg())
    assert [s.reward for s in traj] == [0.0, 0.0, 1.0]  # +1 win on last only

    traj = [_mk_step(0, 0.0, 1.0, 1.0) for _ in range(3)]
    terminal_shaper(traj, seat=0, result=1, cfg=_cfg())
    assert [s.reward for s in traj] == [0.0, 0.0, -1.0]  # loss

    traj = [_mk_step(0, 0.0, 1.0, 1.0) for _ in range(2)]
    terminal_shaper(traj, seat=0, result=-1, cfg=_cfg())
    assert [s.reward for s in traj] == [0.0, 0.0]  # draw


# --------------------------------------------------------------------------- #
# GAE numerics (hand-computed reference)
# --------------------------------------------------------------------------- #


def test_gae_matches_reference():
    cfg = _cfg(gamma=0.9, gae_lambda=0.8)
    # rewards: [0, 1], values: [0.5, 0.2]
    traj = [_mk_step(0, 0.5, 1, 1), _mk_step(0, 0.2, 1, 1)]
    traj[0].reward, traj[1].reward = 0.0, 1.0
    gae(traj, cfg)
    # t=1 (terminal): delta = 1 + 0 - 0.2 = 0.8 ; adv = 0.8
    # t=0: delta = 0 + 0.9*0.2 - 0.5 = -0.32 ; adv = -0.32 + 0.9*0.8*0.8 = 0.256
    assert traj[1].adv == pytest.approx(0.8)
    assert traj[0].adv == pytest.approx(0.256)
    assert traj[0].ret == pytest.approx(0.256 + 0.5)
    assert traj[1].ret == pytest.approx(0.8 + 0.2)


# --------------------------------------------------------------------------- #
# prize_potential reduces to terminal when coef == 0
# --------------------------------------------------------------------------- #


def _two_seat_steps() -> list[_Step]:
    return [
        _mk_step(0, 0.1, 1.0, 1.0),
        _mk_step(1, -0.1, 1.0, 1.0),
        _mk_step(0, 0.2, 0.83, 1.0),  # seat 0 took a prize
        _mk_step(1, -0.2, 0.83, 0.83),
        _mk_step(0, 0.3, 0.66, 0.83),
    ]


def test_prize_potential_coef0_equals_terminal():
    base = _two_seat_steps()
    a = copy.deepcopy(base)
    b = copy.deepcopy(base)
    assign_targets(a, result=0, cfg=_cfg(shaping="terminal"))
    assign_targets(b, result=0, cfg=_cfg(shaping="prize_potential", shaping_coef=0.0))
    for sa, sb in zip(a, b):
        assert sa.reward == pytest.approx(sb.reward)
        assert sa.adv == pytest.approx(sb.adv)
        assert sa.ret == pytest.approx(sb.ret)


def test_prize_potential_adds_intermediate_signal():
    steps = _two_seat_steps()
    assign_targets(steps, result=0, cfg=_cfg(shaping="prize_potential", shaping_coef=1.0))
    seat0 = [s for s in steps if s.seat == 0]
    # non-terminal steps now carry nonzero shaped reward (terminal-only would be 0)
    assert any(abs(s.reward) > 0 for s in seat0[:-1])


def test_prize_potential_policy_invariance_telescopes():
    # Sum of shaped rewards over a seat's trajectory == terminal + coef*(gamma^T*phi_T - phi_0)
    cfg = _cfg(shaping="prize_potential", shaping_coef=1.0)
    steps = _two_seat_steps()
    assign_targets(steps, result=0, cfg=cfg)
    for seat in (0, 1):
        traj = [s for s in steps if s.seat == seat]
        g = cfg.train.gamma
        phi = lambda s: float(s.features.globals[8] - s.features.globals[7])  # noqa: E731
        # discounted sum of the shaping term only (strip terminal ±1)
        term = 1.0 if seat == 0 else -1.0
        shaped_sum = sum(g**t * traj[t].reward for t in range(len(traj)))
        expected_terminal = g ** (len(traj) - 1) * term
        potential_part = shaped_sum - expected_terminal
        telescoped = g ** (len(traj) - 1) * phi(traj[-1]) - phi(traj[0])
        assert potential_part == pytest.approx(telescoped, abs=1e-5)


# --------------------------------------------------------------------------- #
# registry / errors
# --------------------------------------------------------------------------- #


def test_unknown_keys_raise():
    with pytest.raises(ValueError, match="unknown shaping"):
        assign_targets([], result=0, cfg=_cfg(shaping="nope"))
    with pytest.raises(ValueError, match="unknown advantage"):
        assign_targets([], result=0, cfg=_cfg(advantage="nope"))


def test_registries_populated():
    assert "terminal" in SHAPERS and "prize_potential" in SHAPERS
    assert "gae" in ESTIMATORS
