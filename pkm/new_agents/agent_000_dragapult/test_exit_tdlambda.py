"""Tests for ExIt improvements ③ TD(λ) value targets + ④ W-world determinization.

  * config: the new knobs are hashed + backfill (defaults reproduce v1 "mc");
  * mcts.search/search_worlds optionally return the root value (in [-1,1]);
  * _assign_value_targets: "mc" == raw outcome (unchanged), "tdlambda" == the
    bounded backward blend;
  * end-to-end ExIt collect with tdlambda + W=2 fills finite, in-range targets.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pkm.new_agents.agent_000_dragapult import mcts
from pkm.new_agents.agent_000_dragapult.cabt import battle_finish, battle_start
from pkm.new_agents.agent_000_dragapult.config import Config, build_model
from pkm.new_agents.agent_000_dragapult import deck
from pkm.new_agents.agent_000_dragapult.trainers.exit import (
    ExItSample,
    ExItTrainer,
    _assign_value_targets,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_new_knobs_hash_and_backfill() -> None:
    base = Config()
    assert base.train.mcts_worlds == 1
    assert base.train.exit_value_target == "mc"
    td = Config()
    object.__setattr__(td.train, "exit_value_target", "tdlambda")
    object.__setattr__(td.train, "mcts_worlds", 4)
    assert td.hash() != base.hash()
    # Old checkpoint config lacking the fields backfills to the v1 defaults.
    d = base.to_dict()
    for k in ("mcts_worlds", "exit_value_target", "exit_lambda"):
        d["train"].pop(k)
    got = Config.from_dict(d)
    assert got.train.mcts_worlds == 1 and got.train.exit_value_target == "mc"


# --------------------------------------------------------------------------- #
# _assign_value_targets (pure)
# --------------------------------------------------------------------------- #


def _sample(seat: int, root_value: float) -> ExItSample:
    return ExItSample(
        features=None,  # type: ignore[arg-type]  # unused by target assignment
        policy_target=np.zeros(1, dtype=np.float32),
        seat=seat,
        root_value=root_value,
    )


def _cfg(scheme: str, lam: float = 0.9):
    return SimpleNamespace(
        train=SimpleNamespace(exit_value_target=scheme, exit_lambda=lam)
    )


def test_mc_targets_are_raw_outcome() -> None:
    # seat 0 wins (result==0): seat0 -> +1, seat1 -> -1, regardless of root_value.
    samples = [_sample(0, 0.7), _sample(1, -0.2), _sample(0, -0.9)]
    _assign_value_targets(samples, result=0, cfg=_cfg("mc"))
    assert [s.value_target for s in samples] == [1.0, -1.0, 1.0]


def test_tdlambda_targets_blend_and_stay_in_range() -> None:
    samples = [_sample(0, 0.5), _sample(0, -0.3), _sample(1, 0.1)]
    _assign_value_targets(samples, result=0, cfg=_cfg("tdlambda", lam=0.9))
    # every target is a mean of a value in [-1,1] and a root_value in [-1,1]
    assert all(-1.0 <= s.value_target <= 1.0 for s in samples)
    # last seat-0 decision anchors to (outcome + root)/2 = (1 + -0.3)/2 = 0.35
    seat0 = [s for s in samples if s.seat == 0]
    assert abs(seat0[-1].value_target - 0.35) < 1e-6
    # differs from the raw-outcome scheme (it actually blends)
    mc = [_sample(0, 0.5), _sample(0, -0.3), _sample(1, 0.1)]
    _assign_value_targets(mc, result=0, cfg=_cfg("mc"))
    assert [s.value_target for s in samples] != [s.value_target for s in mc]


# --------------------------------------------------------------------------- #
# mcts.search / search_worlds return the root value
# --------------------------------------------------------------------------- #


def _search_cfg(sims: int = 6):
    return SimpleNamespace(
        train=SimpleNamespace(
            mcts_simulations=sims,
            mcts_c_puct=1.25,
            mcts_temperature=1.0,
            determinization="sample",
        ),
        run=SimpleNamespace(deck=deck.DEFAULT_DECK),
    )


def test_search_returns_root_value_in_range() -> None:
    torch.manual_seed(0)
    model = build_model(Config()).eval()
    deck60 = deck.deck_60(deck.DEFAULT_DECK)
    obs, _ = battle_start(deck60, deck60)
    try:
        seat = obs["current"]["yourIndex"]
        gen = torch.Generator().manual_seed(0)
        pi, v = mcts.search(obs, seat, model, _search_cfg(), gen, return_value=True)
        assert pi.ndim == 1 and pi.shape[0] == len(obs["select"]["option"])
        assert np.isfinite(v) and -1.0 <= v <= 1.0
        # search_worlds W=2 also returns a finite in-range value
        pi2, v2 = mcts.search_worlds(
            obs, seat, model, _search_cfg(), gen, n_worlds=2, return_value=True
        )
        assert pi2.shape == pi.shape and np.isfinite(v2) and -1.0 <= v2 <= 1.0
        # default (no return_value) still yields a bare array (back-compat)
        pi3 = mcts.search(obs, seat, model, _search_cfg(), gen)
        assert isinstance(pi3, np.ndarray)
    finally:
        battle_finish()


# --------------------------------------------------------------------------- #
# End-to-end: ExIt collect with tdlambda + W worlds
# --------------------------------------------------------------------------- #


def test_exit_collect_tdlambda_worlds() -> None:
    cfg = Config()
    object.__setattr__(cfg.train, "method", "exit")
    object.__setattr__(cfg.train, "mcts_simulations", 4)
    object.__setattr__(cfg.train, "mcts_worlds", 2)
    object.__setattr__(cfg.train, "exit_value_target", "tdlambda")
    torch.manual_seed(0)
    model = build_model(cfg)
    samples, _ = ExItTrainer().collect(model, n_games=1, cfg=cfg)
    assert samples and all(isinstance(s, ExItSample) for s in samples)
    for s in samples:
        assert np.isfinite(s.value_target) and -1.0 <= s.value_target <= 1.0
        assert np.isfinite(s.root_value) and -1.0 <= s.root_value <= 1.0
    # the update step consumes these targets without NaN
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    upd = ExItTrainer().update(model, opt, samples, cfg)
    assert upd["value_loss"] == upd["value_loss"]
