"""Pluggable reward shaping + advantage estimation for the self-play trainer.

Two separable concerns, deliberately split so each is chosen independently by
config (`TrainConfig.shaping` / `.advantage`) and serialized into every
checkpoint:

  * **RewardShaper** — writes ``step.reward`` for one seat's trajectory, given
    the game ``result``. This is where *reward shaping* lives (terminal-only
    ±1, potential-based prize differential, …).
  * **AdvantageEstimator** — reads ``step.reward`` + ``step.value`` and writes
    ``step.adv`` / ``step.ret`` for one seat's trajectory. This is where the
    *estimator* lives (GAE(λ), and later TD(λ)/V-trace/…).

The estimator knows nothing about prizes or terminals — it only sees rewards.
Neither imports :mod:`.train`; they duck-type the ``Step`` record (fields
``features``, ``value``, ``reward``, ``adv``, ``ret``), so there's no import
cycle. :func:`assign_targets` is the single entry point the trainer calls.

**Policy invariance:** the potential-based term ``F = γ·Φ(s') − Φ(s)`` (Ng et
al. 1999) provably leaves the optimal policy unchanged — it only densifies the
learning signal — so ``prize_potential`` is safe to enable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:  # avoid an import cycle with train.py at runtime
    from pkm.new_agents.agent_000_dragapult.config import Config


class StepLike(Protocol):
    """The subset of ``train.Step`` the shapers/estimators read and write."""

    value: float
    reward: float
    adv: float
    ret: float

    @property
    def features(self) -> object: ...


# (trajectory, seat, result, cfg) -> mutates step.reward
RewardShaper = Callable[[list["StepLike"], int, int, "Config"], None]
# (trajectory, cfg) -> mutates step.adv / step.ret
AdvantageEstimator = Callable[[list["StepLike"], "Config"], None]


# --------------------------------------------------------------------------- #
# Reward shapers
# --------------------------------------------------------------------------- #


def _seat_reward(result: int, seat: int) -> float:
    """Zero-sum terminal reward from ``seat``'s view: +1 win, -1 loss, 0 draw."""
    if result == seat:
        return 1.0
    if result in (0, 1):  # a decisive result for the other seat
        return -1.0
    return 0.0  # draw / unknown


def terminal_shaper(traj: list["StepLike"], seat: int, result: int, cfg: "Config") -> None:
    """Sparse ±1 on the last step, 0 everywhere else (the v1 baseline)."""
    last = len(traj) - 1
    for t, s in enumerate(traj):
        s.reward = _seat_reward(result, seat) if t == last else 0.0


# Prize counts are stored (normalised, fraction remaining) in features.globals.
# See features.py: g[7]=own_prize, g[8]=opp_prize.
_OWN_PRIZE = 7
_OPP_PRIZE = 8


def prize_potential_shaper(
    traj: list["StepLike"], seat: int, result: int, cfg: "Config"
) -> None:
    """Terminal ±1 plus a policy-invariant potential on the prize differential.

    Potential from this seat's view Φ(s) = opp_prize_remaining − own_prize_remaining
    (higher is better for us: fewer of our prizes left, more of theirs). The
    shaping reward added to step ``t`` is ``coef · (γ·Φ(s_{t+1}) − Φ(s_t))``,
    which telescopes to ``coef·(γ^T·Φ_terminal − Φ_0)`` and cannot change the
    optimal policy.
    """
    terminal_shaper(traj, seat, result, cfg)  # keep the ±1 terminal reward
    coef = cfg.train.shaping_coef
    if coef == 0.0:  # exactly reproduce terminal_shaper
        return
    gamma = cfg.train.gamma

    def phi(s: "StepLike") -> float:
        g = s.features.globals  # type: ignore[attr-defined]
        return float(g[_OPP_PRIZE] - g[_OWN_PRIZE])

    for t in range(len(traj) - 1):
        traj[t].reward += coef * (gamma * phi(traj[t + 1]) - phi(traj[t]))


SHAPERS: dict[str, RewardShaper] = {
    "terminal": terminal_shaper,
    "prize_potential": prize_potential_shaper,
}


# --------------------------------------------------------------------------- #
# Advantage estimators
# --------------------------------------------------------------------------- #


def gae(traj: list["StepLike"], cfg: "Config") -> None:
    """Generalized Advantage Estimation over one seat's trajectory.

    Reads ``step.reward`` (set by the shaper) and ``step.value``; writes
    ``step.adv`` and ``step.ret``. The final recorded step is treated as
    terminal (``next_v = 0``); the game genuinely ends there.
    """
    gamma, lam = cfg.train.gamma, cfg.train.gae_lambda
    adv = 0.0
    for t in reversed(range(len(traj))):
        last = t == len(traj) - 1
        next_v = 0.0 if last else traj[t + 1].value
        nonterm = 0.0 if last else 1.0
        delta = traj[t].reward + gamma * next_v * nonterm - traj[t].value
        adv = delta + gamma * lam * nonterm * adv
        traj[t].adv = adv
        traj[t].ret = adv + traj[t].value


ESTIMATORS: dict[str, AdvantageEstimator] = {
    "gae": gae,
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def assign_targets(steps: list["StepLike"], result: int, cfg: "Config") -> None:
    """Fill ``reward``/``adv``/``ret`` on every step, per seat, per config.

    Each seat's decisions form their own trajectory (the other seat's moves are
    hidden environment dynamics), so shaping + estimation run once per seat.
    """
    try:
        shaper = SHAPERS[cfg.train.shaping]
    except KeyError:
        raise ValueError(
            f"unknown shaping {cfg.train.shaping!r}; choose from {sorted(SHAPERS)}"
        ) from None
    try:
        estimator = ESTIMATORS[cfg.train.advantage]
    except KeyError:
        raise ValueError(
            f"unknown advantage {cfg.train.advantage!r}; choose from {sorted(ESTIMATORS)}"
        ) from None
    for seat in (0, 1):
        traj = [s for s in steps if s.seat == seat]  # type: ignore[attr-defined]
        shaper(traj, seat, result, cfg)
        estimator(traj, cfg)
