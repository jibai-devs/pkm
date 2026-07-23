"""PUCT MCTS over the engine's search_* forward model, guided by model.evaluate.

Node = one engine search node (searchId + observation). Value is backed up
negamax (sign flips when the child's acting seat differs). v1 uses single-sample
determinization (K=1) and lets the engine resolve chance inside search_step
(no explicit chance nodes) — see docs/specs §5.2, §5.3, §7.

**Branching-tree variant (per Task 5's characterization,
`SEARCH_STEP_BRANCHES=True`):** `cabt.search_step(node.search_id, [a])` returns
a distinct, persistent child `searchId` per call from the same parent -- the
engine's search API is a real tree, not a single mutable cursor. So children
are memoized (keyed by the submitted select -- see below) on the `_Node`
itself and simulations descend via repeated `search_step` calls, never
re-rooting. The re-root-and-replay alternative (for a
`SEARCH_STEP_BRANCHES=False` engine) is intentionally not implemented
(YAGNI): it does not match the observed engine.

**Multi-count decisions (found empirically, not in the brief):** the engine's
own `search_step` validates ``select.minCount <= len(select) <=
select.maxCount`` (`pkm/engine/api.py`); submitting a single index is only
valid where ``maxCount == 1``. Deeper nodes (discards, energy attachment,
prize picks, ...) can require ``maxCount > 1``, and this is reachable well
within the range of simulation counts Task 8 will use (confirmed empirically
-- see task-7-report.md). Each simulation therefore selects
``k = clamp(maxCount, minCount, n_opts)`` distinct option indices by PUCT
score (mirrors `policy.py`'s existing ``select_count``/without-replacement
convention for multi-count decisions) and submits all ``k`` together in one
`search_step` call; children are keyed by the sorted tuple of chosen indices,
and every chosen index's `N`/`W` is updated on backup (marginal per-option
visit/value stats). At a single-count node this reduces exactly to the
brief's one-index-per-call behaviour.
"""

from __future__ import annotations

import math
from typing import Literal, overload

import numpy as np
import torch

from pkm.new_agents.agent_000_dragapult import cabt
from pkm.new_agents.agent_000_dragapult.determinize import DETERMINIZERS
from pkm.new_agents.agent_000_dragapult.features import featurize
from pkm.new_agents.agent_000_dragapult.model import collate
from pkm.new_agents.agent_000_dragapult.policy import select_count


class _Node:
    __slots__ = ("search_id", "obs", "seat", "n_opts", "P", "N", "W", "children", "terminal_v")

    def __init__(self, state: cabt.SearchState):
        self.search_id = state.searchId
        self.obs = state.observation
        self.seat = self.obs.current.yourIndex if self.obs.current else 0
        self.n_opts = len(self.obs.select.option) if self.obs.select else 0
        self.P = np.zeros(self.n_opts, dtype=np.float32)
        self.N = np.zeros(self.n_opts, dtype=np.float32)
        self.W = np.zeros(self.n_opts, dtype=np.float32)
        self.children: dict[tuple[int, ...], "_Node"] = {}
        self.terminal_v: float | None = None


def _evaluate(node: _Node, model) -> float:
    """Fill node.P from priors; return the node's value estimate in [-1, 1]."""
    if node.obs.current is not None and node.obs.current.result >= 0:
        # Terminal: +1 if the node's own seat is the winner, else -1. Matches
        # the task brief's terminal-value convention exactly (result codes
        # other than a decisive 0/1 -- e.g. a draw -- are not exercised by
        # this engine's search nodes in practice; see task-7-report.md).
        node.terminal_v = 1.0 if node.obs.current.result == node.seat else -1.0
        return node.terminal_v
    f = featurize(node.obs)
    b = collate([f])
    with torch.no_grad():
        priors, value = model.evaluate(b)  # softmax over options, scalar value
    p = priors[0, : node.n_opts].cpu().numpy().astype(np.float32)
    s = p.sum()
    node.P = p / s if s > 0 else np.full(node.n_opts, 1.0 / max(node.n_opts, 1), np.float32)
    return float(value[0])


def _select(node: _Node, c_puct: float) -> tuple[int, ...]:
    """PUCT-score the node's options and pick the ``k`` best (without replacement).

    ``k`` matches this decision's own ``minCount``/``maxCount`` (usually 1);
    for ``k > 1`` this greedily takes the top-``k`` PUCT scores as the search
    analogue of `policy.py`'s sampling-time ``select_count``/without-replacement
    convention for multi-count decisions.
    """
    sqrt_total = math.sqrt(max(node.N.sum(), 1.0))
    q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
    u = c_puct * node.P * sqrt_total / (1.0 + node.N)
    scores = q + u
    sel = node.obs.select
    k = select_count(sel.minCount, sel.maxCount, node.n_opts) if sel else 1
    k = max(1, min(k, node.n_opts))
    top_k = np.argpartition(-scores, k - 1)[:k] if k < node.n_opts else np.arange(node.n_opts)
    return tuple(sorted(int(i) for i in top_k))


@overload
def search(
    root_obs: dict, seat: int, model, cfg, gen: torch.Generator,
    return_value: Literal[False] = ...,
) -> np.ndarray: ...
@overload
def search(
    root_obs: dict, seat: int, model, cfg, gen: torch.Generator,
    return_value: Literal[True],
) -> tuple[np.ndarray, float]: ...
def search(
    root_obs: dict,
    seat: int,
    model,
    cfg,
    gen: torch.Generator,
    return_value: bool = False,
) -> "np.ndarray | tuple[np.ndarray, float]":
    """Run PUCT MCTS from `root_obs` (acting seat `seat`); return the root visit policy.

    Owns the engine search lifecycle only (`search_begin`/`search_step` per
    simulation, `search_end` in a `finally`); the caller owns `battle_start`/
    `battle_finish` around this call.

    ``return_value=True`` also returns the MCTS-refined root value estimate in
    ``[-1, 1]`` from the acting seat's perspective — the visit-weighted mean Q
    over the root's children (``sum(W)/sum(N)``), or the net's raw root value
    when no simulation expanded. ExIt uses it as the bootstrap for TD(λ) value
    targets. Default ``False`` keeps the return type a bare policy array so
    existing callers (inference, tests) are unchanged.
    """
    determinize = DETERMINIZERS[cfg.train.determinization]
    # Resolve the seat's deck from the config when present (training passes a full
    # Config; inference's lightweight search-cfg may carry only cfg.run.deck).
    # Falls back to the default deck so a bare search-cfg still works.
    from pkm.new_agents.agent_000_dragapult import deck as _deck

    _deck_name = getattr(getattr(cfg, "run", None), "deck", _deck.DEFAULT_DECK)
    world = determinize(root_obs, seat, gen, _deck.deck_60(_deck_name))
    root_state = cabt.search_begin(
        root_obs,
        your_deck=world.your_deck,
        your_prize=world.your_prize,
        opponent_deck=world.opponent_deck,
        opponent_prize=world.opponent_prize,
        opponent_hand=world.opponent_hand,
        opponent_active=world.opponent_active,
    )
    root = _Node(root_state)
    c_puct = cfg.train.mcts_c_puct

    try:
        root_v0 = _evaluate(root, model)  # net's raw root value (bootstrap fallback)
        for _ in range(cfg.train.mcts_simulations):
            path: list[tuple[_Node, tuple[int, ...]]] = []
            node = root
            leaf_v = 0.0
            # descend to a leaf, expanding exactly one new node per simulation
            while True:
                if node.n_opts == 0 or node.terminal_v is not None:
                    break
                picks = _select(node, c_puct)
                path.append((node, picks))
                if picks in node.children:
                    node = node.children[picks]
                    continue
                child = _Node(cabt.search_step(node.search_id, list(picks)))
                node.children[picks] = child
                leaf_v = _evaluate(child, model)
                node = child
                break

            if not path:
                # Root itself has no legal options or is already terminal --
                # no simulation can proceed; every further iteration would be
                # identical, so stop early rather than spin.
                break

            v = node.terminal_v if node.terminal_v is not None else leaf_v
            # Negamax backup keyed on SEAT (PTCG has multi-decision turns
            # where the same seat acts consecutively, so depth parity would
            # be wrong): +v for edges whose parent's acting seat equals the
            # leaf's seat, -v otherwise. A multi-count decision's visit/value
            # stats are attributed marginally to every option in the picked
            # slate (see module docstring).
            for parent, picks in reversed(path):
                signed = v if parent.seat == node.seat else -v
                for a in picks:
                    parent.N[a] += 1
                    parent.W[a] += signed
    finally:
        cabt.search_end()

    # MCTS-refined root value (acting seat's perspective): visit-weighted mean Q
    # over children, or the net's raw root value when no simulation expanded.
    n_root = float(root.N.sum())
    root_value = float(root.W.sum() / n_root) if n_root > 0 else float(root_v0)

    if root.n_opts == 0 or root.N.sum() == 0:
        # No sims expanded (terminal/optionless root, or the loop above broke
        # on its first iteration) -- fall back to uniform over root options.
        pi = np.full(root.n_opts, 1.0 / max(root.n_opts, 1), dtype=np.float32)
        return (pi, root_value) if return_value else pi

    tau = cfg.train.mcts_temperature
    if tau > 0:
        counts = root.N ** (1.0 / tau)
    else:
        counts = (root.N == root.N.max()).astype(np.float32)
    pi = (counts / counts.sum()).astype(np.float32)
    return (pi, root_value) if return_value else pi


@overload
def search_worlds(
    root_obs: dict, seat: int, model, cfg, gen: torch.Generator,
    n_worlds: int = ..., return_value: Literal[False] = ...,
) -> np.ndarray: ...
@overload
def search_worlds(
    root_obs: dict, seat: int, model, cfg, gen: torch.Generator,
    n_worlds: int, return_value: Literal[True],
) -> tuple[np.ndarray, float]: ...
def search_worlds(
    root_obs: dict,
    seat: int,
    model,
    cfg,
    gen: torch.Generator,
    n_worlds: int = 1,
    return_value: bool = False,
) -> "np.ndarray | tuple[np.ndarray, float]":
    """Average the root visit policy over ``n_worlds`` independent determinizations.

    This is IS-MCTS with W>1 worlds (the full version the module docstring
    promises drops in behind the `DETERMINIZERS` seam). A single-sample
    determinization (`search`, W=1) is *biased*: the search optimizes hard for
    the one guessed layout (deck order / opponent hand), so its move looks great
    only if that exact guess holds. Re-sampling W worlds and averaging their
    per-option root policies makes a move earn its rank across *many* possible
    draws, not one — the errors from any single unlucky/lucky guess wash out.

    Each world re-samples hidden info (advancing ``gen`` and thus the engine
    world) and runs a full independent PUCT `search`; the root option set is the
    real observation's legal options, identical across worlds, so the returned
    per-option policies are directly averageable. ``n_worlds <= 1`` is exactly
    `search`. Cost scales linearly in W (W searches per decision) — trade it
    against `mcts_simulations` under a fixed time budget.
    """
    # Single world: exactly `search` (no kwarg on the value-free path, so
    # monkeypatched 5-arg `search` stand-ins keep working).
    if n_worlds <= 1:
        if return_value:
            return search(root_obs, seat, model, cfg, gen, return_value=True)
        return search(root_obs, seat, model, cfg, gen)

    acc: np.ndarray | None = None
    v_acc = 0.0
    for _ in range(n_worlds):
        if return_value:
            pi, v = search(root_obs, seat, model, cfg, gen, return_value=True)
            v_acc += v
        else:
            pi = search(root_obs, seat, model, cfg, gen)
        pi = pi.astype(np.float64)
        if acc is None:
            acc = pi
        elif pi.shape == acc.shape:
            acc = acc + pi
        else:
            # Root options are the real obs's legal choices and must not vary by
            # determinized world; if they somehow do, align on the shorter length
            # rather than crash (a single-sample approximation is already lossy).
            m = min(acc.shape[0], pi.shape[0])
            acc = acc[:m] + pi[:m]
    pi_avg = (acc / n_worlds).astype(np.float32)  # type: ignore[operator]  # acc set in loop
    return (pi_avg, v_acc / n_worlds) if return_value else pi_avg
