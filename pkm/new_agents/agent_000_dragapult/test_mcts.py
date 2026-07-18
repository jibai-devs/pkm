from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pkm.new_agents.agent_000_dragapult import cabt, deck, mcts
from pkm.new_agents.agent_000_dragapult.config import Config, build_model


def _root_obs():
    obs, _ = cabt.battle_start(deck.DECK_60, deck.DECK_60)
    n = 0
    while obs["select"] is None or obs["current"] is None:
        obs = cabt.battle_select(list(deck.DECK_60))
        n += 1
        if n > 50:
            break
    return obs


@pytest.mark.slow
def test_search_returns_valid_policy():
    cfg = Config()
    object.__setattr__(cfg.train, "mcts_simulations", 8)
    torch.manual_seed(0)
    model = build_model(cfg)
    obs = _root_obs()
    if obs["current"]["result"] >= 0:
        cabt.battle_finish()
        pytest.skip("game ended during setup")
    seat = obs["current"]["yourIndex"]
    gen = torch.Generator().manual_seed(0)
    try:
        pi = mcts.search(obs, seat, model, cfg, gen)
    finally:
        cabt.battle_finish()

    n_opts = len(obs["select"]["option"])
    assert pi.shape == (n_opts,)
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)
    assert (pi >= 0).all()


def _make_node(min_count: int, max_count: int, n_opts: int) -> mcts._Node:
    """Build a minimal, duck-typed `_Node` for `_select` without touching the engine.

    `_Node.__init__` only reads `state.searchId` / `state.observation`, and
    `_select` only reads `node.obs.select.{option,minCount,maxCount}` plus
    `node.n_opts`/`node.P`/`node.N`/`node.W` (all zero-initialized from
    `n_opts`) -- so a `SimpleNamespace` stand-in is sufficient, no real
    `cabt.SearchState`/`Observation` required.
    """
    select = SimpleNamespace(option=[None] * n_opts, minCount=min_count, maxCount=max_count)
    current = SimpleNamespace(yourIndex=0, result=-1)
    obs = SimpleNamespace(select=select, current=current)
    state = SimpleNamespace(searchId=0, observation=obs)
    return mcts._Node(state)


@pytest.mark.parametrize(
    "min_count,max_count,n_opts,expected_len",
    [
        (1, 1, 4, 1),  # single-count decision (the only case the slow test hits)
        (1, 3, 5, 3),  # multi-count, plenty of options
        (2, 4, 3, 3),  # n_opts < maxCount -- clamps to n_opts
        (1, 3, 2, 2),  # n_opts < maxCount, minCount satisfied by clamp
    ],
)
def test_select_slate_length_invariant(min_count, max_count, n_opts, expected_len):
    """`_select`'s slate must always satisfy minCount <= len(slate) <= maxCount.

    This is the engine-required invariant (`minCount <= len(select) <=
    maxCount`, enforced by `search_step`) that the multi-count slate/keying
    deviation exists to satisfy. It was previously only exercised implicitly
    by the slow, single-count (k=1) engine test -- this locks it directly and
    fast, including maxCount > 1 and n_opts < maxCount cases.
    """
    node = _make_node(min_count, max_count, n_opts)

    slate = mcts._select(node, c_puct=1.5)

    assert len(slate) == expected_len
    assert min_count <= len(slate) <= max_count
    assert len(slate) <= n_opts
    assert len(set(slate)) == len(slate), "picks must be distinct"
    assert all(0 <= i < n_opts for i in slate), "picks must be valid option indices"
