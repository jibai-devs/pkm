import numpy as np
import pytest
import torch

from pkm.new_agents.agent_000_dragapult import cabt, deck, mcts
from pkm.new_agents.agent_000_dragapult.config import Config, build_model

pytestmark = pytest.mark.slow


def _root_obs():
    obs, _ = cabt.battle_start(deck.DECK_60, deck.DECK_60)
    n = 0
    while obs["select"] is None or obs["current"] is None:
        obs = cabt.battle_select(list(deck.DECK_60))
        n += 1
        if n > 50:
            break
    return obs


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
