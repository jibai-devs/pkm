"""Inference-time MCTS wiring: the InferenceConfig toggle + bundle round-trip.

The fast tests here need no engine (they exercise the config logic and the
pack-bundle -> from_checkpoint path). The engine-backed smoke test that MCTS
actually produces a legal move from a live observation is marked ``slow``.
"""

from pathlib import Path

import pytest
import torch

from pkm.new_agents.agent_000_dragapult import cabt, deck
from pkm.new_agents.agent_000_dragapult.agent import DragapultAgent, InferenceConfig
from pkm.new_agents.agent_000_dragapult.config import Config, build_model


def test_use_mcts_toggle():
    # policy mode never searches, whatever the budget
    assert not InferenceConfig(type="policy", mcts_sims=64).use_mcts
    # mcts mode with K=0 is disabled (the "K=0 turns MCTS off" contract)
    assert not InferenceConfig(type="mcts", mcts_sims=0).use_mcts
    # mcts mode with K>0 is on
    assert InferenceConfig(type="mcts", mcts_sims=8).use_mcts
    # default is plain policy
    assert not InferenceConfig().use_mcts


def test_inference_config_roundtrip():
    inf = InferenceConfig(type="mcts", mcts_sims=16, c_puct=2.0, temperature=0.5)
    assert InferenceConfig.from_dict(inf.to_dict()) == inf
    # unknown keys are ignored, missing ones backfilled (forward/backward compat)
    loaded = InferenceConfig.from_dict({"type": "mcts", "mcts_sims": 4, "bogus": 1})
    assert loaded.type == "mcts" and loaded.mcts_sims == 4
    assert loaded.c_puct == InferenceConfig().c_puct


def test_bundle_embeds_and_loads_inference(tmp_path: Path):
    """A packed bundle carrying an inference config configures the loaded agent."""
    model = build_model(Config())
    inf = InferenceConfig(type="mcts", mcts_sims=8, c_puct=1.5)
    bundle = tmp_path / "weights.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": None,
            "inference": inf.to_dict(),
        },
        bundle,
    )
    agent = DragapultAgent.from_checkpoint(str(bundle), greedy=True)
    assert agent.inference == inf
    assert agent.inference.use_mcts


def test_from_checkpoint_defaults_to_policy_when_no_inference(tmp_path: Path):
    """Legacy/plain bundles (no inference key) load as plain policy, not MCTS."""
    model = build_model(Config())
    bundle = tmp_path / "weights.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": None}, bundle)
    agent = DragapultAgent.from_checkpoint(str(bundle), greedy=True)
    assert not agent.inference.use_mcts


def test_explicit_inference_overrides_bundle(tmp_path: Path):
    """An explicit `inference=` at load time wins over the bundle's own config."""
    model = build_model(Config())
    bundle = tmp_path / "weights.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": None,
            "inference": InferenceConfig(type="mcts", mcts_sims=8).to_dict(),
        },
        bundle,
    )
    override = InferenceConfig(type="policy")
    agent = DragapultAgent.from_checkpoint(
        str(bundle), greedy=True, inference=override
    )
    assert not agent.inference.use_mcts


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
def test_mcts_agent_returns_legal_move():
    """End-to-end: an MCTS-configured agent picks legal option indices in a game."""
    torch.manual_seed(0)
    model = build_model(Config())
    agent = DragapultAgent(
        model=model,
        greedy=True,
        seed=0,
        inference=InferenceConfig(type="mcts", mcts_sims=8),
    )
    obs = _root_obs()
    try:
        if obs["current"]["result"] >= 0:
            pytest.skip("game ended during setup")
        n_opts = len(obs["select"]["option"])
        picks = agent(obs)
    finally:
        cabt.battle_finish()

    assert 1 <= len(picks) <= n_opts
    assert len(set(picks)) == len(picks)
    assert all(0 <= i < n_opts for i in picks)
