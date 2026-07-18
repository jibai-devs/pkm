import pytest
import torch

from pkm.new_agents.agent_000_dragapult.config import Config, build_model
from pkm.new_agents.agent_000_dragapult.trainers.exit import ExItTrainer, ExItSample

pytestmark = pytest.mark.slow


def test_exit_collect_then_update_smoke():
    cfg = Config()
    object.__setattr__(cfg.train, "method", "exit")
    object.__setattr__(cfg.train, "mcts_simulations", 6)
    torch.manual_seed(0)
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    trainer = ExItTrainer()

    samples, stats = trainer.collect(model, n_games=1, cfg=cfg)
    assert len(samples) > 0
    assert all(isinstance(s, ExItSample) for s in samples)
    # value targets filled to ±1 (zero-sum) or 0 (draw)
    assert all(s.value_target in (-1.0, 0.0, 1.0) for s in samples)

    upd = trainer.update(model, opt, samples, cfg)
    assert "policy_loss" in upd and upd["policy_loss"] == upd["policy_loss"]
    assert "value_loss" in upd and upd["value_loss"] == upd["value_loss"]
