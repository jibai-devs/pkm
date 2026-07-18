import torch

from pkm.new_agents.agent_000_dragapult.config import Config, build_model
from pkm.new_agents.agent_000_dragapult.trainers.ppo import PpoTrainer, Step


def test_ppo_trainer_collect_then_update_smoke():
    cfg = Config()
    torch.manual_seed(0)
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    trainer = PpoTrainer()

    samples, stats = trainer.collect(model, n_games=2, cfg=cfg)
    assert isinstance(samples, list) and len(samples) > 0
    assert all(isinstance(s, Step) for s in samples)
    assert stats["games"] == 2

    upd = trainer.update(model, opt, samples, cfg)
    for k in ("pol_loss", "val_loss", "entropy", "explained_var"):
        assert k in upd
        assert upd[k] == upd[k]  # not NaN
