import torch

from pkm.new_agents.agent_000_dragapult.config import Config, build_model
from pkm.new_agents.agent_000_dragapult.parallel import ParallelRollout


def test_parallel_collect_takes_trainer_and_returns_samples():
    cfg = Config()
    torch.manual_seed(0)
    model = build_model(cfg)
    from pkm.new_agents.agent_000_dragapult.trainers.ppo import PpoTrainer

    pool = ParallelRollout(cfg, num_workers=2, base_seed=0)
    try:
        samples, stats = pool.collect(PpoTrainer(), total_games=4)
    finally:
        pool.close()
    assert stats["games"] == 4
    assert len(samples) > 0
