import pytest

from pkm.new_agents.agent_000_dragapult.config import Config
from pkm.new_agents.agent_000_dragapult import trainers


def test_get_trainer_unknown_method_raises():
    cfg = Config()
    object.__setattr__(cfg.train, "method", "nope")  # frozen dataclass
    with pytest.raises(ValueError, match="unknown training method 'nope'"):
        trainers.get_trainer(cfg)


def test_registry_is_a_dict():
    assert isinstance(trainers.TRAINERS, dict)
