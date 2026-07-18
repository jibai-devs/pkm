import dataclasses
import torch

from pkm.new_agents.agent_000_dragapult.config import (
    Config,
    TrainConfig,
    _hash_dict,
    build_model,
)
from pkm.new_agents.agent_000_dragapult.train import TrainState


def test_new_fields_have_ppo_defaults():
    tc = TrainConfig()
    assert tc.method == "ppo"
    assert tc.mcts_simulations == 32
    assert tc.determinization == "sample"


def test_from_dict_backfills_missing_method():
    # An "old" config dict predating the method field.
    d = Config().to_dict()
    del d["train"]["method"]
    del d["train"]["mcts_simulations"]
    del d["train"]["mcts_c_puct"]
    del d["train"]["mcts_temperature"]
    del d["train"]["determinization"]
    cfg = Config.from_dict(d)  # must not raise; fills defaults
    assert cfg.train.method == "ppo"


def test_load_accepts_old_schema_checkpoint(tmp_path):
    # Save a current checkpoint, then rewrite its blob to look "old"
    # (no method/mcts fields, hash recomputed over the old dict).
    cfg = Config()
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    ts = TrainState(cfg=cfg, model=model, optimizer=opt, update_idx=7)
    path = tmp_path / "latest.pt"
    ts.save(path)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("method", "mcts_simulations", "mcts_c_puct",
              "mcts_temperature", "determinization"):
        blob["config"]["train"].pop(k, None)
    blob["config_hash"] = _hash_dict(blob["config"])
    torch.save(blob, path)

    loaded = TrainState.load(path)  # must not raise "config hash mismatch"
    assert loaded.update_idx == 7
    assert loaded.cfg.train.method == "ppo"
