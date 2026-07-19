"""Tests for configurable network size/depth (ModelConfig presets + build_model)."""

from __future__ import annotations

import pytest

import torch

from pkm.new_agents.agent_000_dragapult.config import (
    Config,
    ModelConfig,
    build_model,
    build_model_config,
    resolve_device,
)


def test_small_preset_is_v1_defaults():
    # "small" must equal the untouched ModelConfig defaults (n_layers == 1),
    # so it stays bit-for-bit checkpoint-compatible with old runs.
    assert build_model_config("small") == ModelConfig()
    assert ModelConfig().n_layers == 1


def test_presets_scale_depth_and_width():
    m = build_model_config("medium")
    assert (m.n_layers, m.d_state, m.n_heads) == (2, 256, 8)
    xl = build_model_config("xl")
    assert (xl.n_layers, xl.d_state) == (4, 512)


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown model preset"):
        build_model_config("gigantic")


def test_overrides_win_and_none_is_ignored():
    m = build_model_config("small", {"n_layers": 3, "d_state": None})
    assert m.n_layers == 3  # override applied
    assert m.d_state == 128  # None ignored -> preset/default kept


def test_extra_layers_count_matches_depth():
    # n_layers == 1 -> zero extra params (old checkpoints load unchanged).
    assert len(build_model(build_model_config("small")).encoder.extra_layers) == 0
    assert len(build_model(build_model_config("medium")).encoder.extra_layers) == 1
    assert len(build_model(build_model_config("large")).encoder.extra_layers) == 2


def test_deeper_wider_has_more_params():
    small = sum(p.numel() for p in build_model(build_model_config("small")).parameters())
    large = sum(p.numel() for p in build_model(build_model_config("large")).parameters())
    assert large > small


def test_small_statedict_keys_are_backward_compatible():
    # The v1 encoder had no `extra_layers`; a small model must not introduce any
    # such parameter keys, or old checkpoints would fail to load.
    sd = build_model(build_model_config("small")).state_dict()
    assert not any("extra_layers" in k for k in sd)


def test_config_roundtrip_preserves_depth():
    cfg = Config(model=build_model_config("large"))
    rt = Config.from_dict(cfg.to_dict())
    assert rt.model.n_layers == 3 and rt.model.d_state == 384
    assert cfg.hash() != Config(model=build_model_config("small")).hash()


def test_from_dict_backfills_missing_depth_fields():
    # An old checkpoint's model dict has no n_layers/ff_mult/dropout -> defaults.
    d = Config().to_dict()
    for k in ("n_layers", "ff_mult", "dropout"):
        d["model"].pop(k, None)
    cfg = Config.from_dict(d)
    assert cfg.model.n_layers == 1


# --------------------------------------------------------------------------- #
# device selection
# --------------------------------------------------------------------------- #


def test_resolve_device_cpu_and_auto():
    assert resolve_device("cpu") == "cpu"
    # auto resolves to whatever is actually available (cpu on a CPU-only build).
    assert resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")


def test_resolve_device_unknown_raises():
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")


def test_resolve_device_cuda_unavailable_raises_clearly():
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
    else:
        with pytest.raises(RuntimeError, match="cuda"):
            resolve_device("cuda")


def test_device_is_not_in_config_hash():
    # Device is a runtime choice, not part of Config -> cpu/cuda are one experiment.
    assert "device" not in Config().to_dict().get("train", {})
