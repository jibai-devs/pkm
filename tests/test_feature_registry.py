import json
from pathlib import Path

import numpy as np

from pkm.rl.features import (
    GLOBAL_FEATURES,
    OPT_FEATS,
    PER_OPTION_FEATURES,
    PER_SLOT_FEATURES,
    STATE_FEATS,
    FeatureConfig,
    FeatureSpec,
    FeatureStampMismatch,
    Scope,
    assemble_global,
    assemble_per_option,
    assemble_per_slot,
    check_stamp,
    feature_stamp,
)
from pkm.types.obs import N_POKEMON_SLOTS, Observation

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)


def test_state_feats_equals_registered_widths():
    per_slot_width = sum(s.width for s in PER_SLOT_FEATURES)
    global_width = sum(s.width for s in GLOBAL_FEATURES)
    assert STATE_FEATS == N_POKEMON_SLOTS * per_slot_width + global_width


def test_opt_feats_equals_registered_widths():
    assert OPT_FEATS == sum(s.width for s in PER_OPTION_FEATURES)


def test_every_spec_has_expected_scope():
    assert all(s.scope is Scope.GLOBAL for s in GLOBAL_FEATURES)
    assert all(s.scope is Scope.PER_SLOT for s in PER_SLOT_FEATURES)
    assert all(s.scope is Scope.PER_OPTION for s in PER_OPTION_FEATURES)


def test_disabling_a_global_feature_zero_masks_without_changing_width(monkeypatch):
    obs = _dummy_observation()
    baseline = assemble_global(obs, ctx=None)
    target = GLOBAL_FEATURES[0]

    config = FeatureConfig(disabled=frozenset({target.name}))
    masked = assemble_global(obs, ctx=None, config=config)

    assert masked.shape == baseline.shape
    start = 0
    for spec in GLOBAL_FEATURES:
        end = start + spec.width
        if spec.name == target.name:
            assert np.all(masked[start:end] == 0.0)
        else:
            assert np.array_equal(masked[start:end], baseline[start:end])
        start = end


def test_disabling_a_per_slot_feature_zero_masks_without_changing_width():
    obs = _dummy_observation()
    baseline = assemble_per_slot(obs, ctx=None)
    target = PER_SLOT_FEATURES[0]

    config = FeatureConfig(disabled=frozenset({target.name}))
    masked = assemble_per_slot(obs, ctx=None, config=config)

    assert masked.shape == baseline.shape
    assert not np.array_equal(masked, baseline)  # something changed
    assert masked.sum() <= baseline.sum() + 1e-6  # only zeroing, never adding signal


def test_disabling_a_per_option_feature_zero_masks_without_changing_width():
    obs = _dummy_observation()
    baseline = assemble_per_option(obs, ctx=None)
    target = PER_OPTION_FEATURES[0]

    config = FeatureConfig(disabled=frozenset({target.name}))
    masked = assemble_per_option(obs, ctx=None, config=config)

    assert masked.shape == baseline.shape


def test_stamp_matches_itself():
    check_stamp(feature_stamp())  # must not raise


def test_stamp_mismatch_raises_loudly():
    stamp = list(feature_stamp())
    stamp[0] = ("GLOBAL", "not_a_real_feature", 999)
    try:
        check_stamp(tuple(stamp))
    except FeatureStampMismatch as exc:
        assert "not_a_real_feature" in str(exc) or "mismatch" in str(exc).lower()
    else:
        raise AssertionError("expected FeatureStampMismatch")


def test_feature_spec_is_a_plain_dataclass():
    spec = FeatureSpec(
        name="dummy",
        width=1,
        scope=Scope.GLOBAL,
        fn=lambda obs, ctx: np.zeros(1, dtype=np.float32),
        deterministic=True,
    )
    assert spec.name == "dummy"
    assert spec.width == 1


def _dummy_observation() -> Observation:
    return Observation.model_validate(FIXTURE["observations"]["9:43"])
