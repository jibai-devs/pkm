"""Task 5 safety nets: checkpoint-compatibility stamping (export.py/profile.py
wiring) and numpy/torch forward-pass parity.

Both guard the same failure mode from opposite ends: a registry change that
alters STATE_FEATS/OPT_FEATS without also updating what's loaded into the
network would otherwise silently misalign tensor slices instead of raising.
"""

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pkm.agents.profile import AgentProfile
from pkm.rl.encoder import EncodedDecision
import pkm.agents.profile as profile_module
from pkm.rl.export import export_checkpoint, export_npz
from pkm.rl.features import (
    OPT_FEATS,
    STATE_FEATS,
    FeatureStampMismatch,
    feature_stamp,
    stamp_sidecar_path,
    write_stamp_sidecar,
)
from pkm.rl.model import OPT_ENC, PolicyValueNet
from pkm.rl.numpy_policy import NEG_INF, NumpyPolicy
from pkm.types.obs import MAX_HAND, N_BOARD_SLOTS, NUM_ATTACKS, NUM_CARDS, NUM_OPT_TYPES

BAD_STAMP = json.dumps([["GLOBAL", "not_a_real_feature", 999]])


# --- checkpoint-compatibility stamping --------------------------------------


def test_export_npz_embeds_feature_stamp(tmp_path):
    model = PolicyValueNet()
    out = tmp_path / "policy.npz"
    export_npz(model, str(out))
    with np.load(out) as z:
        assert "__feature_stamp__" in z.files
        stamp = tuple(tuple(x) for x in json.loads(str(z["__feature_stamp__"])))
    assert stamp == feature_stamp()


def test_numpy_policy_load_rejects_stale_npz_stamp(tmp_path):
    model = PolicyValueNet()
    out = tmp_path / "policy.npz"
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(out, __feature_stamp__=np.array(BAD_STAMP), **arrays)
    with pytest.raises(FeatureStampMismatch):
        NumpyPolicy.load(str(out))


def test_numpy_policy_load_allows_npz_with_no_stamp(tmp_path):
    """Legacy .npz exported before Task 5 has no stamp key -- can't verify,
    must not hard-fail."""
    model = PolicyValueNet()
    out = tmp_path / "policy.npz"
    export_npz_without_stamp(model, str(out))
    NumpyPolicy.load(str(out))  # must not raise


def export_npz_without_stamp(model: PolicyValueNet, path: str) -> None:
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(path, **arrays)


def test_export_checkpoint_rejects_stale_pt_stamp(tmp_path):
    model = PolicyValueNet()
    ckpt = tmp_path / "ppo_latest.pt"
    torch.save(model.state_dict(), ckpt)
    stamp_sidecar_path(ckpt).write_text(BAD_STAMP)

    with pytest.raises(FeatureStampMismatch):
        export_checkpoint(str(ckpt), str(tmp_path / "policy.npz"))


def test_export_checkpoint_allows_pt_with_no_sidecar(tmp_path):
    model = PolicyValueNet()
    ckpt = tmp_path / "ppo_latest.pt"
    torch.save(model.state_dict(), ckpt)

    export_checkpoint(str(ckpt), str(tmp_path / "policy.npz"))  # must not raise


def test_export_checkpoint_allows_pt_with_matching_sidecar(tmp_path):
    model = PolicyValueNet()
    ckpt = tmp_path / "ppo_latest.pt"
    torch.save(model.state_dict(), ckpt)
    write_stamp_sidecar(ckpt)

    export_checkpoint(str(ckpt), str(tmp_path / "policy.npz"))  # must not raise


def test_profile_latest_checkpoint_rejects_stale_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_module, "AGENTS_DIR", tmp_path / "agents")
    prof = AgentProfile("test_agent")
    prof.ensure_dirs()
    ckpt = prof.checkpoint_dir / "ppo_latest.pt"
    torch.save({}, ckpt)
    stamp_sidecar_path(ckpt).write_text(BAD_STAMP)

    with pytest.raises(FeatureStampMismatch):
        prof.latest_checkpoint("ppo")


def test_profile_latest_checkpoint_allows_missing_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_module, "AGENTS_DIR", tmp_path / "agents")
    prof = AgentProfile("test_agent")
    prof.ensure_dirs()
    ckpt = prof.checkpoint_dir / "ppo_latest.pt"
    torch.save({}, ckpt)

    assert prof.latest_checkpoint("ppo") == ckpt


def test_profile_latest_checkpoint_allows_matching_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_module, "AGENTS_DIR", tmp_path / "agents")
    prof = AgentProfile("test_agent")
    prof.ensure_dirs()
    ckpt = prof.checkpoint_dir / "ppo_latest.pt"
    torch.save({}, ckpt)
    write_stamp_sidecar(ckpt)

    assert prof.latest_checkpoint("ppo") == ckpt


# --- numpy/torch forward-pass parity ----------------------------------------


def _random_decision(rng: np.random.Generator, n_options: int) -> EncodedDecision:
    opt_type = rng.integers(0, NUM_OPT_TYPES, size=n_options)
    opt_card = rng.integers(0, NUM_CARDS, size=n_options)
    opt_card2 = rng.integers(0, NUM_CARDS, size=n_options)
    opt_attack = rng.integers(0, NUM_ATTACKS, size=n_options)
    opt_feats = rng.standard_normal((n_options, OPT_FEATS)).astype(np.float32)
    max_count = int(rng.integers(1, n_options + 1))
    min_count = int(rng.integers(0, max_count + 1))
    k_deck = int(rng.integers(0, 10))
    return EncodedDecision(
        board_cards=rng.integers(0, NUM_CARDS, size=N_BOARD_SLOTS).astype(np.int64),
        hand_cards=rng.integers(0, NUM_CARDS, size=MAX_HAND).astype(np.int64),
        state_feats=rng.standard_normal(STATE_FEATS).astype(np.float32),
        deck_card_ids=rng.integers(1, NUM_CARDS, size=k_deck).astype(np.int64),
        deck_card_counts=rng.integers(1, 5, size=k_deck).astype(np.float32),
        opt_type=opt_type.astype(np.int64),
        opt_card=opt_card.astype(np.int64),
        opt_card2=opt_card2.astype(np.int64),
        opt_attack=opt_attack.astype(np.int64),
        opt_feats=opt_feats,
        min_count=min_count,
        max_count=max_count,
    )


def _torch_priors(model: PolicyValueNet, d: EncodedDecision) -> np.ndarray:
    """First-pick priors, mirroring NumpyPolicy.priors() exactly."""
    with torch.no_grad():
        board = torch.from_numpy(d.board_cards).unsqueeze(0)
        hand = torch.from_numpy(d.hand_cards).unsqueeze(0)
        feats = torch.from_numpy(d.state_feats).unsqueeze(0)
        deck_ids = torch.from_numpy(d.deck_card_ids).unsqueeze(0)
        deck_counts = torch.from_numpy(d.deck_card_counts).unsqueeze(0)
        h = model.encode_state(board, hand, feats, deck_ids, deck_counts)
        opts = model.encode_options(
            torch.from_numpy(d.opt_type).unsqueeze(0),
            torch.from_numpy(d.opt_card).unsqueeze(0),
            torch.from_numpy(d.opt_card2).unsqueeze(0),
            torch.from_numpy(d.opt_attack).unsqueeze(0),
            torch.from_numpy(d.opt_feats).unsqueeze(0),
        )
        n = len(d.opt_type)
        mask = torch.ones(1, n + 1, dtype=torch.bool)
        picked_sum = torch.zeros(1, OPT_ENC)
        logits = model.option_logits(h, opts, picked_sum, mask)
        logits[0, -1] = NEG_INF
        probs = F.softmax(logits, dim=-1)[0, :-1]
        return probs.numpy()


def test_numpy_torch_parity_on_random_decisions():
    torch.manual_seed(0)
    model = PolicyValueNet()
    model.eval()
    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np_pol = NumpyPolicy(weights)

    rng = np.random.default_rng(0)
    for _ in range(30):
        n_options = int(rng.integers(1, 12))
        d = _random_decision(rng, n_options)

        t_value = float(model.act(d, greedy=True).value)
        n_value = np_pol.value(d)
        assert abs(t_value - n_value) < 1e-4

        t_priors = _torch_priors(model, d)
        n_priors = np_pol.priors(d)
        assert t_priors.shape == n_priors.shape
        assert np.abs(t_priors - n_priors).max() < 1e-4

        t_belief = model.act(d, greedy=True).belief
        n_belief = np_pol.archetype_belief(d)
        assert t_belief.shape == n_belief.shape
        assert np.abs(t_belief - n_belief).max() < 1e-4
