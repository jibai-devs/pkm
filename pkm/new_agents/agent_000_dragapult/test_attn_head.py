"""Tests for the transformer-decoder policy head (policy_head="attn").

Guarantees (mirrors test_combo_head.py / test_autoreg_head.py):
  * opt-in + backward compat: default is still "marginal"; attn builds a
    different module and hashes differently; old configs backfill to "marginal";
  * the [B,L] head keeps its meaning (per-option logits, padding-masked) so
    sampling / MCTS `evaluate` / ExIt / inference-MCTS all work unchanged — attn
    reuses the marginal sampling path verbatim (no policy.py changes);
  * the decoder actually contextualizes options — changing one option's features
    moves ANOTHER option's logit (self-attention couples the slate), which the
    marginal per-option scorer provably does NOT; and cross-attention to the
    board makes a logit depend on the board;
  * a sampled action's returned logprob equals its batched recompute (the core
    consistency invariant the PPO trainer relies on);
  * gradients flow into the decoder; degenerate rows never NaN.
"""

from __future__ import annotations

import numpy as np
import torch

from pkm.new_agents.agent_000_dragapult import deck, policy
from pkm.new_agents.agent_000_dragapult.config import (
    Config,
    ModelConfig,
    build_model,
)
from pkm.new_agents.agent_000_dragapult.features import F, G, O, Features
from pkm.new_agents.agent_000_dragapult.model import MASK_FILL, collate

VOCAB = deck.VOCAB_SIZE


def _feats(
    n_opts: int,
    min_count: int,
    max_count: int,
    *,
    active: bool = True,
) -> Features:
    """Minimal Features with `n_opts` distinguishable board-less options."""
    ent_feat = np.zeros((12, F), dtype=np.float32)
    mask = np.zeros(12, dtype=np.float32)
    if active:
        mask[0] = 1.0
        ent_feat[0, 0] = 0.5
    ofeat = np.zeros((n_opts, O), dtype=np.float32)
    for i in range(n_opts):  # make options distinguishable
        ofeat[i, 6] = float(i + 1)  # arbitrary distinct "count" column
    return Features(
        entity_id_row=np.zeros(12, dtype=np.int64),
        entity_card_id=np.zeros(12, dtype=np.int64),
        entity_feat=ent_feat,
        entity_mask=mask,
        hand_hist=np.zeros(VOCAB, dtype=np.float32),
        discard_hist=np.zeros(VOCAB, dtype=np.float32),
        globals=np.zeros(G, dtype=np.float32),
        option_type=np.zeros(n_opts, dtype=np.int64),
        option_feat=ofeat,
        option_entity_slot=np.full(n_opts, -1, dtype=np.int64),
        option_card_id=np.zeros(n_opts, dtype=np.int64),
        option_card_row=np.full(n_opts, deck.UNK_ROW, dtype=np.int64),
        option_attack_id=np.zeros(n_opts, dtype=np.int64),
        select_type=0,
        select_context=0,
        min_count=min_count,
        max_count=max_count,
    )


def _attn_model():
    torch.manual_seed(0)
    return build_model(ModelConfig(policy_head="attn")).eval()


# --------------------------------------------------------------------------- #
# Opt-in + backward compatibility
# --------------------------------------------------------------------------- #


def test_default_is_marginal() -> None:
    assert build_model().policy_head == "marginal"
    assert ModelConfig().policy_head == "marginal"


def test_attn_builds_named_module() -> None:
    m = build_model(ModelConfig(policy_head="attn"))
    assert m.policy_head == "attn"
    assert hasattr(m, "decoder") and not hasattr(m, "scorer")
    # marginal has the scorer MLP, not the decoder
    mm = build_model()
    assert hasattr(mm, "scorer") and not hasattr(mm, "decoder")


def test_hash_differs_across_heads() -> None:
    marg = Config(model=ModelConfig(policy_head="marginal"))
    attn = Config(model=ModelConfig(policy_head="attn"))
    combo = Config(model=ModelConfig(policy_head="combo"))
    assert len({marg.hash(), attn.hash(), combo.hash()}) == 3


def test_old_config_backfills_to_marginal() -> None:
    # An old checkpoint config with neither policy_head nor n_dec_layers must
    # backfill to marginal (+ the default decoder depth), i.e. still load.
    d = Config(model=ModelConfig(policy_head="attn")).to_dict()
    d["model"].pop("policy_head")
    d["model"].pop("n_dec_layers")
    cfg = Config.from_dict(d)
    assert cfg.model.policy_head == "marginal"
    assert cfg.model.n_dec_layers == ModelConfig().n_dec_layers


def test_unknown_policy_head_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_model(ModelConfig(policy_head="nope"))


def test_n_dec_layers_configurable() -> None:
    m = build_model(ModelConfig(policy_head="attn", n_dec_layers=3))
    assert len(m.decoder.layers) == 3


# --------------------------------------------------------------------------- #
# [B,L] contract -> sampling / MCTS / ExIt / inference untouched
# --------------------------------------------------------------------------- #


def test_forward_shapes_and_finite() -> None:
    m = _attn_model()
    b = collate([_feats(4, 1, 1), _feats(3, 0, 2)])
    logits, value = m(b)
    assert logits.shape == (2, 4)  # padded to max L
    assert value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_padding_masked_to_sentinel() -> None:
    m = _attn_model()
    b = collate([_feats(4, 1, 1), _feats(2, 1, 1)])  # row 1 has 2 real, 2 padded
    logits, _ = m(b)
    # padded slots (indices 2,3 of row 1) are the MASK_FILL sentinel
    assert torch.allclose(logits[1, 2:], torch.full((2,), MASK_FILL))
    # real slots are finite, not the sentinel
    assert torch.isfinite(logits[1, :2]).all()
    assert (logits[1, :2] > MASK_FILL / 2).all()


def test_evaluate_returns_valid_priors() -> None:
    # MCTS calls model.evaluate; priors must be a distribution over real options.
    m = _attn_model()
    b = collate([_feats(4, 1, 2)])
    priors, value = m.evaluate(b)
    assert priors.shape == (1, 4)
    assert torch.isfinite(value).all()
    assert torch.allclose(priors[0].sum(), torch.tensor(1.0), atol=1e-5)
    assert (priors >= 0).all()


def test_forward_matches_policy_from_state() -> None:
    m = _attn_model()
    b = collate([_feats(5, 1, 2), _feats(3, 1, 1)])
    with torch.no_grad():
        logits, value = m(b)
        state, ent = m.encode(b)
        logits2 = m.policy_from_state(state, ent, b)
        value2 = m.value_from_state(state)
    assert torch.equal(logits, logits2)
    assert torch.equal(value, value2)


# --------------------------------------------------------------------------- #
# The decoder actually contextualizes options (the whole point)
# --------------------------------------------------------------------------- #


def test_options_are_coupled_unlike_marginal() -> None:
    # Two board-less options; perturb option 1's features only and read option 0.
    # attn: self-attention couples them -> option 0 logit MOVES.
    # marginal: option 0 is scored independently -> option 0 logit UNCHANGED.
    base = _feats(2, 1, 1)
    pert = _feats(2, 1, 1)
    pert.option_feat[1, 0] = 9.0  # change ONLY option 1

    for head, should_change in (("attn", True), ("marginal", False)):
        torch.manual_seed(0)
        m = build_model(ModelConfig(policy_head=head)).eval()
        with torch.no_grad():
            l0 = m(collate([base]))[0][0, 0]
            l1 = m(collate([pert]))[0][0, 0]
        moved = not torch.allclose(l0, l1, atol=1e-6)
        assert moved == should_change, f"{head}: option-0 moved={moved}"


def test_logit_depends_on_board() -> None:
    # cross-attention: changing the board (an occupied entity's features) moves an
    # option's logit even though the option itself is unchanged.
    m = _attn_model()
    a = _feats(3, 1, 1)
    b_ = _feats(3, 1, 1)
    b_.entity_feat[0, 0] = 7.0  # change the active entity only
    with torch.no_grad():
        la = m(collate([a]))[0][0, 0]
        lb = m(collate([b_]))[0][0, 0]
    assert not torch.allclose(la, lb, atol=1e-6)


# --------------------------------------------------------------------------- #
# PPO consistency + gradients + robustness
# --------------------------------------------------------------------------- #


def test_sampled_logprob_matches_recompute() -> None:
    # The invariant the trainer relies on: a sampled action's logprob equals its
    # batched recompute under the same [B,L] logits (attn reuses the marginal path).
    m = _attn_model()
    b = collate([_feats(5, 1, 3)])
    with torch.no_grad():
        logits, _ = m(b)
    valid = b["option_mask"][0].bool()
    gen = torch.Generator().manual_seed(7)
    for _ in range(50):
        picks, lp = policy.sample_action(logits[0], valid, k=3, gen=gen)
        actions = torch.tensor([picks], dtype=torch.long)
        action_len = torch.tensor([len(picks)], dtype=torch.long)
        lp_re = policy.batched_action_logprob(
            logits, b["option_mask"], actions, action_len
        )[0]
        assert abs(lp - float(lp_re)) < 1e-4


def test_gradients_flow_into_decoder() -> None:
    m = build_model(ModelConfig(policy_head="attn")).train()
    b = collate([_feats(4, 1, 2), _feats(3, 1, 1)])
    logits, value = m(b)
    ent = policy.batched_entropy(logits, b["option_mask"]).sum()
    loss = logits[b["option_mask"].bool()].sum() + value.sum() + ent
    loss.backward()
    grads = [
        p.grad for p in m.decoder.parameters() if p.requires_grad
    ]
    assert grads and all(g is not None for g in grads)
    assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_degenerate_rows_never_nan() -> None:
    # A single-option row batched next to a wide row (heavy padding) must stay
    # finite everywhere it matters, and an empty board must not NaN either.
    m = _attn_model()
    b = collate([_feats(1, 1, 1), _feats(8, 1, 4), _feats(2, 1, 1, active=False)])
    logits, value = m(b)
    assert torch.isfinite(value).all()
    real = b["option_mask"].bool()
    assert torch.isfinite(logits[real]).all()
