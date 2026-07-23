"""Tests for the autoregressive STOP-token policy head (policy_head="autoreg").

Guarantees:
  * opt-in + backward compat: default is still "marginal"; the two build
    different modules and hash differently; old configs backfill to "marginal";
  * the [B,L] head keeps its meaning (step-0 marginal) so MCTS `evaluate` works;
  * sampling respects the count bounds (minCount<=m<=maxCount, distinct, valid)
    and STOP is only ever taken once minCount is met;
  * a sampled action's returned logprob equals its batched recompute (the core
    consistency invariant the trainer relies on);
  * gradients flow into the autoregressive head; degenerate rows never NaN.
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
from pkm.new_agents.agent_000_dragapult.model import collate

VOCAB = deck.VOCAB_SIZE


def _feats(n_opts: int, min_count: int, max_count: int) -> Features:
    """Minimal Features with `n_opts` distinguishable board-less options."""
    ent_feat = np.zeros((12, F), dtype=np.float32)
    mask = np.zeros(12, dtype=np.float32)
    mask[0] = 1.0
    ent_feat[0, 0] = 0.5
    ofeat = np.zeros((n_opts, O), dtype=np.float32)
    for i in range(n_opts):  # make options distinguishable
        ofeat[i, 6] = float(i + 1)  # "count" column, arbitrary distinct values
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


def _autoreg_model():
    torch.manual_seed(0)
    return build_model(ModelConfig(policy_head="autoreg")).eval()


# --------------------------------------------------------------------------- #
# Opt-in + backward compatibility
# --------------------------------------------------------------------------- #


def test_default_is_marginal() -> None:
    m = build_model()  # default ModelConfig
    assert m.policy_head == "marginal"
    assert hasattr(m, "scorer") and not hasattr(m, "autoreg")


def test_autoreg_builds_the_head_only() -> None:
    m = build_model(ModelConfig(policy_head="autoreg"))
    assert m.policy_head == "autoreg"
    assert hasattr(m, "autoreg") and not hasattr(m, "scorer")


def test_config_hash_differs_and_backfills() -> None:
    marg = Config(model=ModelConfig(policy_head="marginal"))
    auto = Config(model=ModelConfig(policy_head="autoreg"))
    assert marg.hash() != auto.hash()
    # An old checkpoint config with no policy_head key must backfill to marginal.
    d = marg.to_dict()
    d["model"].pop("policy_head")
    assert Config.from_dict(d).model.policy_head == "marginal"


def test_unknown_policy_head_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_model(ModelConfig(policy_head="nope"))


# --------------------------------------------------------------------------- #
# [B,L] head keeps its meaning (step-0 marginal) -> MCTS/ExIt/inference untouched
# --------------------------------------------------------------------------- #


def test_forward_shapes_and_finite() -> None:
    m = _autoreg_model()
    b = collate([_feats(4, 1, 1), _feats(3, 0, 2)])
    logits, value = m(b)
    assert logits.shape == (2, 4)  # padded to max L
    assert value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_evaluate_returns_valid_priors() -> None:
    # MCTS calls model.evaluate; priors must be a distribution over real options.
    m = _autoreg_model()
    b = collate([_feats(4, 1, 2)])
    priors, value = m.evaluate(b)
    assert priors.shape == (1, 4)
    assert torch.isfinite(value).all()
    assert torch.allclose(priors[0].sum(), torch.tensor(1.0), atol=1e-5)
    assert (priors >= 0).all()


def test_policy_step_shapes() -> None:
    m = _autoreg_model()
    b = collate([_feats(4, 1, 2)])
    state, ent = m.encode(b)
    picked = torch.zeros_like(b["option_mask"])
    opt_logits, stop_logit = m.policy_step(state, ent, b, picked)
    assert opt_logits.shape == (1, 4)
    assert stop_logit.shape == (1,)


def test_policy_step_rejected_on_marginal() -> None:
    import pytest

    m = build_model()  # marginal
    b = collate([_feats(4, 1, 2)])
    state, ent = m.encode(b)
    with pytest.raises(RuntimeError):
        m.policy_step(state, ent, b, torch.zeros_like(b["option_mask"]))


# --------------------------------------------------------------------------- #
# Sampling respects count bounds + STOP legality
# --------------------------------------------------------------------------- #


def test_sample_count_bounds_and_distinct() -> None:
    m = _autoreg_model()
    b = collate([_feats(5, 1, 3)])
    state, ent = m.encode(b)
    gen = torch.Generator().manual_seed(1)
    for _ in range(200):
        picks, _ = policy.sample_action_autoreg(m, state, ent, b, gen=gen)
        assert 1 <= len(picks) <= 3  # minCount..maxCount
        assert len(set(picks)) == len(picks)  # distinct
        assert all(0 <= p < 5 for p in picks)  # valid real options


def test_min_count_zero_allows_empty() -> None:
    # With minCount==0 the policy may legally pick nothing (STOP at step 0).
    m = _autoreg_model()
    b = collate([_feats(4, 0, 2)])
    state, ent = m.encode(b)
    gen = torch.Generator().manual_seed(3)
    lengths = {
        len(policy.sample_action_autoreg(m, state, ent, b, gen=gen)[0])
        for _ in range(300)
    }
    assert 0 in lengths  # empty selection is reachable


def test_min_count_two_never_stops_early() -> None:
    m = _autoreg_model()
    b = collate([_feats(5, 2, 4)])
    state, ent = m.encode(b)
    gen = torch.Generator().manual_seed(5)
    for _ in range(200):
        picks, _ = policy.sample_action_autoreg(m, state, ent, b, gen=gen)
        assert 2 <= len(picks) <= 4


# --------------------------------------------------------------------------- #
# logprob consistency: sample == recompute (the trainer's core invariant)
# --------------------------------------------------------------------------- #


def _recompute(m, state, ent, b, picks):
    k = max(len(picks), 0)
    actions = torch.zeros(1, max(k, 1), dtype=torch.long)
    if k:
        actions[0, :k] = torch.tensor(picks, dtype=torch.long)
    action_len = torch.tensor([k], dtype=torch.long)
    with torch.no_grad():
        return float(
            policy.batched_action_logprob_autoreg(m, state, ent, b, actions, action_len)[0]
        )


def test_sample_logprob_matches_recompute() -> None:
    m = _autoreg_model()
    for (n, lo, hi), seed in [((5, 1, 3), 1), ((4, 0, 2), 7), ((6, 2, 4), 9), ((3, 1, 1), 2)]:
        b = collate([_feats(n, lo, hi)])
        state, ent = m.encode(b)
        gen = torch.Generator().manual_seed(seed)
        for _ in range(25):
            picks, lp = policy.sample_action_autoreg(m, state, ent, b, gen=gen)
            recomputed = _recompute(m, state, ent, b, picks)
            assert abs(lp - recomputed) < 1e-4, (n, lo, hi, picks, lp, recomputed)


def test_batched_logprob_matches_per_row() -> None:
    m = _autoreg_model()
    feats = [_feats(5, 1, 3), _feats(4, 0, 2), _feats(3, 1, 1)]
    picks_each = [[2, 0], [], [1]]
    # per-row
    per_row = []
    for f, picks in zip(feats, picks_each):
        b1 = collate([f])
        s1, e1 = m.encode(b1)
        per_row.append(_recompute(m, s1, e1, b1, picks))
    # batched
    b = collate(feats)
    state, ent = m.encode(b)
    kmax = max(len(p) for p in picks_each)
    actions = torch.zeros(len(feats), max(kmax, 1), dtype=torch.long)
    for i, p in enumerate(picks_each):
        if p:
            actions[i, : len(p)] = torch.tensor(p, dtype=torch.long)
    action_len = torch.tensor([len(p) for p in picks_each], dtype=torch.long)
    with torch.no_grad():
        batched = policy.batched_action_logprob_autoreg(m, state, ent, b, actions, action_len)
    for i in range(len(feats)):
        assert abs(float(batched[i]) - per_row[i]) < 1e-4, (i, batched[i], per_row[i])


# --------------------------------------------------------------------------- #
# Gradients + degenerate rows
# --------------------------------------------------------------------------- #


def test_gradient_flows_into_autoreg_head() -> None:
    torch.manual_seed(0)
    m = build_model(ModelConfig(policy_head="autoreg")).train()
    b = collate([_feats(4, 1, 2)])
    state, ent = m.encode(b)
    actions = torch.tensor([[1, 3]], dtype=torch.long)
    action_len = torch.tensor([2], dtype=torch.long)
    lp = policy.batched_action_logprob_autoreg(m, state, ent, b, actions, action_len)
    lp.sum().backward()
    assert m.autoreg.stop_scorer[0].weight.grad is not None
    assert m.autoreg.opt_scorer[0].weight.grad is not None
    assert m.autoreg.pick_proj.weight.grad is not None


def test_entropy_finite_and_positive() -> None:
    m = _autoreg_model()
    b = collate([_feats(4, 0, 2), _feats(3, 1, 1)])
    state, ent = m.encode(b)
    h = policy.batched_entropy_autoreg(m, state, ent, b)
    assert h.shape == (2,)
    assert torch.isfinite(h).all() and (h >= 0).all()


def test_single_option_no_nan() -> None:
    m = _autoreg_model()
    b = collate([_feats(1, 1, 1)])
    state, ent = m.encode(b)
    gen = torch.Generator().manual_seed(0)
    picks, lp = policy.sample_action_autoreg(m, state, ent, b, gen=gen)
    assert picks == [0]
    assert np.isfinite(lp)
    assert abs(lp - _recompute(m, state, ent, b, picks)) < 1e-4


# --------------------------------------------------------------------------- #
# End-to-end against the real engine (rollout -> update -> ExIt -> inference)
# --------------------------------------------------------------------------- #


def _autoreg_cfg():
    return Config(model=ModelConfig(policy_head="autoreg"))


def test_ppo_rollout_update_smoke_autoreg() -> None:
    from pkm.new_agents.agent_000_dragapult.trainers.ppo import PpoTrainer, Step

    cfg = _autoreg_cfg()
    torch.manual_seed(0)
    m = build_model(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.train.lr)
    trainer = PpoTrainer()

    samples, stats = trainer.collect(m, n_games=2, cfg=cfg)
    assert stats["games"] == 2
    assert samples and all(isinstance(s, Step) for s in samples)
    # The learned count must actually vary (autoregressive STOP does something):
    # a real self-play game hits multi-select nodes, so not every action is len 1.
    assert any(len(s.action) != 1 for s in samples)

    upd = trainer.update(m, opt, samples, cfg)
    for key in ("pol_loss", "val_loss", "entropy", "explained_var"):
        assert key in upd and upd[key] == upd[key]  # finite (not NaN)


def test_exit_update_smoke_autoreg() -> None:
    # ExIt consumes only the [B,L] head, so it must work unchanged with autoreg.
    from pkm.new_agents.agent_000_dragapult.trainers.exit import ExItTrainer

    cfg = _autoreg_cfg()
    object.__setattr__(cfg.train, "method", "exit")
    object.__setattr__(cfg.train, "mcts_simulations", 4)
    torch.manual_seed(0)
    m = build_model(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.train.lr)
    samples, _ = ExItTrainer().collect(m, n_games=1, cfg=cfg)
    assert samples
    upd = ExItTrainer().update(m, opt, samples, cfg)
    assert upd["policy_loss"] == upd["policy_loss"]
    assert upd["value_loss"] == upd["value_loss"]


def test_inference_agent_plays_full_game_autoreg() -> None:
    from pkm.new_agents.agent_000_dragapult.agent import DragapultAgent
    from pkm.new_agents.agent_000_dragapult.cabt import (
        battle_finish,
        battle_select,
        battle_start,
    )

    cfg = _autoreg_cfg()
    torch.manual_seed(0)
    m = build_model(cfg)
    agent = DragapultAgent(model=m, seed=0, deck=cfg.run.deck)
    deck60 = deck.deck_60(cfg.run.deck)
    obs, _ = battle_start(deck60, deck60)
    n_iter = 0
    while obs["current"]["result"] < 0 and n_iter < 100000:
        picks = agent(obs)  # exercises the autoregressive inference sampler
        if obs.get("select") is not None:
            sel = obs["select"]
            assert len(set(picks)) == len(picks)  # distinct
            assert all(0 <= p < len(sel["option"]) for p in picks)  # valid
        obs = battle_select(picks)
        n_iter += 1
    battle_finish()
    assert obs["current"]["result"] in (0, 1, 2)
