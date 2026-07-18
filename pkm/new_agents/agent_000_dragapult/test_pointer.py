"""Tests for the grounded pointer: option -> board-entity resolution + gather.

Covers two things:
  * ``_option_entity_slot`` / ``_slot_of`` map an option to the correct board
    slot (0-11) or -1 (no board target), matching :func:`featurize`'s layout.
  * :class:`PolicyValueModel` actually *consumes* the referenced entity — the
    encoder's per-entity embedding reaches the scorer, so changing which entity
    an option points at changes that option's logit.
"""

from __future__ import annotations

import numpy as np
import torch

from pkm.new_agents.agent_000_dragapult.cabt import AreaType, Option, OptionType
from pkm.new_agents.agent_000_dragapult.config import build_model
from pkm.new_agents.agent_000_dragapult.features import (
    F,
    G,
    O,
    Features,
    _option_entity_slot,
    _slot_of,
)
from pkm.new_agents.agent_000_dragapult.model import collate
from pkm.new_agents.agent_000_dragapult import deck

VOCAB = deck.VOCAB_SIZE


# --------------------------------------------------------------------------- #
# Slot resolution
# --------------------------------------------------------------------------- #


def test_slot_of_layout() -> None:
    you = 0
    assert _slot_of(you, 0, AreaType.ACTIVE, 0) == 0  # own active
    assert _slot_of(you, 0, AreaType.BENCH, 0) == 1  # own bench[0]
    assert _slot_of(you, 0, AreaType.BENCH, 4) == 5  # own bench[4]
    assert _slot_of(you, 1, AreaType.ACTIVE, 0) == 6  # opp active
    assert _slot_of(you, 1, AreaType.BENCH, 3) == 10  # opp bench[3]
    assert _slot_of(you, None, AreaType.ACTIVE, 0) == 0  # None player -> us


def test_slot_of_non_targets() -> None:
    you = 0
    assert _slot_of(you, 0, None, 0) == -1
    assert _slot_of(you, 0, AreaType.ACTIVE, None) == -1
    assert _slot_of(you, 0, AreaType.HAND, 2) == -1  # not a board area
    assert _slot_of(you, 0, AreaType.BENCH, 5) == -1  # out of range


def test_option_entity_slot_by_type() -> None:
    you = 0
    # ATTACK / RETREAT always act on our own active (slot 0).
    assert _option_entity_slot(you, Option(type=OptionType.ATTACK, attackId=1)) == 0
    assert _option_entity_slot(you, Option(type=OptionType.RETREAT)) == 0
    # ATTACH/EVOLVE resolve via inPlayArea/inPlayIndex (own Pokémon).
    evo = Option(type=OptionType.EVOLVE, inPlayArea=AreaType.BENCH, inPlayIndex=2)
    assert _option_entity_slot(you, evo) == 3
    # A direct board target via area/index + playerIndex (opp bench[0] = slot 7).
    tgt = Option(type=OptionType.ENERGY, playerIndex=1, area=AreaType.BENCH, index=0)
    assert _option_entity_slot(you, tgt) == 7
    # A non-board option (YES/NO) targets nothing.
    assert _option_entity_slot(you, Option(type=OptionType.YES)) == -1


# --------------------------------------------------------------------------- #
# Model consumes the referenced entity
# --------------------------------------------------------------------------- #


def _features(
    option_slot: int = 0,
    option_card_id: int = 0,
    option_attack_id: int = 0,
) -> Features:
    """Minimal Features with slots 0 and 1 occupied and one option."""
    ent_feat = np.zeros((12, F), dtype=np.float32)
    mask = np.zeros(12, dtype=np.float32)
    id_row = np.zeros(12, dtype=np.int64)
    card_id = np.zeros(12, dtype=np.int64)
    # two distinct occupied entities so their embeddings differ
    for s in (0, 1):
        mask[s] = 1.0
        ent_feat[s, 0] = 0.5 + 0.3 * s
        ent_feat[s, 7] = 1.0  # is_own
        id_row[s] = 1 + s
        card_id[s] = 119 + s
    return Features(
        entity_id_row=id_row,
        entity_card_id=card_id,
        entity_feat=ent_feat,
        entity_mask=mask,
        hand_hist=np.zeros(VOCAB, dtype=np.float32),
        discard_hist=np.zeros(VOCAB, dtype=np.float32),
        globals=np.zeros(G, dtype=np.float32),
        option_type=np.array([int(OptionType.ATTACK)], dtype=np.int64),
        option_feat=np.zeros((1, O), dtype=np.float32),
        option_entity_slot=np.array([option_slot], dtype=np.int64),
        option_card_id=np.array([option_card_id], dtype=np.int64),
        option_card_row=np.array([deck.row_of(option_card_id)], dtype=np.int64),
        option_attack_id=np.array([option_attack_id], dtype=np.int64),
        select_type=0,
        select_context=0,
        min_count=1,
        max_count=1,
    )


def test_forward_shapes_and_finite() -> None:
    torch.manual_seed(0)
    model = build_model().eval()
    b = collate([_features(0), _features(-1)])
    logits, value = model(b)
    assert logits.shape == (2, 1)
    assert value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_referenced_entity_changes_logit() -> None:
    torch.manual_seed(0)
    model = build_model().eval()
    with torch.no_grad():
        l0 = model(collate([_features(0)]))[0][0, 0]  # option -> entity in slot 0
        l1 = model(collate([_features(1)]))[0][0, 0]  # same option -> slot 1
    # The only thing that differs is which entity the option points at, so if the
    # gathered entity reached the scorer, the logits must differ.
    assert not torch.isclose(l0, l1, atol=1e-5), (l0.item(), l1.item())


def test_gradient_flows_to_pointer() -> None:
    torch.manual_seed(0)
    model = build_model().train()
    logits, _ = model(collate([_features(0)]))
    logits.sum().backward()
    assert model.null_entity.grad is not None
    # the attention/entity path received gradient
    assert model.encoder.cls.grad is not None


def test_option_card_identity_changes_logit() -> None:
    torch.manual_seed(0)
    model = build_model().eval()
    with torch.no_grad():
        a = model(collate([_features(option_card_id=2)]))[0][0, 0]  # R-Energy
        b = model(collate([_features(option_card_id=121)]))[0][0, 0]  # Dragapult ex
    assert not torch.isclose(a, b, atol=1e-5), (a.item(), b.item())


def test_option_attack_identity_changes_logit() -> None:
    torch.manual_seed(0)
    model = build_model().eval()
    with torch.no_grad():
        a = model(collate([_features(option_attack_id=1)]))[0][0, 0]
        b = model(collate([_features(option_attack_id=50)]))[0][0, 0]
    assert not torch.isclose(a, b, atol=1e-5), (a.item(), b.item())


def test_option_card_encoder_is_shared_with_board() -> None:
    # the option encoder must reuse the trunk's card encoder (same object), so a
    # card has one identity whether on the board or being played.
    model = build_model()
    assert model.option_enc.card is model.encoder.card


def test_featurize_resolves_real_option_card_ids() -> None:
    # End-to-end against a captured live-engine observation: options that name a
    # card must resolve to a real (non-zero) card id.
    import json
    from pathlib import Path

    from pkm.new_agents.agent_000_dragapult.cabt import to_observation
    from pkm.new_agents.agent_000_dragapult.features import _option_card_id, featurize

    repo = Path(__file__).resolve().parents[3]
    data = json.loads((repo / "tests/fixtures/observations.json").read_text())

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("current"), dict) and isinstance(o.get("select"), dict):
                if o["select"].get("option"):
                    yield o
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)

    resolved_any = False
    for raw in walk(data):
        obs = to_observation(raw)
        feats = featurize(obs)
        assert feats.option_card_id.shape == feats.option_type.shape
        # a CARD/PLAY/ATTACH/EVOLVE option in this decision should resolve a card
        st, sel = obs.current, obs.select
        assert st is not None and sel is not None
        for o in sel.option:
            if int(o.type) in (3, 7, 8, 9):
                if _option_card_id(st, sel, st.yourIndex, o) > 0:
                    resolved_any = True
    assert resolved_any, "no card-bearing option resolved to a real id"
