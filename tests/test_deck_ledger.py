import numpy as np
import torch

from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import CardLocation, DeckTracker
from pkm.rl.features import deck_ledger
from pkm.rl.model import PolicyValueNet
from pkm.rl.numpy_policy import NumpyPolicy


def _ctx(deck: list[int]) -> GameContext:
    return GameContext(my_deck=deck, tracker=DeckTracker(deck))


def _slots_by_id(tracker: DeckTracker) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for slot, cs in tracker.cards.items():
        out.setdefault(cs.card_id, []).append(slot)
    return out


# --- deck_ledger tally ---------------------------------------------------


def test_deck_ledger_nothing_seen_returns_full_decklist_counts():
    deck = [1, 1, 1, 2, 2, 3]
    ctx = _ctx(deck)
    ids, counts = deck_ledger(ctx)
    tally = dict(zip(ids.tolist(), counts.tolist()))
    assert tally == {1: 3.0, 2: 2.0, 3: 1.0}


def test_deck_ledger_everything_seen_returns_empty():
    deck = [1, 1, 2]
    ctx = _ctx(deck)
    for card in ctx.tracker.cards.values():
        card.location = CardLocation.HAND
    ids, counts = deck_ledger(ctx)
    assert len(ids) == 0
    assert len(counts) == 0


def test_deck_ledger_partial_reflects_only_unseen():
    deck = [1, 1, 1, 2, 2, 3]
    ctx = _ctx(deck)
    slots = _slots_by_id(ctx.tracker)
    ctx.tracker.cards[slots[1][0]].location = CardLocation.HAND
    ctx.tracker.cards[slots[2][0]].location = CardLocation.HAND
    ctx.tracker.cards[slots[2][1]].location = CardLocation.DISCARD

    ids, counts = deck_ledger(ctx)
    tally = dict(zip(ids.tolist(), counts.tolist()))
    assert tally == {1: 2.0, 3: 1.0}
    assert 2 not in tally


def test_deck_ledger_none_context_returns_empty():
    ids, counts = deck_ledger(None)
    assert len(ids) == 0
    assert len(counts) == 0


# --- model/numpy pooling math ---------------------------------------------


def _fixed_net(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    model = PolicyValueNet()
    model.eval()
    return model


def test_model_deck_pool_matches_hand_computed_weighted_sum():
    model = _fixed_net()
    ids = torch.tensor([[5, 12, 0]], dtype=torch.int64)  # 0 = padding
    counts = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32)

    pooled = model._pool_deck(ids, counts)[0]

    emb = model.card_emb.weight.detach()
    expected = 3.0 * emb[5] + 1.0 * emb[12] + 0.0 * emb[0]
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_model_deck_pool_empty_is_zero():
    model = _fixed_net()
    ids = torch.zeros((1, 0), dtype=torch.int64)
    counts = torch.zeros((1, 0), dtype=torch.float32)

    pooled = model._pool_deck(ids, counts)[0]
    assert torch.allclose(pooled, torch.zeros_like(pooled))


def test_model_board_pool_matches_hand_computed_mean():
    model = _fixed_net(seed=1)
    ids = torch.tensor([[7, 9, 0]], dtype=torch.int64)  # 0 = empty slot

    pooled = model._pool_cards(ids)[0]

    emb = model.card_emb.weight.detach()
    expected = (emb[7] + emb[9]) / 2.0  # mean over the 2 present slots only
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_model_board_pool_all_empty_is_zero():
    model = _fixed_net(seed=1)
    ids = torch.zeros((1, 3), dtype=torch.int64)

    pooled = model._pool_cards(ids)[0]
    assert torch.allclose(pooled, torch.zeros_like(pooled))


def test_numpy_pool_deck_matches_torch():
    model = _fixed_net(seed=3)
    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np_pol = NumpyPolicy(weights)

    ids = np.array([7, 20], dtype=np.int64)
    counts = np.array([2.0, 4.0], dtype=np.float32)
    pooled_np = np_pol._pool_deck(ids, counts)

    with torch.no_grad():
        pooled_t = model._pool_deck(
            torch.tensor(ids[None, :]), torch.tensor(counts[None, :])
        )[0].numpy()
    assert np.allclose(pooled_np, pooled_t, atol=1e-5)


def test_numpy_pool_board_matches_torch():
    model = _fixed_net(seed=4)
    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np_pol = NumpyPolicy(weights)

    ids = np.array([3, 0, 8], dtype=np.int64)
    pooled_np = np_pol._pool_cards(ids)

    with torch.no_grad():
        pooled_t = model._pool_cards(torch.tensor(ids[None, :]))[0].numpy()
    assert np.allclose(pooled_np, pooled_t, atol=1e-5)
