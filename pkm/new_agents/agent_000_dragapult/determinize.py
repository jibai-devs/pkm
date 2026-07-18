"""Single-sample (K=1) determinization of hidden zones for `cabt.search_begin`.

Both seats play a copy of the known ``deck.DECK_60`` (see `train.py`'s
`battle_start(DECK_60, DECK_60)`): the engine concatenates the two decks
(`pkm/engine/api.py:battle_start` -- ``cards = deck0 + deck1``) and each
player's 60 cards are tracked independently (via `playerIndex` on every
`CardRef`/`PokemonRef`). So hidden information is only *which* of the
already-known 60 cards sit in a player's own deck/hand/prize -- not an
open opponent vocabulary. There are therefore **two separate 60-card
multisets** (yours and the opponent's), each reduced only by that same
seat's own publicly-visible cards; they are never pooled together.

From the acting seat's view:
  - ``your_deck`` / ``your_prize``: your own copy of DECK_60 minus everything
    you can see of your own cards (hand, board, discard). Your own prizes are
    face-down even to you, so they are NOT subtracted -- they stay in the
    unknown pool and get dealt back out into `your_prize`'s face-down slots.
  - ``opponent_deck`` / ``opponent_hand`` / ``opponent_prize`` /
    ``opponent_active``: the opponent's own copy of DECK_60 minus the
    opponent's publicly-visible cards (their face-up board + discard; their
    hand and face-down prizes are hidden from us), dealt back out to match
    the observed `handCount` / `deckCount` / face-down prize slots.

This is honest IS-MCTS with K=1: one sampled world per search. Full IS-MCTS
(sample K worlds, search each, average visits) drops in later behind the same
`DETERMINIZERS` seam without touching `mcts.py`. See
`docs/specs/2026-07-18-pluggable-trainers-mcts-exit-design.md` Sec 5.3.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Callable

import torch

from pkm.new_agents.agent_000_dragapult import deck
from pkm.new_agents.agent_000_dragapult.cabt import all_card_data


@dataclass(frozen=True)
class Predictions:
    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]


def _pokemon_card_ids(p: dict) -> list[int]:
    """A Pokémon in play reveals its own id plus any attached/nested cards."""
    ids = [p["id"]]
    for zone in ("preEvolution", "energyCards", "tools"):
        ids.extend(c["id"] for c in (p.get(zone) or []) if c)
    return ids


def _visible_card_ids(player: dict) -> list[int]:
    """Card IDs this player's own zones reveal face-up, from the raw obs dict.

    Works unchanged for either seat: hidden zones are already `None` (or
    omitted) at the JSON level (e.g. opponent `hand` is the bare value
    `None`, opponent `prize` slots are `None`), so this only ever picks up
    what is genuinely public.
    """
    ids: list[int] = []
    for zone in ("active", "bench"):
        for slot in player.get(zone) or []:
            if slot:
                ids.extend(_pokemon_card_ids(slot))
    for c in player.get("discard") or []:
        if c:
            ids.append(c["id"])
    for c in player.get("prize") or []:
        if c:
            ids.append(c["id"])
    for c in player.get("hand") or []:
        if c:
            ids.append(c["id"])
    return ids


def _shuffled_unknown_pool(player: dict, gen: torch.Generator) -> list[int]:
    """This player's own DECK_60 copy minus their own publicly-visible cards."""
    known = collections.Counter(deck.DECK_60)
    for cid in _visible_card_ids(player):
        if known[cid] > 0:
            known[cid] -= 1
    pool = [c for c, n in known.items() for _ in range(n)]
    order = torch.randperm(len(pool), generator=gen).tolist()
    return [pool[i] for i in order]


def _deal_prize(prize_field: list, pool: list[int]) -> tuple[list[int], list[int]]:
    """Fill face-down (`None`) prize slots from `pool`; return (prize, remaining pool)."""
    prize = [c["id"] if c else 0 for c in prize_field]
    facedown_slots = [i for i, c in enumerate(prize_field) if c is None]
    fill, rest = pool[: len(facedown_slots)], pool[len(facedown_slots) :]
    for slot, cid in zip(facedown_slots, fill):
        prize[slot] = cid
    return prize, rest


def sample_world(obs: dict, seat: int, gen: torch.Generator) -> Predictions:
    state = obs["current"]
    me = state["players"][seat]
    opp = state["players"][1 - seat]

    # --- my own hidden zones ---
    my_pool = _shuffled_unknown_pool(me, gen)
    deck_n = me["deckCount"]
    your_deck, my_pool = my_pool[:deck_n], my_pool[deck_n:]
    your_prize, _my_pool = _deal_prize(me.get("prize") or [], my_pool)

    # --- opponent's hidden zones ---
    opp_pool = _shuffled_unknown_pool(opp, gen)

    opponent_active: list[int] = []
    active = opp.get("active") or []
    if active and active[0] is None:
        # Only happens momentarily during setup (both seats' active are placed
        # face-down before being revealed together); the placed Pokémon must
        # be a Basic. Pick one out of the opponent's own remaining pool.
        cards = all_card_data()
        pick_idx = next(
            (i for i, c in enumerate(opp_pool) if cards.get(c) and cards[c].basic),
            None,
        )
        if pick_idx is None:
            raise ValueError("no Basic Pokémon left in opponent's unknown pool")
        opponent_active = [opp_pool.pop(pick_idx)]

    n_hand = opp["handCount"]
    opponent_hand, opp_pool = opp_pool[:n_hand], opp_pool[n_hand:]
    opponent_prize, opponent_deck = _deal_prize(opp.get("prize") or [], opp_pool)

    return Predictions(
        your_deck=your_deck,
        your_prize=your_prize,
        opponent_deck=opponent_deck,
        opponent_prize=opponent_prize,
        opponent_hand=opponent_hand,
        opponent_active=opponent_active,
    )


DETERMINIZERS: dict[str, Callable] = {"sample": sample_world}
