"""Determinization: sample hidden zones (decks, hands, prizes) consistent with
what the observing player has seen, producing the predicted-card-ID lists that
``search_begin`` requires.
"""

import random
from collections import Counter

from pkm.data import get_card_data


def _pokemon_card_ids(p: dict) -> list[int]:
    ids = [p["id"]]
    for key in ("preEvolution", "energyCards", "tools"):
        ids.extend(c["id"] for c in (p.get(key) or []))
    return ids


def _visible_counter(state: dict, player_index: int, include_hand: bool) -> Counter:
    """Multiset of the player's cards that are face-up somewhere."""
    c: Counter = Counter()
    pl = state["players"][player_index]
    if include_hand:
        c.update(card["id"] for card in (pl.get("hand") or []))
    for p in pl.get("active") or []:
        if p:
            c.update(_pokemon_card_ids(p))
    for p in pl.get("bench") or []:
        c.update(_pokemon_card_ids(p))
    c.update(card["id"] for card in pl["discard"])
    c.update(card["id"] for card in pl["prize"] if card)
    for card in state.get("stadium") or []:
        if card["playerIndex"] == player_index:
            c[card["id"]] += 1
    for card in state.get("looking") or []:
        if card and card.get("playerIndex") == player_index:
            c[card["id"]] += 1
    return c


def _fit(
    pool: list[int], need: int, filler: list[int], rng: random.Random
) -> list[int]:
    """Pad/truncate the unseen-card pool to exactly `need` entries."""
    if len(pool) > need:
        return pool[:need]
    while len(pool) < need:
        pool.append(rng.choice(filler))
    return pool


def infer_opponent_decklist(obs: dict, energy_fallback: int = 3) -> list[int]:
    """Crude 60-card opponent decklist estimate from visible cards.

    Used at inference time when the true opponent list is unknown; training
    self-play should pass the exact list instead.
    """
    state = obs["current"]
    you = state["yourIndex"]
    visible = _visible_counter(state, 1 - you, include_hand=False)
    decklist = list(visible.elements())

    cards = get_card_data()
    basics = [cid for cid in visible if cards.get(cid) and cards[cid].basic]
    filler_basic = basics[0] if basics else energy_fallback
    # pad: a couple of extra basics (redraw targets), rest basic energy
    while len(decklist) < 8:
        decklist.append(filler_basic)
    while len(decklist) < 60:
        decklist.append(energy_fallback)
    return decklist[:60]


def sample_determinization(
    obs: dict,
    my_decklist: list[int],
    opp_decklist: list[int],
    rng: random.Random,
) -> dict[str, list[int]]:
    """Sample one assignment of all hidden zones.

    Returns kwargs for pkm.search.search_begin: your_deck, your_prize,
    opponent_deck, opponent_prize, opponent_hand, opponent_active.
    """
    state = obs["current"]
    select = obs["select"]
    you = state["yourIndex"]
    me = state["players"][you]
    opp = state["players"][1 - you]
    cards = get_card_data()

    # --- my hidden zones: deck order + facedown prizes ---
    seen = _visible_counter(state, you, include_hand=True)
    deck_revealed = select.get("deck") or []
    seen.update(card["id"] for card in deck_revealed if card)
    pool = list((Counter(my_decklist) - seen).elements())
    rng.shuffle(pool)

    deck_n = 0 if deck_revealed else me["deckCount"]
    my_facedown = [i for i, c in enumerate(me["prize"]) if c is None]
    pool = _fit(pool, deck_n + len(my_facedown), my_decklist, rng)
    your_deck = pool[:deck_n]
    your_prize = [c["id"] if c else 0 for c in me["prize"]]
    for slot, cid in zip(my_facedown, pool[deck_n:]):
        your_prize[slot] = cid

    # --- opponent hidden zones: active (if facedown), hand, deck, prizes ---
    seen_o = _visible_counter(state, 1 - you, include_hand=False)
    pool_o = list((Counter(opp_decklist) - seen_o).elements())
    rng.shuffle(pool_o)

    opponent_active: list[int] = []
    active = opp.get("active") or []
    if active and active[0] is None:
        # must be a Pokémon (it was placed during setup, so prefer a Basic)
        basics = [c for c in pool_o if cards.get(c) and cards[c].basic]
        pick = (
            basics[0]
            if basics
            else next(
                (c for c in pool_o if cards.get(c) and cards[c].card_type == 0), None
            )
        )
        if pick is None:
            raise ValueError("no Pokémon left in opponent decklist estimate")
        pool_o.remove(pick)
        opponent_active = [pick]

    hand_n = opp["handCount"]
    deck_n_o = opp["deckCount"]
    opp_facedown = [i for i, c in enumerate(opp["prize"]) if c is None]
    pool_o = _fit(pool_o, hand_n + deck_n_o + len(opp_facedown), opp_decklist, rng)

    opponent_hand = pool_o[:hand_n]
    opponent_deck = pool_o[hand_n : hand_n + deck_n_o]
    opponent_prize = [c["id"] if c else 0 for c in opp["prize"]]
    for slot, cid in zip(opp_facedown, pool_o[hand_n + deck_n_o :]):
        opponent_prize[slot] = cid

    # at setup the engine requires a Basic Pokémon in the opponent's deck
    if state["turn"] == 0 and opponent_deck:
        if not any(cards.get(c) and cards[c].basic for c in opponent_deck):
            swap = next(
                (
                    i
                    for i, c in enumerate(opponent_hand)
                    if cards.get(c) and cards[c].basic
                ),
                None,
            )
            if swap is not None:
                j = rng.randrange(len(opponent_deck))
                opponent_hand[swap], opponent_deck[j] = (
                    opponent_deck[j],
                    opponent_hand[swap],
                )

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
    }
