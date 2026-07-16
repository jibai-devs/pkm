"""Tracks where each of our own 60 deck cards currently is, across turns.

Cards are identified by the engine's per-instance ``serial`` once observed in
a public zone (hand, board, discard, ...); until then they're assumed to
still be in the deck. Slots are keyed by position in the original 60-card
list, so duplicate card IDs (e.g. 4x Basic Energy) get distinct entries.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum

from pkm.types.obs import Observation


class CardLocation(StrEnum):
    DECK = "deck"
    HAND = "hand"
    ACTIVE = "active"
    BENCH = "bench"
    DISCARD = "discard"
    PRIZE = "prize"
    STADIUM = "stadium"
    ATTACHED_ENERGY = "attached_energy"
    ATTACHED_TOOL = "attached_tool"


@dataclass
class CardState:
    card_id: int
    serial: int | None = None
    location: CardLocation = CardLocation.DECK


class DeckTracker:
    """One `CardState` per slot in our original 60-card deck list."""

    def __init__(self, deck: list[int]) -> None:
        self.cards: dict[int, CardState] = {
            i: CardState(card_id=cid) for i, cid in enumerate(deck)
        }
        self.prizes_known = False
        self._serial_to_slot: dict[int, int] = {}
        self._unbound_by_id: dict[int, deque[int]] = defaultdict(deque)
        for slot, card_id in enumerate(deck):
            self._unbound_by_id[card_id].append(slot)

    def _bind(self, card_id: int, serial: int, location: CardLocation) -> None:
        slot = self._serial_to_slot.get(serial)
        if slot is None:
            pending = self._unbound_by_id.get(card_id)
            if not pending:
                return  # not one of ours; ignore
            slot = pending.popleft()
            self._serial_to_slot[serial] = slot
            self.cards[slot].serial = serial
        self.cards[slot].location = location

    def observe(self, obs: dict) -> None:
        """Update card locations from the current snapshot in `obs`."""
        parsed = Observation.model_validate(obs)
        if parsed.current is None:
            return
        me = parsed.current.players[parsed.current.yourIndex]

        for ref in me.hand or []:
            self._bind(ref.id, ref.serial, CardLocation.HAND)
        for ref in me.discard:
            if ref is not None:
                self._bind(ref.id, ref.serial, CardLocation.DISCARD)
        for ref in me.prize:
            if ref is not None:
                self._bind(ref.id, ref.serial, CardLocation.PRIZE)

        board = [(p, CardLocation.ACTIVE) for p in me.active]
        board += [(p, CardLocation.BENCH) for p in me.bench]
        for pokemon, location in board:
            if pokemon is None:
                continue
            self._bind(pokemon.id, pokemon.serial, location)
            for pre in pokemon.preEvolution:
                self._bind(pre.id, pre.serial, location)
            for energy in pokemon.energyCards:
                self._bind(energy.id, energy.serial, CardLocation.ATTACHED_ENERGY)
            for tool in pokemon.tools:
                self._bind(tool.id, tool.serial, CardLocation.ATTACHED_TOOL)

        for ref in parsed.current.stadium:
            if ref is not None and ref.playerIndex == parsed.current.yourIndex:
                self._bind(ref.id, ref.serial, CardLocation.STADIUM)

    def is_search_reveal(self, obs: dict) -> bool:
        """True if `obs` is a search effect showing our *entire* deck.

        `select.deck` is populated whenever any deck-selection effect is
        resolving; it only tells us the whole deck (as opposed to e.g. "look
        at the top N cards") when its length matches our current deck size.
        """
        parsed = Observation.model_validate(obs)
        if parsed.current is None or parsed.select is None:
            return False
        if parsed.select.deck is None:
            return False
        me = parsed.current.players[parsed.current.yourIndex]
        shown = [ref for ref in parsed.select.deck if ref is not None]
        return len(shown) == me.deckCount

    def record_search_reveal(self, obs: dict) -> None:
        """Bind every card a full deck search just showed us, then deduce
        the prize pile: any slot still unbound (never seen in
        hand/discard/board/deck) has nowhere left to be but the prizes —
        the one zone we can't observe directly. Call only after
        `is_search_reveal(obs)` returns True.
        """
        parsed = Observation.model_validate(obs)
        assert parsed.select is not None and parsed.select.deck is not None
        for ref in parsed.select.deck:
            if ref is not None:
                self._bind(ref.id, ref.serial, CardLocation.DECK)
        for card in self.cards.values():
            if card.serial is None:
                card.location = CardLocation.PRIZE
        self.prizes_known = True

    def by_location(self, location: CardLocation) -> list[CardState]:
        return [c for c in self.cards.values() if c.location == location]

    def known_prizes(self) -> list[int]:
        """Card IDs deduced to be in the prize pile (order not meaningful)."""
        return [c.card_id for c in self.by_location(CardLocation.PRIZE)]
