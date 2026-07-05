from pathlib import Path

from .card_data import get_card_by_id, CardData


class Deck:
    """Represents a 60-card Pokemon TCG deck."""

    def __init__(self, card_ids: list[int]):
        if len(card_ids) != 60:
            raise ValueError(f"Deck must have 60 cards, got {len(card_ids)}")
        self.card_ids = card_ids

    @classmethod
    def from_csv(cls, path: str | Path) -> "Deck":
        """Load a deck from a CSV file (one card ID per line)."""
        card_ids = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    card_ids.append(int(line))
        return cls(card_ids)

    def to_csv(self, path: str | Path) -> None:
        """Save deck to a CSV file."""
        with open(path, "w") as f:
            for card_id in self.card_ids:
                f.write(f"{card_id}\n")

    def get_cards(self) -> list[CardData]:
        """Get CardData for all cards in the deck."""
        cards = []
        for cid in self.card_ids:
            card = get_card_by_id(cid)
            if card is not None:
                cards.append(card)
        return cards

    def get_pokemon_ids(self) -> list[int]:
        """Get unique Pokemon card IDs in the deck."""
        result = []
        for cid in self.card_ids:
            card = get_card_by_id(cid)
            if card is not None and card.card_type == 0:
                result.append(cid)
        return list(set(result))

    def get_energy_ids(self) -> list[int]:
        """Get unique energy card IDs in the deck."""
        result = []
        for cid in self.card_ids:
            card = get_card_by_id(cid)
            if card is not None and card.card_type in (5, 6):
                result.append(cid)
        return list(set(result))

    def count(self, card_id: int) -> int:
        """Count how many of a specific card are in the deck."""
        return self.card_ids.count(card_id)

    def __len__(self) -> int:
        return len(self.card_ids)

    def __repr__(self) -> str:
        return f"Deck({len(self.card_ids)} cards)"
