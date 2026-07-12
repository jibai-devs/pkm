"""Pokemon TCG AI Battle Challenge - Main entry point."""

from pathlib import Path

from kaggle_environments import make

from pkm.agents import make_random_agent
from pkm.data import Deck, get_card_data, get_attack_data


def _resolve_deck(path: str = "deck.csv") -> Deck:
    """Load deck.csv from the submission root (Kaggle) or a local path."""
    p = Path(path)
    if p.exists():
        return Deck.from_csv(p)
    # Fallback: try agent-named deck under deck/
    for candidate in sorted(Path("deck").glob("*.csv")):
        return Deck.from_csv(candidate)
    raise FileNotFoundError(f"No deck found at {path} or deck/*.csv")


def run_battle(deck_path: str = "deck.csv", render: bool = True):
    """Run a battle between two random agents."""
    deck = _resolve_deck(deck_path)
    agent = make_random_agent(deck.card_ids)

    env = make("cabt", configuration={"decks": [deck.card_ids, deck.card_ids]})
    env.run([agent, agent])

    if render:
        with open("result.html", "w") as f:
            f.write(env.render(mode="html"))
        print("Battle result saved to result.html")

    return env


def main():
    print("Pokemon TCG AI Battle Challenge")
    print("=" * 40)

    cards = get_card_data()
    attacks = get_attack_data()
    print(f"Available cards: {len(cards)}")
    print(f"Available attacks: {len(attacks)}")

    deck = _resolve_deck("deck.csv")
    print(f"Deck loaded: {deck}")

    env = run_battle()

    final = env.steps[-1]
    print(f"Player 0: {final[0].status}, reward: {final[0].reward}")
    print(f"Player 1: {final[1].status}, reward: {final[1].reward}")
    print("Battle complete!")


if __name__ == "__main__":
    main()
