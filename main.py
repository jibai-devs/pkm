"""Pokemon TCG AI Battle Challenge - Main entry point."""

from kaggle_environments import make

from pkm.agents import make_random_agent
from pkm.data import Deck, get_card_data, get_attack_data


def run_battle(deck_path: str = "deck.csv", render: bool = True):
    """Run a battle between two random agents."""
    deck = Deck.from_csv(deck_path)
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

    deck = Deck.from_csv("deck.csv")
    print(f"Deck loaded: {deck}")

    env = run_battle()

    final = env.steps[-1]
    print(f"Player 0: {final[0].status}, reward: {final[0].reward}")
    print(f"Player 1: {final[1].status}, reward: {final[1].reward}")
    print("Battle complete!")


if __name__ == "__main__":
    main()
