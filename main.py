"""Pokemon TCG AI Battle Challenge - Main entry point."""

from pathlib import Path
from typing import Callable

from kaggle_environments import make

from pkm.agents import make_neural_agent
from pkm.data import Deck, get_card_data, get_attack_data


_KAGGLE_AGENT_DIR = Path("/kaggle_simulations/agent")
_submission_agent: Callable[[dict], list[int]] | None = None


def _resolve_deck(path: str = "deck.csv") -> Deck:
    """Load a deck from a local path or relative to the submission module."""
    p = Path(path)
    try:
        module_dir = Path(__file__).resolve().parent
    except NameError:
        module_dir = _KAGGLE_AGENT_DIR
    candidates = [p]
    if not p.is_absolute():
        candidates.append(module_dir / p)

    for candidate in candidates:
        if candidate.is_file():
            return Deck.from_csv(candidate)

    for deck_dir in (Path("deck"), module_dir / "deck"):
        for candidate in sorted(deck_dir.glob("*.csv")):
            return Deck.from_csv(candidate)
    raise FileNotFoundError(f"No deck found at {path} or deck/*.csv")


def run_battle(deck_path: str = "deck.csv", render: bool = True):
    """Run a local battle between two copies of the submission agent."""
    deck = _resolve_deck(deck_path)
    agent = make_neural_agent(deck.card_ids)

    env = make("cabt", configuration={"decks": [deck.card_ids, deck.card_ids]})
    env.run([agent, agent])

    if render:
        with open("result.html", "w") as f:
            f.write(env.render(mode="html"))
        print("Battle result saved to result.html")

    return env


def main(obs: dict) -> list[int]:
    """Return the neural agent's action for Kaggle's agent protocol."""
    global _submission_agent
    if _submission_agent is None:
        _submission_agent = make_neural_agent(_resolve_deck("deck.csv").card_ids)
    return _submission_agent(obs)


def run_local_battle() -> None:
    """Run and report a local battle for development."""
    print("Pokemon TCG AI Battle Challenge")
    print("=" * 40)

    cards = get_card_data()
    attacks = get_attack_data()
    print(f"Available cards: {len(cards)}")
    print(f"Available attacks: {len(attacks)}")

    deck = _resolve_deck("deck.csv")
    print(f"Deck loaded: {deck}")

    env = run_battle("deck/02_dragapult.csv")

    final = env.steps[-1]
    print(f"Player 0: {final[0].status}, reward: {final[0].reward}")
    print(f"Player 1: {final[1].status}, reward: {final[1].reward}")
    print("Battle complete!")


if __name__ == "__main__":
    run_local_battle()
