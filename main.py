"""Kaggle submission entry point for the 03_pult_munki agent.

Routes through singaporean_middleman so the first-turn MCTS agent handles our
opening turn and the neural policy handles the rest -- and so each decision
prints which sub-agent made it (see pkm/agents/singaporean_middleman.py)."""

from pathlib import Path

from pkm.agents import make_singaporean_middleman
from pkm.data import Deck


_KAGGLE_AGENT_DIR = Path("/kaggle_simulations/agent")


def _module_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return _KAGGLE_AGENT_DIR


def _resolve_deck() -> Deck:
    """Load the bundled deck, with the repository's pult_munki deck as fallback."""
    module_dir = _module_dir()
    candidates = (
        module_dir / "deck.csv",
        module_dir / "deck" / "03_pult_munki.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return Deck.from_csv(candidate)
    raise FileNotFoundError("No 03_pult_munki deck found")


DECK = _resolve_deck().card_ids
agent = make_singaporean_middleman(DECK)
