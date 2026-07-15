"""Kaggle submission entry point for the 02_dragapult agent.

Kaggle only ever sees ``agent`` below: it dispatches each turn's decisions
to one of several sub-agents internally (see
``pkm/agents/singaporean_middleman.py``).
"""

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
    """Load the bundled deck, with the repository's Dragapult deck as fallback."""
    module_dir = _module_dir()
    candidates = (
        module_dir / "deck.csv",
        module_dir / "deck" / "02_dragapult.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return Deck.from_csv(candidate)
    raise FileNotFoundError("No 02_dragapult deck found")


DECK = _resolve_deck().card_ids
agent = make_singaporean_middleman(DECK)
