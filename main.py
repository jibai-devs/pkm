"""Kaggle submission entry point for the 03_pult_munki agent.

Routes through singaporean_middleman so the first-turn MCTS agent handles our
opening turn and the dragapult_default agent handles the rest -- and so each decision
prints which sub-agent made it (see pkm/agents/singaporean_middleman.py).

Turn planning is switched on below. It is **purely diagnostic**: at the first
decision of each turn it simulates the whole turn in a separate engine process
and prints the intended move list, so the episode log shows what the bot meant
to do. It never influences a decision. Set PKM_TURN_PLAN=0 to disable.
"""

import os
import tempfile
from pathlib import Path

# Must be set before the agent is built: the middleman boots the planner's
# worker process at construction time. A temp dir because the submission
# directory itself is not reliably writable. Every planner failure path
# (no subprocess, no disk, timeout) degrades to "no planning", never to a
# broken match -- see pkm/agents/turn_planner/client.py.
if os.environ.get("PKM_TURN_PLAN", "1") != "0":
    os.environ.setdefault(
        "PKM_TURN_PLAN_DIR", str(Path(tempfile.gettempdir()) / "pkm_turn_plans")
    )

from pkm.agents import make_singaporean_middleman  # noqa: E402
from pkm.data import Deck  # noqa: E402

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
