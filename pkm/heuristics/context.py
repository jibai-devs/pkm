"""GameContext: one instance per match, threaded through every place a
decision gets encoded so heuristics/features/hard-rules can read per-game
memory (`tracker`) alongside the raw observation.

Never construct or reuse a GameContext (or its tracker) across game
boundaries -- self-play runs many games back-to-back, and a leaked
reference would silently contaminate one game's prize knowledge into an
unrelated game.
"""

from dataclasses import dataclass

import numpy as np

from pkm.heuristics.deck_tracker import DeckTracker


@dataclass
class GameContext:
    my_deck: list[int]
    tracker: DeckTracker
    opp_decklist: list[int] | None = None
    # Task 8: opponent-archetype belief, updated by the acting policy after
    # each real decision (pkm/rl/rollout.py:TorchPolicy.act) from the
    # trunk's own archetype head. None until the first update -- read as
    # zero/uninformative by pkm/rl/features.py's GLOBAL feature until then.
    archetype_belief: np.ndarray | None = None
