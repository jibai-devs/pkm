"""GameContext: one instance per match, threaded through every place a
decision gets encoded so heuristics/features/hard-rules can read per-game
memory (`tracker`) alongside the raw observation.

Never construct or reuse a GameContext (or its tracker) across game
boundaries -- self-play runs many games back-to-back, and a leaked
reference would silently contaminate one game's prize knowledge into an
unrelated game.
"""

from dataclasses import dataclass

from pkm.heuristics.deck_tracker import DeckTracker


@dataclass
class GameContext:
    my_deck: list[int]
    tracker: DeckTracker
    opp_decklist: list[int] | None = None
