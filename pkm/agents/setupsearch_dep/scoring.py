"""Leaf evaluator for `dragapult_setup_search_agent`.

Adapts the setup rubric in `pkm/rl/setup_turn_score.py` to the signature the
turn-scoped search expects: ``(final_obs, events, went_first) -> float``.

Why reuse that rubric rather than write another one: it already encodes what
this turn is for -- bench charged Drakloaks behind a disposable staller, a
Dragapult ex in hand to evolve into, Itchy Pollen if it's available. It was
built as a *training reward* for the RL setup agent; here it becomes a search
objective instead. That is the whole point of this agent:

    RL setup agent      -- learns to *approximate* the rubric from experience
    setup SEARCH agent  -- simulates real move sequences and *maximises* it

Approximating a proxy is worse than maximising it: the RL agent inherited both
the proxy's blind spots and its own approximation error, and could not tell you
which decision came from which. The search has only the proxy's blind spots,
and every choice it makes is directly attributable to a number you can read.
"""

from __future__ import annotations

from pkm.rl.setup_turn_score import score_end_of_turn
from pkm.types.obs import Observation


def evaluate(final_obs: dict, events: dict, went_first: bool) -> float:
    """Score the board this action sequence would leave at end of turn.

    `events` is the search's own record of what the sequence did; its
    ``itchy_pollen`` flag is exactly the input `score_end_of_turn` cannot
    recover from a static board (engine `Log` entries carry no payload naming
    the attack used).

    `final_obs` is always from our own point of view -- the search only scores
    states where it is still our turn -- so our hand is visible and the
    hand-dependent terms (Dragapult ex in hand, energy in hand, both readiness
    tiers) actually fire. Scoring an opponent-POV board would silently zero
    them, which is precisely the bug that made those terms dead during RL
    training.
    """
    if not final_obs:
        return 0.0
    state = final_obs.get("current")
    if not state:
        return 0.0
    obs = Observation.model_validate(final_obs)
    # Meowth ex's charge is waived only when the Supporter its ability fetches
    # was actually cashed in, and only for the two that justify two prizes:
    # Lillie's Determination always, Judge only if the line had not started
    # when it was played.
    meowth_excused = bool(
        events.get("lillies_played") or events.get("judge_no_dreepy")
    )
    return score_end_of_turn(
        obs,
        itchy_pollen_used=bool(events.get("itchy_pollen")),
        seat=state.get("yourIndex", 0),
        meowth_excused=meowth_excused,
        retreats=int(events.get("retreats", 0)),
    ).total
