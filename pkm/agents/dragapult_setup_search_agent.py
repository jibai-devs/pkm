"""Search-based setup agent: no weights, no training.

Where `dragapult_setup_agent` is a trained network answering one decision at a
time, this plays the same turn by *searching*: it simulates whole action
sequences to the end of the turn on the engine's own forward model and keeps
the one that leaves the best-scoring board, judged by the rubric in
`pkm/rl/setup_turn_score.py`.

**Why this shape.** The RL setup agent measured 9 points of win rate *worse*
than not using it (400 games, z=3.28). The cause was structural: it was trained
on a 2-turn episode whose terminal reward was that board rubric, so it never
saw a game it had to win -- and a one-shot network can only approximate the
rubric anyway, without knowing what it will do later in the same turn. This
agent removes the approximation layer entirely. It optimises the same rubric,
but directly and with real lookahead, so:

  * every choice is attributable to a number you can read and tune,
  * a bad move means the rubric is wrong, not that training under-fit it,
  * there is nothing to retrain when the rubric changes.

It shares the turn-scoped MCTS with the first-turn agent
(`turn1agent_dep/search.py`), which takes its leaf evaluator as a parameter --
same machinery, different objective.

**The honest caveat**: maximising a proxy is still maximising a proxy. The
rubric scores a board, not a win. Search makes the proxy be pursued
*correctly*; it cannot make the proxy *right*.
"""

import random
import sys
import traceback
from typing import Callable

from pkm.types.obs import AreaType, OptionType, forced_picks

from .setupsearch_dep.scoring import evaluate as setup_evaluate
from .turn1agent_dep.search import Turn1Search


DRAKLOAK_CARD_ID = 120
CRUSHING_HAMMER_CARD_ID = 1120
LILLIES_DETERMINATION_CARD_ID = 1227

# Gates for the "dig before you attack" rule -- see _lillies_dig_option.
LILLIES_DIG_MAX_HAND = 5
LILLIES_DIG_MAX_DRAKLOAK = 2


def _recon_directive_option(obs: dict, sel: dict) -> int | None:
    """Index of an option that uses a Drakloak's Recon Directive, if offered.

    "Once during your turn, you may look at the top 2 cards of your deck and
    put 1 of them into your hand. Put the other card on the bottom of your
    deck." It costs nothing, cannot miss, and is once-per-Drakloak-per-turn, so
    there is no board state where declining it is right -- which makes it a
    hard rule rather than something to spend search on.

    Identified by resolving the option's (area, index) back to the Pokemon in
    that slot: ABILITY options carry no card id, and this deck has other
    abilities (Munkidori, Fezandipiti ex) that are *not* unconditionally good.
    """
    state = obs.get("current") or {}
    players = state.get("players") or []
    if not players:
        return None
    me = players[state.get("yourIndex", 0)]
    for i, opt in enumerate(sel.get("option") or []):
        if opt.get("type") != int(OptionType.ABILITY):
            continue
        area, idx = opt.get("area"), opt.get("index")
        if area == int(AreaType.ACTIVE):
            slots = me.get("active") or []
        elif area == int(AreaType.BENCH):
            slots = me.get("bench") or []
        else:
            continue
        if idx is None or not 0 <= idx < len(slots):
            continue
        poke = slots[idx]
        if poke and poke.get("id") == DRAKLOAK_CARD_ID:
            return i
    return None


def _first_option_of_type(sel: dict, opt_type: OptionType) -> int | None:
    """Index of the first offered option of `opt_type`, or None."""
    for i, opt in enumerate(sel.get("option") or []):
        if opt.get("type") == int(opt_type):
            return i
    return None


def _crushing_hammer_option(obs: dict, sel: dict) -> int | None:
    """Index of a playable Crushing Hammer, if one is offered.

    "Flip a coin. If heads, discard an Energy from 1 of your opponent's
    Pokemon." Hardcoded because **the rubric cannot see its value at all**:
    `_score_opponent` in pkm/rl/setup_turn_score.py scores only damage dealt,
    with no term for stripping the opponent's energy. So playing it registers
    as pure hand-size loss and the search would never do it -- not because the
    card is weak, but because the objective is blind to what it does.

    The engine only offers the option when it is legally playable, so this is a
    no-op when there is no valid target. Note it *is* a coin flip: roughly half
    the time the card is spent for nothing. That is an accepted cost of the
    rule, not an oversight -- an unplayed Hammer does nothing 100% of the time.
    """
    state = obs.get("current") or {}
    players = state.get("players") or []
    if not players:
        return None
    hand = players[state.get("yourIndex", 0)].get("hand") or []
    for i, opt in enumerate(sel.get("option") or []):
        if opt.get("type") != int(OptionType.PLAY):
            continue
        cid = opt.get("cardId") or 0
        idx = opt.get("index")
        if not cid and idx is not None and 0 <= idx < len(hand):
            cid = hand[idx].get("id")
        if cid == CRUSHING_HAMMER_CARD_ID:
            return i
    return None


def _lillies_dig_option(obs: dict, sel: dict) -> int | None:
    """Index of Lillie's Determination if we should dig with it before ending.

    Checked **only at the moment the turn is about to end** (an attack or a bare
    END TURN), which is the whole point: by then everything worth playing has
    been played, so the hand Lillie's shuffles away is leftovers, and the six
    fresh cards are a clean gain. Checked any earlier and it would bin cards the
    turn still wanted.

    Three gates, all required:
      * Lillie's is actually offered (so the Supporter-per-turn limit is free,
        and it also means a second one can never chain -- the engine stops
        offering Supporters once one is played).
      * **hand < 5.** The rubric pays 0.9/card up to five and nothing beyond, so
        refreshing a short hand to six is a *scored* gain (a hand of 3 goes
        2.70 -> 4.50); refreshing a full hand is not.
      * **fewer than 2 Drakloak in play.** With two already down the board is
        built and there is nothing left to dig for; below that, Drakloak is the
        hard prerequisite for Phantom Dive and worth digging toward.

    Exists because the search reliably declines this on its own: attacking is an
    exact leaf (+7 Itchy Pollen, no estimation error) while Lillie's is scored
    by a rollout, and an exact number beats an estimated one. Diagnostic sample
    9 was precisely this -- two Lillie's held, turn ended after two moves.
    """
    state = obs.get("current") or {}
    players = state.get("players") or []
    if not players:
        return None
    me = players[state.get("yourIndex", 0)]
    hand = me.get("hand") or []
    hand_size = len(hand) if hand else int(me.get("handCount") or 0)
    if hand_size >= LILLIES_DIG_MAX_HAND:
        return None

    in_play = [c for c in [*(me.get("active") or []), *(me.get("bench") or [])] if c]
    if sum(1 for c in in_play if c.get("id") == DRAKLOAK_CARD_ID) >= LILLIES_DIG_MAX_DRAKLOAK:
        return None

    for i, opt in enumerate(sel.get("option") or []):
        if opt.get("type") != int(OptionType.PLAY):
            continue
        cid = opt.get("cardId") or 0
        idx = opt.get("index")
        if not cid and idx is not None and 0 <= idx < len(hand):
            cid = hand[idx].get("id")
        if cid == LILLIES_DETERMINATION_CARD_ID:
            return i
    return None


def _hardened_rollout(base_policy: Callable[[dict], list[int]]) -> Callable[[dict], list[int]]:
    """Wrap a playout policy with the setup turn's no-brainer rules.

    The rollout is what MCTS uses to estimate a branch, and at ~3% tree
    expansion it plays most of every branch -- so its blind spots become the
    search's blind spots. `base_policy` (the default net) has one that matters
    here: it does not use Budew's Itchy Pollen, so every branch rolls out to a
    no-attack ending and "attack now" -- the one option scored without a
    rollout -- beats every line that would attack *after* setting up. That is
    exactly what cost samples 2 and 10 several points each.

    Three rules, in priority order:

      1. **Take Recon Directive the moment it is offered.** Free, cannot miss,
         once per Drakloak -- never wrong. Highest priority.
      2. **Play Crushing Hammer whenever it is offered.** The rubric has no
         term for stripping opponent energy, so the search is blind to it and
         would never play it on its own. High priority: it is an Item, so it
         never ends the turn and costs nothing but the card.
      3. **Never end the turn with an attack still available -- attack
         instead.** Deliberately the *lowest* priority: it only overrides an
         END-TURN, so the policy plays out everything productive first and the
         attack lands *last*. In this deck the available attack on the setup
         turn is Itchy Pollen (+7 and an item lock), essentially always right.

    Everything between -- evolve, Poffin, Poke Pad, attach, play a basic --
    stays with `base_policy`, which already handles them. This is a rollout
    estimate, not the real decision (the search still chooses that), so a rule
    being occasionally suboptimal only adds noise, never a forfeited move.
    """

    def policy(obs: dict) -> list[int]:
        sel = obs.get("select") or {}
        recon = _recon_directive_option(obs, sel)
        if recon is not None:
            return [recon]
        hammer = _crushing_hammer_option(obs, sel)
        if hammer is not None:
            return [hammer]
        picks = base_policy(obs)
        end_idx = _first_option_of_type(sel, OptionType.END)
        attack_idx = _first_option_of_type(sel, OptionType.ATTACK)
        # The turn is about to be over either way -- dig first if it pays.
        ending = (end_idx is not None and picks == [end_idx]) or (
            attack_idx is not None and picks == [attack_idx]
        )
        if ending:
            lil = _lillies_dig_option(obs, sel)
            if lil is not None:
                return [lil]
        # Only intervene when the base policy would end the turn.
        if picks == [end_idx] and end_idx is not None and attack_idx is not None:
            return [attack_idx]
        return picks

    return policy


def make_dragapult_setup_search_agent(
    deck: list[int],
    n_determinizations: int = 2,
    n_simulations: int = 1_000_000,
    time_budget_s: float = 1.7,
    seed: int | None = None,
    guided_rollouts: bool = True,
    rollout_hard_rules: bool = True,
    rollout_source: str = "setup",
    log_sink: Callable[[str], None] | None = None,
) -> Callable[[dict], list[int]]:
    """Create the search-based setup agent.

    **Time-bound, not count-bound.** `n_simulations` is deliberately set
    absurdly high so it never binds: the search simulates until
    `time_budget_s` is spent, split evenly across determinizations. That makes
    the knob you tune a wall-clock budget, which is the quantity that actually
    matters on a cumulative-time-limited platform.

    It used to run a fixed 100 simulations, which measurement showed was far
    too few: the space of ways to play a setup turn exceeds 20,000 sequences,
    100 simulations sampled ~0.5% of it, and raising to 500 found boards
    scoring ~3 points higher. The old default also used just 1.5% of its own
    wall-clock allowance -- the giveaway that it was stopping early for no
    reason.
    """
    rng = random.Random(seed)

    # --- playout policy -----------------------------------------------------
    # Estimate a branch by playing it out *competently* rather than at random.
    # A random playout scores a branch by its average continuation while the
    # real game takes the best one, so it understates every move that widens
    # the position -- Lillie's, Judge, Ultra Ball -- and understates nothing at
    # all about ending the turn. That asymmetry, not the rubric, is why this
    # agent was declining hand refreshes.
    #
    # Which trained net drives the playout, `rollout_source`:
    #   "setup"   -- policy_setup.npz, the net trained on THIS turn (2-turn
    #                episodes scored by the same rubric the search maximises).
    #                It has seen far more setup turns than the default net, so
    #                it should play the continuation more relevantly; the risk
    #                is that it shares the rubric's blind spots, since it is a
    #                learned approximation of the very thing being optimised.
    #   "default" -- policy.npz, the whole-game PPO net. Less setup-specific but
    #                an independent objective (winning), so its mistakes are at
    #                least uncorrelated with the rubric's.
    # Falls back to "default" if setup weights are missing (e.g. mid-retrain).
    #
    # Two things this deliberately does NOT do:
    #   * it uses `NumpyPolicy` directly, not `make_dragapult_*_agent`. Those
    #     factories close over a GameContext and call `tracker.observe()` on
    #     every call; as a playout policy that would swallow thousands of
    #     observations from branches that never happened and leak the corrupted
    #     deck-tracking state into the real game's decisions.
    #   * it never calls `observe()` on the scratch context below, for the same
    #     reason. The context stays neutral: its card-location features are
    #     uninformative rather than wrong, which is the safe failure mode.
    rollout_policy = None
    if guided_rollouts:
        try:
            from pkm.heuristics.context import GameContext
            from pkm.heuristics.deck_tracker import DeckTracker

            from .dragapult_default_agent import load_policy as _load_default
            from .dragapult_setup_agent import _find_setup_weights

            _policy = None
            if rollout_source == "setup":
                from pkm.rl.numpy_policy import NumpyPolicy

                _setup_path = _find_setup_weights(None)
                if _setup_path is not None:
                    _policy = NumpyPolicy.load(_setup_path)
                else:
                    msg = "setup_search: no policy_setup.npz -- rollout falls back to default net"
                    (log_sink or (lambda m: print(m, file=sys.stderr, flush=True)))(msg)
            if _policy is None:  # "default", or setup weights missing
                _policy = _load_default(None)

            if _policy is not None:
                _scratch = GameContext(list(deck), DeckTracker(deck))

                def rollout_policy(o: dict) -> list[int]:  # noqa: F811
                    return _policy.select(o, _scratch)
        except Exception as exc:  # weights missing -> stay on random playouts
            msg = f"setup_search: guided rollouts unavailable ({exc}) -- using random"
            if log_sink is not None:
                log_sink(msg)
            else:
                print(msg, file=sys.stderr, flush=True)

    # Layer the deck's no-brainer rules on top of the net playout: take Recon
    # Directive, and never end a rollout turn with Itchy Pollen unused. Kept
    # here rather than in Turn1Search so the search stays deck-agnostic.
    if rollout_hard_rules and rollout_policy is not None:
        rollout_policy = _hardened_rollout(rollout_policy)

    search = Turn1Search(
        n_determinizations=n_determinizations,
        n_simulations=n_simulations,
        time_budget_s=time_budget_s,
        rng=rng,
        rollout_policy=rollout_policy,
        evaluate_fn=setup_evaluate,
    )

    def _report(exc: Exception) -> None:
        """Never degrade to random silently -- see first_turn_agent."""
        msg = (
            f"setup_search: search failed ({type(exc).__name__}: {exc}) "
            "-- falling back to RANDOM picks"
        )
        if log_sink is not None:
            log_sink(msg)
        else:
            print(msg, file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    def agent(obs: dict) -> list[int]:
        sel = obs.get("select")
        if sel is None:
            return deck
        forced = forced_picks(sel)
        if forced is not None:
            return forced
        # Take Recon Directive the moment it is offered, before searching.
        # The agent is re-entered on every decision, so this also catches the
        # case where the ability only becomes available part-way through the
        # turn -- a Drakloak that was a Dreepy until this turn's evolution.
        recon = _recon_directive_option(obs, sel)
        if recon is not None:
            return [recon]
        # Same reasoning as Recon: the rubric cannot score what Crushing Hammer
        # does, so searching over it is wasted budget on a decision the rubric
        # is guaranteed to get wrong. Take it and move on.
        hammer = _crushing_hammer_option(obs, sel)
        if hammer is not None:
            return [hammer]
        try:
            picks = search.choose(obs, deck)
            # The search prices an attack exactly and a Lillie's by rollout, so
            # it systematically prefers the attack. Override at the one moment
            # the comparison is safe: the turn is ending anyway.
            end_idx = _first_option_of_type(sel, OptionType.END)
            attack_idx = _first_option_of_type(sel, OptionType.ATTACK)
            ending = (end_idx is not None and picks == [end_idx]) or (
                attack_idx is not None and picks == [attack_idx]
            )
            if ending:
                lil = _lillies_dig_option(obs, sel)
                if lil is not None:
                    return [lil]
            return picks
        except Exception as exc:
            # Deliberately broad: on Kaggle an uncaught exception forfeits the
            # match, which is strictly worse than one arbitrary pick.
            _report(exc)
            n = len(sel["option"])
            lo = min(max(int(sel.get("minCount") or 0), 0), n)
            k = min(max(int(sel.get("maxCount") or 0), lo), n)
            return rng.sample(range(n), k)

    return agent
