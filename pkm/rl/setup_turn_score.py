"""End-of-turn scoring for the Dragapult **setup** agent.

The setup agent is trained on a 2-turn episode, not a full match: the
first-turn agent plays turn 1 for both sides, the setup agent plays turn 2 for
both sides, and then this module scores the board the setup agent left behind
(after its attack). That terminal score is the dominant training signal; the
default agent's usual shaping terms ride along underneath at reduced weight.

**What a good turn-2 board looks like, and why.** `deck/03_pult_munki.csv`
carries no Rare Candy, so the only route to a Dragapult ex is
Dreepy -> Drakloak -> Dragapult ex, one evolution per turn. That makes a
**benched Drakloak the hard prerequisite** for attacking on turn 3: no
Drakloak in play at the end of turn 2 means no Phantom Dive on turn 3, full
stop. So the target state is:

  * bench   -- Drakloaks (carrying Fire/Psychic, which survives evolution),
               backed by Dreepies
  * active  -- a cheap, disposable staller (Budew, Munkidori); *not* a
               dreepy-line piece and *not* an ex, both of which hand the
               opponent tempo or a two-prize knockout
  * hand    -- a Dragapult ex to evolve into, plus energy
  * enemy   -- item-locked by Itchy Pollen, ideally damaged

Every term below is a module-level constant so the whole rubric can be tuned
in one place. Terms marked ``(spec)`` are the user-supplied rubric; those
marked ``(added)`` are proposed extensions -- set their weight to 0.0 to
disable one without touching the code.

Pure functions of the observation: no engine calls, no card database, so this
is cheap enough to call on every episode and importable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pkm.types.obs import EnergyType, Observation, Player, PokemonRef

# --- the cards this rubric reasons about -----------------------------------
DREEPY_CARD_ID = 119
DRAKLOAK_CARD_ID = 120
DRAGAPULT_EX_CARD_ID = 121
BUDEW_CARD_ID = 235
MUNKIDORI_CARD_ID = 112
FEZANDIPITI_EX_CARD_ID = 140
MEOWTH_EX_CARD_ID = 1071
MOLTRES_CARD_ID = 791

DREEPY_LINE_CARD_IDS = {DREEPY_CARD_ID, DRAKLOAK_CARD_ID, DRAGAPULT_EX_CARD_ID}
# The bodies this deck wants holding the active spot while the line builds:
# cheap, one-prize, and disruptive rather than valuable.
STALLER_CARD_IDS = {BUDEW_CARD_ID, MUNKIDORI_CARD_ID}
# Two prizes if knocked out -- a liability in the active spot on turn 2.
EX_CARD_IDS = {DRAGAPULT_EX_CARD_ID, FEZANDIPITI_EX_CARD_ID, MEOWTH_EX_CARD_ID}
# Phantom Dive's cost. Only these two types advance the line; a Dark energy
# belongs on Munkidori.
COMBO_ENERGY_TYPES = {EnergyType.FIRE, EnergyType.PSYCHIC}

STARTING_PRIZE_COUNT = 6

# --- weights: the whole rubric, tunable in one place ------------------------

# (spec) Itchy Pollen locks the opponent out of items for a turn -- the single
# highest-tempo thing turn 2 can do while the line is still assembling.
W_ITCHY_POLLEN = 7.0

# (spec) Per prize taken this turn.
W_KNOCKOUT = 3.0

# (spec) A benched Drakloak, keyed by how many *useful* (Fire/Psychic)
# energies it holds. NOTE the ordering is deliberately non-monotonic: one
# energy scores above two, which spreads investment across several Drakloaks
# rather than loading one up. See the module notes in the review for the
# monotonic alternative.
# Raised from 4/6/4. The level, not the shape, changed: the whole point of an
# Ultra Ball is the Drakloak it puts on the *board*, so the board term has to
# be large enough to pay for the three cards the ball costs. At 4.0 the full
# chain (ball, two discards, fetch, evolve) netted +0.3; at 6.0 it nets +1.4,
# and a plain evolution (Drakloak already in hand) went from +1.1 to +3.1.
# The +2.0 step from 0 to 1 energy is preserved, and stays above the +1.0 a
# Dreepy gets for the same energy, so charging a Drakloak always outranks
# charging a Dreepy.
W_BENCH_DRAKLOAK_BY_ENERGY: dict[int, float] = {0: 6.0, 1: 8.0, 2: 6.0}

# (spec) A benched Dreepy -- next turn's Drakloak. Tiered by how many are
# already down, because the first body is the one that matters: with no Dreepy
# in play there is no route to a Drakloak at all (this deck has no Rare Candy),
# while a fourth Dreepy is bench space a staller would use better.
#
# Raised from a flat 1.0. At 1.0 a benched Dreepy was worth *one tenth of a
# card*: playing one cost 0.9 of `hand_size` and returned 1.0, so the net was
# +0.10 and whether the bot bothered was decided by rounding noise. That also
# capped the payoff of everything upstream of it -- Poke Pad fetched a Dreepy
# that then had no reason to be played, and the evolution at the end of the
# Ultra Ball chain paid +0.10.
#
# Ceiling check: the top tier (2.0) plus the capped energy credit (1.0) tops
# out at 3.0, below the *cheapest* benched Drakloak (6.0), so a Dreepy never
# outranks a Drakloak at any energy count and evolving one is always a strict
# gain. That cap is what MAX_CREDITED_DREEPY_ENERGY exists for: uncapped, a
# 2-energy Dreepy outscored a 2-energy Drakloak, which made evolving it a
# negative-value move.
W_BENCH_DREEPY_BY_COUNT: dict[int, float] = {1: 2.0, 2: 1.5, 3: 1.0}
W_BENCH_DREEPY_BEYOND = 1.0  # a fourth and later Dreepy: as before

# (spec) Cards in hand: 0.9 each for the first five, and **nothing** beyond.
#
# Reduced from 2.0/1.0 with a knee at 6. At the old weights a full hand was
# worth +10 or more, which swamped every board term the rubric actually cares
# about -- in the diagnostic samples `hand_size` was routinely the single
# largest contributor, so "keep cards" outranked "build the Dragapult line".
# Holding cards is a means, not the goal; it is now priced accordingly.
#
# The tail is now flat (was 0.5/card) because paying anything for the 6th card
# onward makes hand-refresh Supporters strictly negative. Lillie's Determination
# shuffles the hand away and draws six: from a hand of eight that was
# 6.00 -> 5.00, a flat -1.00 charge *before* the new cards do anything, so the
# rubric paid the bot to sit on a dead hand. Observed directly -- with an empty
# bench and Ultra Ball, Drakloak and Lillie's in hand, the search agent's entire
# turn was "ATTACK", while the default net played Lillie's and built a
# three-Dreepy bench for +5 more points.
#
# Flat beyond five also makes discard costs free once the hand is large, which
# is what Ultra Ball's two-card discard needs to stop looking expensive.
W_HAND_CARD = 0.9
W_HAND_CARD_BEYOND = 0.0
HAND_SIZE_KNEE = 5

# (spec) A Dragapult ex in hand is what a benched Drakloak evolves into.
W_DRAGAPULT_IN_HAND = 1.0

# (added) The actual next-turn payoff: a bench Drakloak that can attack the
# turn after this one, paired with a Dragapult ex in hand to evolve it into.
# Scored per matched pair, since one Dragapult ex evolves one Drakloak. This
# is the term that rewards *completing* a combo, and so partly offsets the
# 1-over-2 energy ordering above.
#
# Two ways to be ready, because a Pokemon gets one attachment per turn:
#   READY      -- already holds Fire+Psychic; evolve and attack, no attachment
#                 spent, and nothing left to draw or hold onto.
#   READY_NEXT -- holds one of the two, with the *other type in hand*; evolve,
#                 attach it, attack. Worth less: it consumes the turn's only
#                 attachment and depends on still having that card.
W_PHANTOM_DIVE_READY = 6.0
W_PHANTOM_DIVE_READY_NEXT = 4.0

# (spec) An active Drakloak, keyed by useful energy. Deliberately small: it
# is worth something (it attacks, and it still evolves) but must stay far
# below the bench tiers so the search always prefers to charge the *benched*
# Drakloak. Replaces the -4 exposure penalty for this one card rather than
# stacking with it, so the net really is +0.5 / +1.0 as specified.
W_ACTIVE_DRAKLOAK_BY_ENERGY: dict[int, float] = {0: 0.5, 1: 1.0}

# (added) Who is holding the active spot when the turn ends.
W_ACTIVE_STALLER = 3.0  # Budew / Munkidori: exactly the intended body
W_ACTIVE_DREEPY_LINE = -4.0  # a combo piece exposed to a knockout
W_ACTIVE_EX = -3.0  # two prizes on the line

# (added) Energy on a benched Dreepy still counts -- it survives evolution
# into Drakloak, so it is early progress rather than waste.
W_BENCH_DREEPY_ENERGY = 1.0
MAX_CREDITED_DREEPY_ENERGY = 1

# (added) Fire/Psychic sitting on something outside the dreepy line is
# stranded: it can never pay for a Phantom Dive.
W_STRANDED_COMBO_ENERGY = -1.5

# (added) A third energy on a line member is wasted -- Phantom Dive needs two.
W_OVERCHARGED_LINE = -2.0

# (added) The deck expects at most three of the line in play; a fourth is
# bench space that a staller or Fezandipiti ex would use better. Mirrors
# `dreepy_line_field_potential`'s ceiling in pkm/rl/encoder.py.
#
# Charged per line member beyond the third. Raised from -2.0 to -5.0: at -2 a
# fourth *Dreepy* still netted positive against its own bench credit, so the
# ceiling was advisory rather than binding. At -5 a fourth Dreepy is clearly
# negative while a fourth *Drakloak* stays marginally worth playing, which is
# the intended ordering -- overcommitting with evolved pieces is the forgivable
# version of this mistake.
MAX_USEFUL_LINE_COUNT = 3
W_LINE_OVERCOMMIT = -5.0

# (added) Chip damage left on the opponent's board. Over a 2-turn episode all
# of it is ours, so this needs no before/after comparison.
W_DAMAGE_DEALT_PER_10HP = 0.3

# (added) Energy in hand, but only once this turn's attachment has already
# been used -- then it is genuinely *next* turn's attachment. Held while the
# attachment is still available it is not a resource, it is a move not made,
# and rewarding it competes directly with charging the line. Flat, not
# per-card: you can only ever attach one per turn, so the second and third
# energy in hand are not worth more than the first.
W_ENERGY_IN_HAND = 1.0

# (spec) A Buddy-Buddy Poffin still in hand at end of turn while the bench is
# still short of Dreepy. Poffin's whole job is to put Dreepy down for free, so
# holding it is not "resource conservation", it is a play left unmade.
#
# This exists to counteract a real pathology in the hand-size term: because
# holding a card pays and a card's effect only pays off on a *later* decision,
# every play prices as an immediate loss and ending the turn scores highest.
# Measured on a post-Lillie's board, every option was -0.5 to -1.0 while END
# TURN was +0.00 -- so the agent correctly stopped playing cards. Charging for
# an unplayed Poffin restores the incentive for the one card that should
# essentially always be played.
BUDDY_BUDDY_POFFIN_CARD_ID = 1086
W_POFFIN_UNPLAYED = -1.0
POFFIN_DREEPY_TARGET = 3  # below this many Dreepy in play, holding it is waste

# (spec) Poké Pad, same idea. "Search your deck for a Pokemon that doesn't have
# a Rule Box" -- which is precisely Dreepy and Drakloak (it cannot fetch
# Dragapult ex). While the bench is short of Dreepy there is an obvious target
# for it, so holding it is a play not made.
#
# Unlike Poffin it fetches to *hand* rather than to the bench, so playing it
# leaves hand size unchanged (the Pad leaves, the Pokemon arrives) and the
# board only improves once that Pokemon is actually played. The penalty is
# what makes the first half of that chain worth starting.
POKE_PAD_CARD_ID = 1152
W_POKE_PAD_UNPLAYED = -1.0

# (spec) Ultra Ball, but gated hard: it should fetch a Drakloak **only when
# there is no Drakloak anywhere on our field**.
#
# Measured on real boards, the rubric priced playing Ultra Ball at -2.2, +0.5,
# -3.3, -7.1 and +0.1 points, so declining it really was optimal. The cause is
# a cost/credit mismatch: the ball is three cards (itself plus two discards)
# plus one more to play what it found, i.e. -2.7 of `hand_size`, while a
# Drakloak sitting in hand scored exactly nothing -- unlike a Dragapult ex,
# which has always been worth W_DRAGAPULT_IN_HAND.
#
# The fix is deliberately *not* to pay for a Drakloak in hand. A Drakloak in
# hand is not the goal -- a Drakloak on the board is -- so W_DRAKLOAK_IN_HAND
# is 0.0 and the ball is justified by the bench term it unlocks instead.
# Paying for it in hand was also actively harmful: because that credit is gated
# on `drakloak_in_play == 0`, evolving the Drakloak destroyed its own bonus,
# and holding it scored 1.9 points better than putting it into play.
#
# What remains is the pressure to go get one, conditioned on
# `drakloak_in_play == 0` exactly as specified, and capped at one copy: one
# Drakloak ends the drought, so a second ball is not twice as urgent. Once a
# Drakloak is down the pressure vanishes and Ultra Ball is priced on its real
# cost again -- which is the "don't chain balls" behaviour we want.
ULTRA_BALL_CARD_ID = 1121
# The two Supporters that can excuse a Meowth ex -- see W_MEOWTH_EX_IN_PLAY.
LILLIES_DETERMINATION_CARD_ID = 1227
JUDGE_CARD_ID = 1213
CRISPIN_CARD_ID = 1198
W_DRAKLOAK_IN_HAND = 0.0  # deliberately zero -- see above
W_ULTRA_BALL_UNPLAYED = -1.0  # only while no Drakloak is in play

# (spec) Meowth ex on our board. A two-prize liability whose payoff does not
# serve the Dragapult plan, so putting it into play is charged for. Applies
# wherever it sits -- an ex on the bench is still two prizes waiting to be
# gusted up. Stacks with W_ACTIVE_EX when it is the one holding the active.
#
# **Conditional**, because the goal is not "never bench Meowth ex" but "only
# bench it when the Supporter it fetches earns the two prizes". Its ability
# searches a Supporter, so the charge is waived when the turn actually cashed
# that in, and only for the two Supporters worth the trade:
#
#   * Lillie's Determination -- always. Refreshing into six fresh cards is the
#     reason to run the Meowth at all.
#   * Judge -- only when we have **no Dreepy in play** at the moment it is
#     played. Judge cuts both hands to four, so it is a desperation dig: worth
#     it when the line has not started, a straight loss of cards when it has.
#
# Fetching a Supporter and not playing it, or playing any other Supporter,
# leaves the charge in place. Note the flag is evaluated *when the Supporter
# is played*, not at end of turn -- a Judge that successfully finds a Dreepy
# should stay excused, and an end-of-turn test would punish it for working.
W_MEOWTH_EX_IN_PLAY = -1.0

# (spec) Energy attached to a Meowth ex sitting on the bench. Comprehensively
# wasted: Meowth ex is not an attacker in this plan, it is not going to be
# promoted, and the energy could have gone on a Drakloak instead -- so this is
# a wasted attachment (the turn only gets one) *plus* two prizes fattened up.
#
# Priced at -20 deliberately: that is larger than the entire rest of a typical
# board score, so no combination of bench credit can make it worth doing. This
# is a "never" rule expressed as a weight rather than a hard rule, so the
# search and the trainer both see it. Charged per energised benched Meowth ex,
# and NOT waived by `meowth_excused` -- that waiver is about whether *playing*
# the Meowth was justified by the Supporter it fetched, which says nothing
# about whether feeding it energy afterwards was.
W_MEOWTH_BENCH_ENERGY = -20.0

# (spec) Crispin played while the board is still short of Drakloak.
#
# Crispin is a **Supporter**, and you get one Supporter per turn -- so playing
# it does not merely cost a card, it *forfeits Lillie's Determination for the
# turn*. That trade is: one energy attached plus one in hand, against a fresh
# six-card hand. While the line is still being assembled the refresh is worth
# far more, because what the board lacks is Drakloak, not energy.
#
# Allowed once three Drakloak are already down: at that point the board is
# built, there is nothing left to dig for, and attaching energy is the useful
# thing to do with the Supporter slot.
#
# Evaluated when Crispin is *played*, not at end of turn -- playing it into an
# empty board is the mistake regardless of what the rest of the turn recovers.
# Priced to dominate any energy it could attach (a 0->1 energy Drakloak is
# +2.0), so this is a prohibition expressed as a weight.
MIN_DRAKLOAK_FOR_CRISPIN = 3
W_CRISPIN_EARLY = -15.0

# (spec) Judge played while Lillie's Determination was sitting in hand.
#
# Same one-Supporter-per-turn economics as Crispin, but a worse trade: Judge
# cuts *both* hands to four, so it hands the opponent a fresh grip too, while
# Lillie's gives us six and costs them nothing. With Lillie's available there
# is no board state on the setup turn where Judge is the better use of the
# slot, so this is priced as a flat prohibition.
#
# Judge remains fine when Lillie's is *not* in hand -- that is the desperation
# dig the Meowth ex waiver already recognises (see W_MEOWTH_EX_IN_PLAY).
W_JUDGE_OVER_LILLIES = -15.0

# (spec) A second Budew on the field. One Budew is the deck's intended active
# staller (free Itchy Pollen, one prize); a *second* is dead weight -- it eats a
# bench slot the dreepy line wants, adds a prize, and its attack does nothing a
# single Budew is not already doing. Charged -5 per extra Budew, large enough to
# dominate the +0.9 hand-size gain from having played the card, so the search
# never benches the second one just to thin its hand. Scored on end-of-turn
# board state: two Budew in play means a redundant one was put down.
W_REDUNDANT_BUDEW = -5.0

# (spec) Moltres on our board. It is a one-of that does nothing for the
# Dragapult plan: it is not a dreepy-line piece, not one of the cheap stallers
# this deck wants holding the active, and it occupies a bench slot the line
# needs. Charged wherever it sits. Observed being benched for no scored gain
# (diagnostic sample 6), which is exactly the play this is meant to stop.
W_MOLTRES_IN_PLAY = -4.0

# (spec) Retreating during the setup turn. Retreat costs energy off the active
# and shuffles the board around at a point where the plan is simply "build the
# bench and leave a cheap staller up front" -- so by default it is churn.
#
# **Waived when the retreat lands a Budew in the active spot**, which is the
# one retreat this deck actually wants: Budew is the intended body (free Itchy
# Pollen, one prize). Tested on the end-of-turn active rather than on the
# retreat itself, since the rubric scores boards -- if a Budew is holding the
# active at the end of a turn we retreated in, the retreat did its job.
W_RETREAT = -2.0

# PPO terminal rewards are conventionally O(1) while this rubric totals tens of
# points. Divide by this before it reaches the trainer so it stays comparable
# to the win/loss signal rather than swamping it.
SCORE_SCALE = 20.0


@dataclass
class TurnScore:
    """A scored board, plus the per-term breakdown that produced it.

    The breakdown is the point: a bare total is untunable, whereas the
    contributions make it obvious which term is driving a decision.
    """

    total: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        if value:
            self.parts[name] = self.parts.get(name, 0.0) + value
            self.total += value

    @property
    def normalized(self) -> float:
        """`total` rescaled toward the O(1) range PPO expects."""
        return self.total / SCORE_SCALE

    def report(self) -> str:
        lines = [f"{name:<26}{value:+7.2f}" for name, value in self.parts.items()]
        lines.append(
            f"{'TOTAL':<26}{self.total:+7.2f}  (scaled {self.normalized:+.3f})"
        )
        return "\n".join(lines)


# --- helpers ---------------------------------------------------------------


def _bench(player: Player) -> list[PokemonRef]:
    return [p for p in player.bench if p is not None]


def _in_play(player: Player) -> list[PokemonRef]:
    active = player.active_pokemon
    return ([active] if active else []) + _bench(player)


def _combo_energy_count(poke: PokemonRef) -> int:
    """How many Fire/Psychic energies this Pokemon holds."""
    return sum(1 for e in poke.energies if e in COMBO_ENERGY_TYPES)


def _hand_ids(player: Player) -> list[int]:
    return [c.id for c in (player.hand or []) if c is not None]


def _hand_size(player: Player) -> int:
    """Always `handCount`, never `len(hand)`.

    `handCount` is visible from either point of view and is always current,
    whereas the `hand` list is hidden for the non-acting player -- and where
    the caller has spliced one in (see the trainer's `_with_own_hand`) it may
    be a decision stale. Size and contents therefore come from different
    places on purpose.
    """
    return player.handCount


# The basic energy this deck runs, card id -> the type it provides. Spelled
# out rather than inferred: the card ids happen to equal their EnergyType
# values, which is a coincidence of the card database, not a rule.
BASIC_ENERGY_CARD_TYPES: dict[int, EnergyType] = {
    2: EnergyType.FIRE,
    5: EnergyType.PSYCHIC,
    7: EnergyType.DARKNESS,
}


def _is_energy_card_id(card_id: int) -> bool:
    return card_id in BASIC_ENERGY_CARD_TYPES


def _hand_energy_types(player: Player) -> set[EnergyType]:
    """Which basic energy types are sitting in hand, ready to attach."""
    return {
        BASIC_ENERGY_CARD_TYPES[cid]
        for cid in _hand_ids(player)
        if cid in BASIC_ENERGY_CARD_TYPES
    }


def _attachments_to_phantom_dive(
    poke: PokemonRef, hand_types: set[EnergyType]
) -> int | None:
    """Attachments still needed before `poke` can pay Fire+Psychic.

    0 = already there, 1 = one type missing and that type is in hand, None =
    not reachable next turn. There is no "2": a Pokemon gets a single
    attachment per turn, so anything needing two is at least two turns away.
    """
    missing = COMBO_ENERGY_TYPES - set(poke.energies)
    if not missing:
        return 0
    if len(missing) == 1 and next(iter(missing)) in hand_types:
        return 1
    return None


# --- the rubric ------------------------------------------------------------


def score_end_of_turn(
    obs: Observation,
    itchy_pollen_used: bool = False,
    prizes_taken: int | None = None,
    seat: int | None = None,
    meowth_excused: bool = False,
    retreats: int = 0,
    crispin_early: bool = False,
    judge_over_lillies: bool = False,
) -> TurnScore:
    """Score the board the setup agent leaves at the end of its turn.

    `obs` is the observation *after* the turn's attack resolves. `seat` names
    whose board to score, defaulting to the observation's own point of view.
    Pass it explicitly when scoring a seat other than the one to move -- the
    2-turn trainer scores the first player's board from an observation taken
    once play has already passed to the second player.

    `itchy_pollen_used` must be supplied by the caller: `Log` entries carry
    only a `type`, with no payload naming the attack, so which attack was used
    is not recoverable from the observation alone. The training loop knows it,
    because it chose the option.

    `prizes_taken` likewise defaults to inferring from the prize pile, which is
    only sound because this is a 2-turn episode starting from a full six.
    Pass it explicitly for any other setting.

    `retreats` is how many times we retreated this turn -- an action, not a
    board property, so like `itchy_pollen_used` it has to come from the caller's
    event log. Callers without one leave it 0 and the term never fires.

    `judge_over_lillies` means Judge was played with Lillie's in hand.

    `crispin_early` likewise comes from the caller's event log: it means Crispin
    was played while fewer than MIN_DRAKLOAK_FOR_CRISPIN Drakloak were in play.

    `meowth_excused` waives the Meowth ex charge, and is another caller-supplied
    fact for the same reason as `itchy_pollen_used`: it depends on *which*
    Supporter was played and on the board at the time, neither of which survives
    into the end-of-turn observation. Callers without a turn event log leave it
    False, which reproduces the old unconditional penalty.
    """
    score = TurnScore()
    state = obs.current
    if state is None:
        return score

    if seat is None:
        seat = state.yourIndex
    me = state.players[seat]
    opponent = state.players[1 - seat]

    _score_tempo(score, me, itchy_pollen_used, prizes_taken)
    _score_bench(score, me)
    _score_active(score, me)
    _score_liabilities(
        score, me, meowth_excused, retreats, crispin_early, judge_over_lillies
    )
    _score_hand(score, me, bool(state.energyAttached))
    _score_readiness(score, me)
    _score_energy_discipline(score, me)
    _score_opponent(score, opponent)
    return score


def _score_tempo(
    score: TurnScore,
    me: Player,
    itchy_pollen_used: bool,
    prizes_taken: int | None,
) -> None:
    if itchy_pollen_used:
        score.add("itchy_pollen", W_ITCHY_POLLEN)
    if prizes_taken is None:
        prizes_taken = max(0, STARTING_PRIZE_COUNT - len(me.prize))
    if prizes_taken:
        score.add("knockouts", W_KNOCKOUT * prizes_taken)


def _score_bench(score: TurnScore, me: Player) -> None:
    """The bench is where this deck's turn 2 actually happens."""
    dreepy_seen = 0
    for poke in _bench(me):
        if poke.id == DRAKLOAK_CARD_ID:
            tier = min(_combo_energy_count(poke), 2)
            score.add("bench_drakloak", W_BENCH_DRAKLOAK_BY_ENERGY.get(tier, 0.0))
        elif poke.id == DREEPY_CARD_ID:
            # Diminishing: the first Dreepy is the one that unblocks the line.
            dreepy_seen += 1
            score.add(
                "bench_dreepy",
                W_BENCH_DREEPY_BY_COUNT.get(dreepy_seen, W_BENCH_DREEPY_BEYOND),
            )
            score.add(
                "bench_dreepy_energy",
                W_BENCH_DREEPY_ENERGY
                * min(_combo_energy_count(poke), MAX_CREDITED_DREEPY_ENERGY),
            )

    line_count = sum(1 for p in _in_play(me) if p.id in DREEPY_LINE_CARD_IDS)
    if line_count > MAX_USEFUL_LINE_COUNT:
        score.add(
            "line_overcommit",
            W_LINE_OVERCOMMIT * (line_count - MAX_USEFUL_LINE_COUNT),
        )


def _score_active(score: TurnScore, me: Player) -> None:
    """Who is left holding the active spot when the opponent's turn begins."""
    active = me.active_pokemon
    if active is None:
        return
    if active.id in STALLER_CARD_IDS:
        score.add("active_staller", W_ACTIVE_STALLER)
    if active.id == DRAKLOAK_CARD_ID:
        # An active Drakloak is a mild positive, not the liability an active
        # Dreepy is: it can attack (Dragon Headbutt) and still evolves next
        # turn. Scored on its own scale instead of the -4 exposure penalty,
        # and kept far below the bench tiers (4.0 / 6.0) so charging the
        # *bench* Drakloak always outranks charging this one.
        tier = min(_combo_energy_count(active), 1)
        score.add("active_drakloak", W_ACTIVE_DRAKLOAK_BY_ENERGY.get(tier, 0.0))
    elif active.id in DREEPY_LINE_CARD_IDS:
        score.add("active_dreepy_line", W_ACTIVE_DREEPY_LINE)
    if active.id in EX_CARD_IDS:
        score.add("active_ex", W_ACTIVE_EX)


def _score_liabilities(
    score: TurnScore,
    me: Player,
    meowth_excused: bool = False,
    retreats: int = 0,
    crispin_early: bool = False,
    judge_over_lillies: bool = False,
) -> None:
    """Cards whose presence on our board costs us regardless of where."""
    if crispin_early:
        score.add("crispin_early", W_CRISPIN_EARLY)
    if judge_over_lillies:
        score.add("judge_over_lillies", W_JUDGE_OVER_LILLIES)
    # Never waived -- see the weight's note.
    fed_meowths = sum(
        1 for p in _bench(me) if p.id == MEOWTH_EX_CARD_ID and p.energies
    )
    if fed_meowths:
        score.add("meowth_bench_energy", W_MEOWTH_BENCH_ENERGY * fed_meowths)

    moltres = sum(1 for p in _in_play(me) if p.id == MOLTRES_CARD_ID)
    if moltres:
        score.add("moltres_in_play", W_MOLTRES_IN_PLAY * moltres)

    # Retreating is churn unless it put the intended staller up front.
    if retreats:
        active = me.active_pokemon
        if active is None or active.id != BUDEW_CARD_ID:
            score.add("retreat", W_RETREAT * retreats)
    # A redundant Budew is charged unconditionally -- unlike the Meowth ex
    # penalty below, no play excuses a second one. `-1` so one Budew (the
    # intended active staller) is free and only extras are penalised.
    budews = sum(1 for p in _in_play(me) if p.id == BUDEW_CARD_ID)
    if budews > 1:
        score.add("redundant_budew", W_REDUNDANT_BUDEW * (budews - 1))

    if meowth_excused:
        return
    meowths = sum(1 for p in _in_play(me) if p.id == MEOWTH_EX_CARD_ID)
    score.add("meowth_ex_in_play", W_MEOWTH_EX_IN_PLAY * meowths)


def _score_hand(score: TurnScore, me: Player, attached_this_turn: bool) -> None:
    size = _hand_size(me)
    base = min(size, HAND_SIZE_KNEE) * W_HAND_CARD
    beyond = max(0, size - HAND_SIZE_KNEE) * W_HAND_CARD_BEYOND
    score.add("hand_size", base + beyond)

    hand = _hand_ids(me)
    pults = sum(1 for cid in hand if cid == DRAGAPULT_EX_CARD_ID)
    score.add("dragapult_in_hand", W_DRAGAPULT_IN_HAND * pults)
    # Only counts once the turn's single attachment is already spent.
    has_energy = any(_is_energy_card_id(cid) for cid in hand)
    if attached_this_turn and has_energy:
        score.add("energy_in_hand", W_ENERGY_IN_HAND)

    # Search cards left unplayed while the bench still wants Dreepy. Both can
    # fix that this turn, so holding either is a play not made.
    dreepy_in_play = sum(1 for p in _in_play(me) if p.id == DREEPY_CARD_ID)
    if dreepy_in_play < POFFIN_DREEPY_TARGET:
        poffins = sum(1 for cid in hand if cid == BUDDY_BUDDY_POFFIN_CARD_ID)
        score.add("poffin_unplayed", W_POFFIN_UNPLAYED * poffins)
        pads = sum(1 for cid in hand if cid == POKE_PAD_CARD_ID)
        score.add("pokepad_unplayed", W_POKE_PAD_UNPLAYED * pads)

    # Ultra Ball's target, and the pressure to go get it, exist only while the
    # field has no Drakloak at all. Both capped at one copy -- see the weights.
    if not any(p.id == DRAKLOAK_CARD_ID for p in _in_play(me)):
        if any(cid == DRAKLOAK_CARD_ID for cid in hand):
            score.add("drakloak_in_hand", W_DRAKLOAK_IN_HAND)
        elif any(cid == ULTRA_BALL_CARD_ID for cid in hand):
            # Only charge for the unplayed ball when it still has a job to do:
            # if the Drakloak is already in hand the ball is not the blocker.
            score.add("ultra_ball_unplayed", W_ULTRA_BALL_UNPLAYED)


def _score_readiness(score: TurnScore, me: Player) -> None:
    """Bench Drakloaks that can actually attack the turn after this one.

    Scored as matched pairs: a Drakloak is only a Phantom Dive if there is a
    Dragapult ex in hand to evolve it into, and one Dragapult ex evolves one
    Drakloak.

    Two tiers, because a Pokemon gets one attachment per turn. A Drakloak
    already holding Fire+Psychic needs none. A Drakloak holding one of them,
    with the other type in hand, needs exactly one -- also ready, but it
    spends the turn's only attachment, so only **one** such Drakloak can be
    finished no matter how many are waiting.
    """
    hand = _hand_ids(me)
    pults = sum(1 for cid in hand if cid == DRAGAPULT_EX_CARD_ID)
    hand_types = _hand_energy_types(me)

    charged = 0
    one_away = 0
    for poke in _bench(me):
        if poke.id != DRAKLOAK_CARD_ID:
            continue
        needed = _attachments_to_phantom_dive(poke, hand_types)
        if needed == 0:
            charged += 1
        elif needed == 1:
            one_away += 1

    ready_now = min(charged, pults)
    # Capped at 1: the turn has a single energy attachment to spend, however
    # many one-away Drakloaks are sitting there.
    ready_next = min(one_away, max(0, pults - ready_now), 1)

    score.add("phantom_dive_ready", W_PHANTOM_DIVE_READY * ready_now)
    score.add("phantom_dive_ready_next", W_PHANTOM_DIVE_READY_NEXT * ready_next)


def _score_energy_discipline(score: TurnScore, me: Player) -> None:
    """Fire/Psychic is a scarce resource -- it should all be on the line."""
    stranded = 0
    overcharge = 0
    for poke in _in_play(me):
        useful = _combo_energy_count(poke)
        if poke.id in DREEPY_LINE_CARD_IDS:
            overcharge += max(0, useful - 2)
        else:
            stranded += useful
    score.add("stranded_energy", W_STRANDED_COMBO_ENERGY * stranded)
    score.add("overcharged_line", W_OVERCHARGED_LINE * overcharge)


def _score_opponent(score: TurnScore, opponent: Player) -> None:
    damage = sum(max(0, p.maxHp - p.hp) for p in _in_play(opponent))
    score.add("damage_dealt", W_DAMAGE_DEALT_PER_10HP * (damage / 10.0))


if __name__ == "__main__":  # pragma: no cover - hand-run tuning aid
    from pkm.types.obs import Observation as _Obs

    _serial = 0

    def _poke(card_id: int, energies=(), hp: int = 70, max_hp: int = 70) -> dict:
        global _serial
        _serial += 1
        return {
            "id": card_id,
            "playerIndex": 0,
            "serial": _serial,
            "hp": hp,
            "maxHp": max_hp,
            "energies": list(energies),
        }

    def _state(active, bench, hand, opp_active, prizes: int = 6) -> _Obs:
        me = {
            "active": [active] if active else [],
            "bench": list(bench),
            "hand": [
                {"id": c, "playerIndex": 0, "serial": 900 + i}
                for i, c in enumerate(hand)
            ],
            "handCount": len(hand),
            "discard": [],
            "prize": [None] * prizes,
            "deckCount": 30,
        }
        opp = {
            "active": [opp_active] if opp_active else [],
            "bench": [],
            "hand": None,
            "handCount": 5,
            "discard": [],
            "prize": [None] * 6,
            "deckCount": 30,
        }
        return _Obs.model_validate(
            {
                "current": {
                    "yourIndex": 0,
                    "players": [me, opp],
                    "result": -1,
                    "turn": 2,
                    "firstPlayer": 0,
                },
                "select": None,
            }
        )

    FIRE, PSY = int(EnergyType.FIRE), int(EnergyType.PSYCHIC)

    print("=== ideal turn-2 board ===")
    ideal = _state(
        active=_poke(BUDEW_CARD_ID),
        bench=[
            _poke(DRAKLOAK_CARD_ID, [FIRE, PSY]),
            _poke(DRAKLOAK_CARD_ID, [FIRE]),
            _poke(DREEPY_CARD_ID, [PSY]),
        ],
        hand=[DRAGAPULT_EX_CARD_ID, 2, 1121, 1086, 1152],
        opp_active=_poke(112, hp=40, max_hp=70),
    )
    print(score_end_of_turn(ideal, itchy_pollen_used=True).report())

    print("\n=== bad turn-2 board (line exposed, energy stranded) ===")
    bad = _state(
        active=_poke(DREEPY_CARD_ID, [FIRE]),
        bench=[_poke(MUNKIDORI_CARD_ID, [PSY, PSY])],
        hand=[1121, 1152],
        opp_active=_poke(112),
    )
    print(score_end_of_turn(bad).report())
