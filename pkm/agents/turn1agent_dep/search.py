"""Turn-scoped Monte Carlo tree search for the first turn.

Runs on the engine's native SearchBegin/SearchStep forward simulation (the
same API pkm/mcts uses), but the tree only spans *our own current turn*: a
node is a leaf as soon as the game ends, the turn counter advances, or the
player to act is no longer us. Leaves are scored by ``scoring.evaluate`` (the
user-specified first-turn rubric), so the search picks the in-turn action
sequence with the best end-of-turn outcome. Unexpanded actions are estimated
with a uniformly-random rollout to the end of the turn (the Monte Carlo
part), then refined by UCB1-guided revisits.

Hidden zones are determinized exactly like pkm/mcts (sampled decks/hands
consistent with what we've seen); root visit counts are aggregated across
determinizations and the most-visited first action is played. The agent
re-runs the search on every decision, so the "best sequence" is re-validated
as its actual outcomes (draws, searches, coin flips) come in.
"""

import itertools
import random
import time

from pkm.engine import search_begin, search_end, search_step
from pkm.mcts.determinize import infer_opponent_decklist, sample_determinization
from pkm.types.obs import OptionType, SearchState, forced_picks

from . import scoring
from .cards import (
    BUDEW,
    CRISPIN,
    DRAKLOAK,
    DREEPY,
    DREEPY_LINE,
    JUDGE,
    LILLIES_DETERMINATION,
    ULTRA_BALL,
    XEROSICS_MACHINATIONS,
    count_in_play,
    hand_ids,
)

_MAX_FORCED_SKIP = 100
_MAX_ROLLOUT_STEPS = 80
_MAX_ACTIONS = 24  # per-node action-candidate cap (multi-pick combinatorics)


def _opt_card_id(obs: dict, opt: dict) -> int:
    """Best-effort card ID of an option (hand plays carry area/index)."""
    cid = opt.get("cardId")
    if cid:
        return cid
    if opt.get("area") == 2 and opt.get("index") is not None:  # AreaType.HAND
        state = obs["current"]
        player = state["players"][opt.get("playerIndex", state["yourIndex"])]
        hand = player.get("hand") or []
        if 0 <= opt["index"] < len(hand):
            return hand[opt["index"]]["id"]
    return 0


def _apply_events(events: dict, obs: dict, picks: list[int]) -> dict:
    """Return ``events`` updated with what this pick does (see scoring)."""
    sel = obs.get("select")
    if not sel or not picks:
        return events
    state = obs["current"]
    me = state["players"][state["yourIndex"]]
    out = events
    for i in picks:
        if not 0 <= i < len(sel["option"]):
            continue
        opt = sel["option"][i]
        t = opt.get("type")
        if t == OptionType.RETREAT:
            # An action, not a board property, so the rubric cannot see it
            # without this. Whether it is penalised depends on who ends up
            # active -- see W_RETREAT in pkm/rl/setup_turn_score.py.
            out = dict(out)
            out["retreats"] = out.get("retreats", 0) + 1
        elif t == OptionType.ATTACK:
            out = dict(out)
            out["attacked"] = True
            active = (me.get("active") or [None])[0]
            if active is not None and active["id"] == BUDEW:
                out["itchy_pollen"] = True
        elif t == OptionType.PLAY:
            cid = _opt_card_id(obs, opt)
            if cid == ULTRA_BALL:
                out = dict(out)
                out["ultra_balls"] = out.get("ultra_balls", 0) + 1
                out.setdefault("dreepy_at_ub", count_in_play(me, DREEPY_LINE))
            elif cid == XEROSICS_MACHINATIONS:
                out = dict(out)
                opp = state["players"][1 - state["yourIndex"]]
                out["xerosic_opp_hand"] = opp.get("handCount", 0)
            elif cid == CRISPIN:
                # Record the board as a *fact*; whether it is too early is the
                # setup rubric's call, applied in setupsearch_dep/scoring.py.
                # This module is shared with the first-turn agent, so it must
                # stay free of any one objective's policy.
                out = dict(out)
                out.setdefault("drakloak_at_crispin", count_in_play(me, {DRAKLOAK}))
            elif cid == LILLIES_DETERMINATION:
                out = dict(out)
                out["lillies_played"] = True
            elif cid == JUDGE:
                # Recorded *now*, not at end of turn: Judge is excused as a
                # desperation dig when the line has not started, and a Judge
                # that then finds a Dreepy must not lose its excuse for having
                # worked. See W_MEOWTH_EX_IN_PLAY in pkm/rl/setup_turn_score.py.
                out = dict(out)
                out["judge_played"] = True
                # Judge and Lillie's compete for the one Supporter slot, so
                # holding Lillie's while playing Judge is a fact worth
                # recording; the setup rubric decides what it costs.
                if LILLIES_DETERMINATION in hand_ids(me):
                    out["lillies_in_hand_at_judge"] = True
                if count_in_play(me, {DREEPY}) == 0:
                    out["judge_no_dreepy"] = True
    return out


def _enumerate_actions(sel: dict, rng: random.Random) -> list[tuple[int, ...]]:
    """Candidate pick-lists for one decision, capped at _MAX_ACTIONS."""
    n = len(sel["option"])
    mn, mx = sel["minCount"], sel["maxCount"]
    if mx <= 1:
        actions = [(i,) for i in range(n)]
        if mn == 0:
            actions.append(())
        return actions
    actions = []
    sizes = range(max(mn, 0), min(mx, n) + 1)
    for k in sorted(sizes, reverse=True):  # prefer maximal picks first
        if k == 0:
            actions.append(())
            continue
        combos = itertools.combinations(range(n), k)
        room = _MAX_ACTIONS - len(actions)
        if room <= 0:
            break
        actions.extend(tuple(c) for c in itertools.islice(combos, room))
    if len(actions) > _MAX_ACTIONS:
        actions = rng.sample(actions, _MAX_ACTIONS)
    return actions


class _Node:
    __slots__ = (
        "search_id",
        "obs",
        "events",
        "final_obs",
        "leaf_value",
        "actions",
        "visits",
        "totals",
        "children",
    )

    def __init__(
        self,
        search: "Turn1Search",
        state: SearchState,
        events: dict,
        prev_obs: dict | None = None,
    ):
        obs = state.raw_observation
        # `prev_obs` is the parent's state -- the one the action was taken on.
        # It's the fallback score-on state for a turn-ENDING action (attack /
        # end turn), whose own resulting observation belongs to the opponent
        # and so can't be scored. Without it those leaves evaluated to 0.0,
        # which made the search never voluntarily end the turn (it would
        # retreat rather than attack, since retreating kept a positive score).
        final_obs = obs if _is_ours(obs, search.me) else prev_obs
        for _ in range(_MAX_FORCED_SKIP):
            if obs["current"]["result"] >= 0 or not _is_ours(obs, search.me):
                break
            forced = forced_picks(obs["select"])
            if forced is None:
                break
            final_obs = obs
            events = _apply_events(events, obs, forced)
            state = search_step(state.search_id, forced)
            obs = state.raw_observation

        self.search_id = state.search_id
        self.obs = obs
        self.events = events
        # last our-perspective state seen on the way here; leaves are scored
        # on it (the turn-ending action doesn't change our own board)
        self.final_obs = obs if _is_ours(obs, search.me) else final_obs
        self.leaf_value: float | None = None
        if (
            obs["current"]["result"] >= 0
            or not _is_ours(obs, search.me)
            or obs["current"]["turn"] != search.turn
        ):
            fo = self.final_obs
            self.leaf_value = (
                search.evaluate_fn(fo, events, search.went_first) if fo else 0.0
            )
            self.actions = []
        else:
            self.actions = _enumerate_actions(obs["select"], search.rng)
        self.visits = [0] * len(self.actions)
        self.totals = [0.0] * len(self.actions)
        self.children: dict[int, _Node] = {}


def _is_ours(obs: dict, me: int) -> bool:
    return (
        obs["current"]["result"] < 0
        and obs.get("select") is not None
        and obs["current"]["yourIndex"] == me
    )


class Turn1Search:
    def __init__(
        self,
        n_determinizations: int = 2,
        n_simulations: int = 40,
        ucb_c: float = 1.4,
        time_budget_s: float = 6.0,
        rng: random.Random | None = None,
        rollout_policy=None,
        evaluate_fn=None,
    ):
        self.n_determinizations = n_determinizations
        self.n_simulations = n_simulations
        self.ucb_c = ucb_c
        self.time_budget_s = time_budget_s
        self.rng = rng or random.Random()
        # How an end-of-turn board is scored: ``(final_obs, events,
        # went_first) -> float``. Defaults to the first-turn rubric, so the
        # deployed first-turn agent is unchanged. Swapping it is what lets the
        # same turn-scoped search optimise a *different* turn's goal --
        # `dragapult_setup_search_agent` passes the setup rubric instead.
        self.evaluate_fn = evaluate_fn or scoring.evaluate
        # Optional playout policy. None keeps the original uniform-random
        # playout, which is what the first-turn agent still uses.
        self.rollout_policy = rollout_policy
        # per-choose() context, set in choose()
        self.me = 0
        self.turn = 0
        self.went_first = True
        self._vmin = 0.0
        self._vmax = 1.0

    # --- internals ---

    def _norm(self, v: float) -> float:
        if v < self._vmin:
            self._vmin = v
        if v > self._vmax:
            self._vmax = v
        span = self._vmax - self._vmin
        return (v - self._vmin) / span if span > 0 else 0.5

    def _rollout_picks(self, obs: dict, sel: dict) -> list[int]:
        """One playout decision: the rollout policy if it has one, else random.

        **Why a policy matters here.** A playout estimates a branch by *some*
        continuation, but the real game would take the *best* one -- so the
        estimate is short of the truth by roughly (max - mean) over the
        continuations, and that gap grows with how many continuations exist.
        The error is therefore not uniform across branches: END TURN has none
        left and is scored exactly, while a hand-refresh like Lillie's opens up
        dozens and is scored by an average. Playing randomly makes every move
        that *widens* the position look worse than it is, which is exactly the
        bias that had the setup agent declining Lillie's, Judge and Ultra Ball
        and sometimes ending its turn having done nothing at all.

        Playing the continuation competently shrinks (max - mean) and with it
        the bias. The cost is real -- a numpy forward pass measured 0.294ms
        against 0.004ms for a random pick, ~8.6x per rollout -- so this trades
        many noisy estimates for fewer accurate ones.

        Falls back to random on anything unexpected: a playout is a heuristic,
        and an exception here would abort a search that random picks can finish.
        """
        if self.rollout_policy is not None:
            picks = None
            try:
                picks = self.rollout_policy(obs)
            except Exception:
                picks = None
            if picks is not None:
                n = len(sel["option"])
                lo = int(sel.get("minCount") or 0)
                hi = max(int(sel.get("maxCount") or 0), lo)
                if lo <= len(picks) <= hi and all(0 <= i < n for i in picks):
                    return list(picks)
        acts = _enumerate_actions(sel, self.rng)
        return list(self.rng.choice(acts)) if acts else []

    def _rollout(self, state: SearchState, events: dict) -> float:
        """Playout from `state` to the end of our turn (see _rollout_picks)."""
        obs = state.raw_observation
        final_obs = obs if _is_ours(obs, self.me) else None
        for _ in range(_MAX_ROLLOUT_STEPS):
            if (
                obs["current"]["result"] >= 0
                or not _is_ours(obs, self.me)
                or obs["current"]["turn"] != self.turn
            ):
                break
            final_obs = obs
            sel = obs["select"]
            picks = forced_picks(sel)
            if picks is None:
                picks = self._rollout_picks(obs, sel)
            events = _apply_events(events, obs, picks)
            state = search_step(state.search_id, picks)
            obs = state.raw_observation
        if final_obs is None:
            return 0.0
        return self.evaluate_fn(final_obs, events, self.went_first)

    def _ucb_index(self, node: _Node) -> int:
        import math

        total = sum(node.visits) + 1
        best, best_i = -1e18, 0
        for i in range(len(node.actions)):
            if node.visits[i] == 0:
                u = 1e9 + self.rng.random()  # untried first, random order
            else:
                q = self._norm(node.totals[i] / node.visits[i])
                u = q + self.ucb_c * math.sqrt(math.log(total) / node.visits[i])
            if u > best:
                best, best_i = u, i
        return best_i

    def _simulate(self, root: _Node) -> None:
        node = root
        path: list[tuple[_Node, int]] = []
        while True:
            if node.leaf_value is not None:
                v = node.leaf_value
                break
            if not node.actions:
                v = 0.0
                break
            i = self._ucb_index(node)
            path.append((node, i))
            child = node.children.get(i)
            if child is None:
                picks = list(node.actions[i])
                ev = _apply_events(node.events, node.obs, picks)
                child = _Node(
                    self, search_step(node.search_id, picks), ev, prev_obs=node.obs
                )
                node.children[i] = child
                if child.leaf_value is not None:
                    v = child.leaf_value
                else:
                    # fresh interior node: Monte Carlo rollout to turn end
                    v = self._rollout(
                        SearchState(
                            {
                                "observation": child.obs,
                                "searchId": child.search_id,
                            }
                        ),
                        child.events,
                    )
                break
            node = child
        for parent, i in path:
            parent.visits[i] += 1
            parent.totals[i] += v

    # --- public API ---

    def choose(self, obs: dict, my_decklist: list[int]) -> list[int]:
        """Search the current (real) decision; returns the picks to play."""
        state = obs["current"]
        self.me = state["yourIndex"]
        self.turn = state["turn"]
        self.went_first = state.get("firstPlayer", -1) == self.me
        self._vmin, self._vmax = 0.0, 1.0

        opp_decklist = infer_opponent_decklist(obs)
        agg: dict[tuple[int, ...], int] = {}
        # `time_budget_s` is the budget for the whole decision, split evenly
        # across determinizations. A single shared deadline would let the
        # first sampled world spend everything and leave later ones zero
        # simulations -- silently collapsing to one determinization, which is
        # exactly the hedge against a bad guess about hidden cards that having
        # several is for. Only bites when simulations are time-bound rather
        # than count-bound (see `n_simulations`).
        per_det = self.time_budget_s / max(self.n_determinizations, 1)
        try:
            for _ in range(self.n_determinizations):
                det = sample_determinization(obs, my_decklist, opp_decklist, self.rng)
                root = _Node(self, search_begin(obs, **det), {})
                if root.leaf_value is not None or not root.actions:
                    continue
                deadline = time.monotonic() + per_det
                for _ in range(self.n_simulations):
                    if time.monotonic() > deadline:
                        break
                    self._simulate(root)
                for a, n in zip(root.actions, root.visits):
                    agg[a] = agg.get(a, 0) + n
        finally:
            search_end()

        if not agg:
            raise RuntimeError("turn-1 search produced no root actions")
        best = max(agg.items(), key=lambda kv: kv[1])[0]
        return list(best)
