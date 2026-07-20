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
from .cards import BUDEW, DREEPY_LINE, ULTRA_BALL, XEROSICS_MACHINATIONS, count_in_play

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
        if t == OptionType.ATTACK:
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
                scoring.evaluate(fo, events, search.went_first) if fo else 0.0
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
    ):
        self.n_determinizations = n_determinizations
        self.n_simulations = n_simulations
        self.ucb_c = ucb_c
        self.time_budget_s = time_budget_s
        self.rng = rng or random.Random()
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

    def _rollout(self, state: SearchState, events: dict) -> float:
        """Uniform-random playout from `state` to the end of our turn."""
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
                acts = _enumerate_actions(sel, self.rng)
                picks = list(self.rng.choice(acts)) if acts else []
            events = _apply_events(events, obs, picks)
            state = search_step(state.search_id, picks)
            obs = state.raw_observation
        if final_obs is None:
            return 0.0
        return scoring.evaluate(final_obs, events, self.went_first)

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
        deadline = time.monotonic() + self.time_budget_s
        try:
            for _ in range(self.n_determinizations):
                det = sample_determinization(obs, my_decklist, opp_decklist, self.rng)
                root = _Node(self, search_begin(obs, **det), {})
                if root.leaf_value is not None or not root.actions:
                    continue
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
