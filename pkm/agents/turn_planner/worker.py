"""The planner subprocess: simulates one whole turn in its own engine.

Why a separate process at all: the engine's search context is a single
process-global handle (``pkm/engine/api.py``'s ``_agent_ptr``), and
``search_end()`` recycles its memory arena. Running the planner's simulation
in the live process would share that arena with the deployed first-turn MCTS.
A child process gets its own ``AgentStart()`` handle, so a plan can never
corrupt -- or be corrupted by -- the real match, and a planner crash can't
take the match down with it.

**Fidelity limit (engine-imposed, not a shortcut).** Observations returned by
``search_step`` carry no ``search_begin_input``, so *no* search can be started
from a simulated state -- by any process arrangement. Any search-based agent
(the turn-1 MCTS) therefore cannot act inside the simulation. Where the
router would have picked one, the plan substitutes the search-free policy and
flags the decision ``substituted: true``, so the record never overstates how
faithful it is. Turns the real bot plays with the plain policy (turn 2
onward) are simulated exactly.
"""

from __future__ import annotations

import random
from typing import Any

MAX_SIM_DECISIONS = 200

# Loaded weights, reused across plans. The *agent* is still rebuilt per plan --
# only the immutable policy is shared. Re-reading policy.npz every plan cost
# ~6.6ms of a ~8.4ms plan, i.e. most of the work was redundant disk I/O.
_POLICY_CACHE: dict[str | None, Any] = {}


def _cached_policy(weights_path: str | None):
    """The exported policy for `weights_path`, loaded at most once per worker."""
    if weights_path not in _POLICY_CACHE:
        from pkm.agents.dragapult_default_agent import load_policy

        _POLICY_CACHE[weights_path] = load_policy(weights_path)
    return _POLICY_CACHE[weights_path]


def _warm_up(weights_path: str | None) -> None:
    """Pay the one-time engine + weights load now, off the critical path.

    Importing `pkm.engine` is cheap; *using* it is what costs ~3.5s, because
    the shared library loads lazily on first use. Reading the card table
    forces that load. Run right after the init ack, so it overlaps with the
    parent playing its opening decisions instead of stalling the first plan.
    """
    try:
        from pkm.data.card_data import get_card_data

        get_card_data()  # forces loader.get_lib()
        _cached_policy(weights_path)
    except Exception:
        pass  # a failed warm-up only means the first plan pays the cost itself


def _plan_turn(obs: dict, deck: list[int], seed: int, weights_path: str | None) -> dict:
    """Simulate the acting seat's current turn; return the recorded plan.

    Imports live inside the function so they happen in the *child*, keeping
    the engine (and its handle) entirely on this side of the process boundary.
    """
    from pkm.agents.dragapult_default_agent import make_dragapult_default_agent
    from pkm.agents.singaporean_middleman import _select_agent
    from pkm.engine import search_begin, search_end, search_step
    from pkm.mcts.determinize import infer_opponent_decklist, sample_determinization
    from pkm.types.obs import forced_picks

    from .summary import decision_context, describe_picks

    rng = random.Random(seed)
    state0 = obs["current"]
    me = state0["yourIndex"]
    turn = state0["turn"]

    # Search-free stand-in. It is the real agent for every non-first turn;
    # on the first turn it stands in for the MCTS (see module docstring).
    # Built fresh each plan so its DeckTracker starts clean -- carrying one
    # imagined world's card deductions into the next would corrupt them --
    # but the weights come from the cache rather than off disk again.
    policy = make_dragapult_default_agent(
        deck, weights_path, policy=_cached_policy(weights_path)
    )

    det = sample_determinization(obs, deck, infer_opponent_decklist(obs), rng)
    search_state = search_begin(obs, **det)

    decisions: list[dict[str, Any]] = []
    substituted_any = False
    ended = "turn_complete"
    try:
        for _ in range(MAX_SIM_DECISIONS):
            sim = search_state.raw_observation
            cur = sim.get("current") or {}
            if cur.get("result", -1) >= 0:
                ended = "game_over"
                break
            if sim.get("select") is None:
                ended = "no_select"
                break
            if cur.get("turn") != turn:
                ended = "turn_complete"
                break

            seat = cur.get("yourIndex")
            forced = forced_picks(sim["select"])
            if forced is not None:
                picks, who, substituted = forced, "forced", False
            elif seat != me:
                # the opponent acting inside our turn: play it out randomly
                sel = sim["select"]
                n = len(sel["option"])
                lo = min(max(int(sel.get("minCount") or 0), 0), n)
                k = min(max(int(sel.get("maxCount") or 0), lo), n)
                picks, who, substituted = (
                    rng.sample(range(n), k),
                    "opponent_random",
                    False,
                )
            else:
                routed = _select_agent(sim, {}, {})
                substituted = routed != "dragapult_default"
                substituted_any = substituted_any or substituted
                picks, who = policy(sim), routed

            decisions.append(
                {
                    "step": len(decisions),
                    "agent": who,
                    "substituted": substituted,
                    "picks": list(picks),
                    "means": describe_picks(sim, list(picks)),
                    **decision_context(sim),
                }
            )
            search_state = search_step(search_state.search_id, list(picks))
    finally:
        search_end()

    return {
        "turn": turn,
        "seat": me,
        "seed": seed,
        "ended": ended,
        "substituted_any": substituted_any,
        "decisions": decisions,
    }


def main() -> None:
    """Child entry point: newline-delimited JSON over stdin/stdout.

    Run as ``python -m pkm.agents.turn_planner.worker``. A plain subprocess
    (rather than ``multiprocessing``) on purpose: spawn-based multiprocessing
    re-imports the parent's ``__main__``, which fails outright unless every
    caller wraps itself in ``if __name__ == "__main__"``. A module entry point
    has no such requirement, so the planner works from any script.

    stdout is reserved for the protocol; it is swapped for stderr up front so
    that import banners and agent logging (kaggle_environments is noisy) can
    never corrupt a response line.
    """
    import json
    import sys

    protocol = sys.stdout
    sys.stdout = sys.stderr

    deck: list[int] = []
    weights_path: str | None = None

    def reply(obj: dict) -> None:
        protocol.write(json.dumps(obj) + "\n")
        protocol.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg is None or msg.get("cmd") == "stop":
            break
        if msg.get("cmd") == "init":
            deck = msg.get("deck") or []
            weights_path = msg.get("weights_path")
            # Ack *before* warming up: the parent must never block on the
            # engine load, it just wants to know the worker is alive.
            reply({"ok": True})
            _warm_up(weights_path)
            continue
        try:
            plan = _plan_turn(msg["obs"], deck, msg["seed"], weights_path)
            reply({"ok": True, "plan": plan})
        except Exception as exc:  # a failed plan must not kill the worker
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
