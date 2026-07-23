"""Parent-side turn planner: owns the subprocess, compares plan vs. reality.

Opt-in and inert unless ``PKM_TURN_PLAN_DIR`` is set (it names the output
directory). Off by default so the Kaggle submission never pays the latency --
there's a cumulative 600s clock -- and so a planner bug can't affect a match.

Every entry point is failure-tolerant by construction: if the subprocess won't
start, times out, or errors, planning is silently skipped for that turn and
the live game proceeds untouched. A diagnostic must never cost a match.
"""

from __future__ import annotations

import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .summary import decision_context, describe_picks, picks_match

DEFAULT_TIMEOUT_S = 60.0
WORKER_MODULE = "pkm.agents.turn_planner.worker"

# OptionType -> verb, for printing a plan in something a human can read.
_VERB = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL",
    5: "ENERGY-CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END TURN",
    15: "SKILL",
    16: "CONDITION",
}
_AREA = {4: "active", 5: "bench"}


def plan_dir() -> str | None:
    """Configured output directory, or None when planning is disabled."""
    return os.environ.get("PKM_TURN_PLAN_DIR") or None


def _card_name(card_id: int) -> str:
    """Card name if the engine can resolve it, else a stable placeholder."""
    try:
        from pkm.data.card_data import get_card_by_id

        card = get_card_by_id(card_id)
        return card.name if card else f"Card#{card_id}"
    except Exception:
        return f"Card#{card_id}"


def _attack_name(attack_id: int) -> str:
    """Attack name if resolvable, else a stable placeholder."""
    try:
        from pkm.data.card_data import get_attack_data

        attack = get_attack_data().get(attack_id)
        return attack.name if attack else f"attack#{attack_id}"
    except Exception:
        return f"attack#{attack_id}"


def render_move(means: list[dict]) -> str:
    """One planned decision as readable text, e.g. ``ATTACH Fire Energy -> bench 1``."""
    if not means:
        return "(no options)"
    parts = []
    for d in means:
        text = _VERB.get(d.get("type"), f"OPT{d.get('type')}")
        if "card" in d:
            text += f" {_card_name(d['card'])}"
        if "attack" in d:
            text += f" {_attack_name(d['attack'])}"
        to = d.get("to")
        if to and to[0] in _AREA:
            where = _AREA[to[0]]
            text += f" -> {where}" + (f" {to[1]}" if where == "bench" else "")
        parts.append(text)
    return " + ".join(parts)


class TurnPlanner:
    """Plans a turn at its first decision, then scores reality against it."""

    def __init__(
        self,
        deck: list[int],
        weights_path: str | None = None,
        log_sink: Callable[[str], None] | None = None,
        timeout_s: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.deck = list(deck)
        self.weights_path = weights_path
        self.log_sink = log_sink
        self.timeout_s = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S
        self.rng = random.Random(seed)
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()
        self._broken = False
        self.game_id = f"{int(time.time())}_{os.getpid()}"
        # current turn's plan + how far reality has followed it
        self._plan: dict[str, Any] | None = None
        self._cursor = 0
        self._actual: list[dict[str, Any]] = []
        self._diverged_at: int | None = None

    # --- logging -----------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.log_sink is not None:
            self.log_sink(msg)
        else:
            print(msg, flush=True)

    # --- subprocess --------------------------------------------------------

    def _pump(self) -> None:
        """Drain the child's protocol stream onto a queue (so reads can time out)."""
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if line:
                try:
                    self._replies.put(json.loads(line))
                except ValueError:
                    pass  # not a protocol line; ignore
        self._replies.put(None)  # stream closed => child gone

    def _ensure_worker(self) -> bool:
        """Start the planner process if needed. False if unavailable."""
        if self._broken:
            return False
        if self._proc is not None and self._proc.poll() is None:
            return True
        try:
            # `pkm` must be importable in the child; inherit our own resolution
            env = dict(os.environ)
            import pkm

            root = str(Path(pkm.__file__).resolve().parent.parent)
            env["PYTHONPATH"] = os.pathsep.join(
                [root, env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            self._proc = subprocess.Popen(
                [sys.executable, "-m", WORKER_MODULE],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # child's import noise, discarded
                text=True,
                bufsize=1,
                env=env,
            )
            threading.Thread(target=self._pump, daemon=True).start()
            self._send(
                {"cmd": "init", "deck": self.deck, "weights_path": self.weights_path}
            )
            if self._await_reply() is None:
                raise RuntimeError("worker did not acknowledge init")
            return True
        except Exception as exc:
            self._broken = True
            self._log(f"turn_planner: worker unavailable ({type(exc).__name__}: {exc})")
            return False

    def _send(self, msg: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _await_reply(self) -> dict | None:
        """Next protocol message, or None on timeout / child exit."""
        try:
            return self._replies.get(timeout=self.timeout_s)
        except queue.Empty:
            return None

    def _request(self, obs: dict) -> dict | None:
        if not self._ensure_worker():
            return None
        try:
            self._send({"obs": obs, "seed": self.rng.randrange(2**31)})
            reply = self._await_reply()
        except Exception as exc:
            self._broken = True
            self._log(f"turn_planner: worker died ({type(exc).__name__}: {exc})")
            return None
        if reply is None:
            self._broken = True
            self._log(f"turn_planner: no reply within {self.timeout_s}s -- disabled")
            return None
        if not reply.get("ok"):
            self._log(f"turn_planner: plan failed ({reply.get('error')})")
            return None
        return reply["plan"]

    # --- public API --------------------------------------------------------

    def start(self) -> None:
        """Boot the worker now, without waiting for it to finish warming up.

        `Popen` returns immediately and the child preloads the engine on its
        own, so calling this at agent-construction time lets that ~3.5s land
        in parallel with the opening decisions rather than in front of the
        first plan request. Safe to call more than once.
        """
        if plan_dir() is None:
            return
        self._ensure_worker()

    def phantom_dive_odds(self, obs: dict, sims: int = 12) -> dict | None:
        """Probability that Phantom Dive is reachable this turn.

        Plays the turn out `sims` times, resampling the hidden zones each
        time, and reports how often Phantom Dive became legal. That spread is
        the point: reachability can hinge on what Lillie's Determination
        draws or what Drakloak's ability finds, so a single lookahead would
        report a coin flip as a certainty. Returns None if planning is off or
        the worker is unavailable -- never raises into the match.
        """
        if plan_dir() is None:
            return None
        if not self._ensure_worker():
            return None
        try:
            self._send(
                {
                    "cmd": "phantom_dive_odds",
                    "obs": obs,
                    "seed": self.rng.randrange(2**31),
                    "sims": sims,
                }
            )
            reply = self._await_reply()
        except Exception as exc:
            self._broken = True
            self._log(f"turn_planner: worker died ({type(exc).__name__}: {exc})")
            return None
        if reply is None:
            self._broken = True
            self._log(f"turn_planner: no reply within {self.timeout_s}s -- disabled")
            return None
        if not reply.get("ok"):
            self._log(f"turn_planner: odds failed ({reply.get('error')})")
            return None
        return reply["odds"]

    def start_turn(self, obs: dict) -> None:
        """Called on the first decision of a new turn: plan the whole turn."""
        self._flush()  # persist the turn that just ended
        self._plan, self._cursor, self._actual, self._diverged_at = None, 0, [], None
        if plan_dir() is None:
            return
        plan = self._request(obs)
        if plan is None:
            return
        self._plan = plan
        note = " (first-turn MCTS substituted)" if plan.get("substituted_any") else ""
        decisions = plan["decisions"]
        self._log(
            f"turn_planner: plan for turn {plan['turn']} -- "
            f"{len(decisions)} moves{note}:"
        )
        for dec in decisions:
            mark = "~" if dec.get("substituted") else " "
            self._log(
                f"  {mark}{dec['step'] + 1:>2}. [{dec['agent']}] "
                f"{render_move(dec['means'])}"
            )

    def record_actual(self, obs: dict, picks: list[int], agent: str) -> None:
        """Called on every real decision: score it against the plan."""
        if self._plan is None:
            return
        means = describe_picks(obs, list(picks))
        entry = {
            "step": len(self._actual),
            "agent": agent,
            "picks": list(picks),
            "means": means,
            **decision_context(obs),
        }
        planned = (
            self._plan["decisions"][self._cursor]
            if self._cursor < len(self._plan["decisions"])
            else None
        )
        if planned is None:
            entry["match"] = "past_plan"
        else:
            matched = picks_match(planned["means"], means)
            entry["match"] = "match" if matched else "differs"
            entry["planned_agent"] = planned["agent"]
            entry["planned_means"] = planned["means"]
            if not matched and self._diverged_at is None:
                self._diverged_at = entry["step"]
                self._log(
                    f"turn_planner: diverged from plan at step {entry['step']} "
                    f"(planned {planned['means']}, actual {means})"
                )
        self._actual.append(entry)
        self._cursor += 1

    def close(self) -> None:
        """Persist the in-flight turn and shut the worker down."""
        self._flush()
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._send({"cmd": "stop"})
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
        except Exception:
            pass
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass
        self._proc = None

    # --- persistence -------------------------------------------------------

    def _flush(self) -> None:
        """Write the finished turn's plan + what actually happened."""
        if self._plan is None:
            return
        directory = plan_dir()
        if directory is None:
            self._plan = None
            return
        n_matched = sum(1 for a in self._actual if a.get("match") == "match")
        record = {
            "game_id": self.game_id,
            "turn": self._plan["turn"],
            "seat": self._plan["seat"],
            "plan": self._plan,
            "actual": self._actual,
            "summary": {
                "planned_decisions": len(self._plan["decisions"]),
                "actual_decisions": len(self._actual),
                "matched": n_matched,
                "diverged_at": self._diverged_at,
                "followed_plan_fully": self._diverged_at is None
                and len(self._actual) > 0,
            },
        }
        try:
            os.makedirs(directory, exist_ok=True)
            path = Path(directory) / (
                f"plan_{self.game_id}_turn{self._plan['turn']:03d}.json"
            )
            with open(path, "w") as fh:
                json.dump(record, fh, indent=2)
        except Exception as exc:
            self._log(
                f"turn_planner: could not write plan ({type(exc).__name__}: {exc})"
            )
        self._plan = None
