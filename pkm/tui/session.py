"""The engine hook: run a match while a human answers the prompts.

``env.run()`` owns the game loop and calls agent functions synchronously, so a
human player cannot sit on Textual's event-loop thread. Instead ``env.run`` goes
on a worker thread and the "human agent" is a queue bridge: it pushes the parsed
observation to the UI and blocks until the UI posts picks back.

    main thread (Textual)                worker thread (env.run)
      BattleApp                            human_agent(obs)  <- kaggle calls this
         |                                     |  events.put(Prompt)
         |  <---------- events ---------------  |  picks.get()   [blocks]
         |  render, user chooses                |
         |  ----------- picks ---------------> |
         |                                      |  returns [0] -> battle_select

``GameSession`` is deliberately narrow (``start`` / ``next_event`` / ``submit`` /
``quit``). If this threaded approach ever proves unworkable, a session driving
``battle_start`` / ``battle_select`` directly implements the same four methods and
the UI does not change.

Two engine limits are disarmed at ``make()`` (both verified by measurement, see
the design doc): the cumulative 600 s overage clock, which would otherwise hand
the human a loss for thinking, and ``runTimeout``, which would abort the episode.
"""

import json
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Protocol

from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.types.obs import Observation

HUMAN = "human"

# Large enough to disable kaggle's timeouts. actTimeout has minimum 0 in the
# schema, so this must be large-positive, not negative.
NO_TIMEOUT = 1e9


@dataclass(frozen=True)
class Prompt:
    """The engine is asking the human to choose."""

    obs: Observation


@dataclass(frozen=True)
class Finished:
    """The game ended normally."""

    rewards: tuple[int | None, int | None]
    html_path: str | None = None
    replay_path: str | None = None


@dataclass(frozen=True)
class Failed:
    """The worker died. Without this the UI would wait on a queue forever."""

    error: BaseException


Event = Prompt | Finished | Failed


class _Quit:
    """Sentinel posted on the picks queue when the user quits."""


class _Abort(BaseException):
    """Raised inside the blocked agent to unwind env.run when the user quits.

    Must subclass BaseException, not Exception: kaggle's ``Agent.act()`` wraps
    every agent call in ``except Exception as e: action = e``, so an Exception
    here would be swallowed, turned into the agent's "action", and the player
    marked ERROR — env.run would then return normally and we would write
    result.html/replay.json for a game the human abandoned.
    """


class GameSession(Protocol):
    human_index: int

    def start(self) -> None: ...
    def next_event(self, timeout: float | None = None) -> Event: ...
    def submit(self, picks: list[int]) -> None: ...
    def quit(self) -> None: ...


class ThreadedEnvSession:
    """Runs kaggle's env.run on a worker thread; bridges the human via queues."""

    def __init__(
        self,
        deck: list[int],
        human_index: int,
        opponent: str,
        weights: str | None = None,
        html_path: str | None = "result.html",
        replay_path: str | None = "replay.json",
    ) -> None:
        self.deck = deck
        self.human_index = human_index
        self.opponent = opponent
        self.weights = weights
        self.html_path = html_path
        self.replay_path = replay_path
        self._events: queue.Queue[Event] = queue.Queue()
        self._picks: queue.Queue[list[int] | _Quit] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.ctx = GameContext(list(deck), DeckTracker(deck))

    # -- the human "agent" -------------------------------------------------

    def human_agent(self, obs: dict) -> list[int]:
        """Called by kaggle on the worker thread. Blocks until the user picks.

        Note kaggle has redirect_stdout active for the whole duration of this
        call — never print() from here or anywhere the UI runs.
        """
        self.ctx.tracker.observe(obs)
        if self.ctx.tracker.is_search_reveal(obs):
            self.ctx.tracker.record_search_reveal(obs)

        if obs["select"] is None:
            return self.deck  # deck submission is not a decision
        self._events.put(Prompt(Observation.model_validate(obs)))
        picks = self._picks.get()
        if isinstance(picks, _Quit):
            raise _Abort
        return picks

    # -- GameSession -------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def next_event(self, timeout: float | None = None) -> Event:
        return self._events.get(timeout=timeout)

    def submit(self, picks: list[int]) -> None:
        self._picks.put(picks)

    def quit(self) -> None:
        self._picks.put(_Quit())

    # -- worker ------------------------------------------------------------

    def _worker(self) -> None:
        try:
            self._events.put(self._run_env())
        except _Abort:
            return  # user quit; unwind quietly
        except BaseException as exc:  # noqa: BLE001 - must reach the screen
            self._events.put(Failed(exc))

    def _run_env(self) -> Finished:
        from kaggle_environments import make

        from pkm.rl.play import make_agent_by_name

        opponent_agent = make_agent_by_name(self.opponent, self.deck, self.weights)
        agents: list[Callable[[dict], list[int]]] = [None, None]  # type: ignore[list-item]
        # kaggle's Agent.act() inspects agent.__code__.co_argcount to decide how
        # many positional args to pass. A bound method's co_argcount includes the
        # implicit `self`, so passing self.human_agent directly makes kaggle think
        # it takes two args (obs, configuration) and it gets called with both,
        # raising a TypeError that kaggle silently swallows as an ERROR status.
        # Wrapping in a plain lambda (co_argcount == 1) avoids the miscount.
        agents[self.human_index] = lambda obs: self.human_agent(obs)
        agents[1 - self.human_index] = opponent_agent

        env = make(
            "cabt",
            configuration={
                "decks": [self.deck, self.deck],
                "actTimeout": NO_TIMEOUT,
                "runTimeout": NO_TIMEOUT,
            },
        )
        env.run(agents)

        final = env.steps[-1]
        if self.html_path:
            with open(self.html_path, "w") as f:
                f.write(env.render(mode="html"))
        if self.replay_path:
            data = env.toJSON()
            with open(self.replay_path, "w") as f:
                f.write(data) if isinstance(data, str) else json.dump(data, f)

        return Finished(
            rewards=(final[0].reward, final[1].reward),
            html_path=self.html_path,
            replay_path=self.replay_path,
        )
