import asyncio
import json
import queue
from pathlib import Path

from pkm.obs import Observation
from pkm.tui.app import BattleApp
from pkm.tui.session import Finished, Prompt

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)


class FakeSession:
    """A scripted session: no engine, no threads beyond the app's own pump."""

    def __init__(self, events):
        self.human_index = 0
        self._events = queue.Queue()
        for event in events:
            self._events.put(event)
        self.submitted: list[list[int]] = []
        self.quit_called = False

    def start(self) -> None:
        pass

    def next_event(self, timeout=None):
        return self._events.get(timeout=timeout)

    def submit(self, picks):
        self.submitted.append(picks)

    def quit(self) -> None:
        self.quit_called = True


def _main_obs() -> Observation:
    return Observation.model_validate(FIXTURE["observations"]["0:0"])


def _run(coro):
    asyncio.run(coro)


def test_pressing_a_number_then_enter_submits_that_pick():
    obs = _main_obs()
    session = FakeSession([Prompt(obs), Finished(rewards=(1, -1))])
    app = BattleApp(session, confirm_irreversible=False)

    async def go():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("1")
            await pilot.press("enter")
            await pilot.pause()

    _run(go())
    assert session.submitted == [[0]]


def test_enter_does_not_submit_below_min_count():
    obs = _main_obs()
    obs.select.minCount = 2
    obs.select.maxCount = 2
    session = FakeSession([Prompt(obs)])
    app = BattleApp(session, confirm_irreversible=False)

    async def go():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("1")
            await pilot.press("enter")  # only 1 of 2 picked: must not submit
            await pilot.pause()

    _run(go())
    assert session.submitted == []


def test_finished_event_shows_the_result():
    session = FakeSession([Finished(rewards=(1, -1))])
    app = BattleApp(session, confirm_irreversible=False)

    async def go():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()

    _run(go())
    assert app.result_text is not None
    assert "win" in app.result_text.lower()
