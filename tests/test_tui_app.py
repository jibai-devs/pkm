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


def test_board_shows_the_real_prize_count():
    # Prize entries are all None (face-down, even your own); the list shrinks as
    # prizes are taken. Counting non-None entries would always render 0.
    from pkm.tui.widgets import BoardPanel

    obs = _main_obs()
    player = obs.me
    assert player.prize == [None] * 6, "fixture should have 6 face-down prizes"

    panel = BoardPanel("YOU")
    rendered: list[str] = []
    panel.update = rendered.append  # type: ignore[method-assign]
    panel.show(player)

    assert "prizes 6" in rendered[0]


def test_input_is_inert_while_waiting_for_the_agent():
    # Pressing enter twice must not queue a second pick: the extra picks would
    # be consumed blind by the *next* prompt, which the human never sees.
    obs = _main_obs()
    session = FakeSession([Prompt(obs)])
    app = BattleApp(session, confirm_irreversible=False)

    async def go():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("1")
            await pilot.press("enter")
            await pilot.press("2")  # ignored: still waiting
            await pilot.press("enter")  # ignored: still waiting
            await pilot.pause()

    _run(go())
    assert session.submitted == [[0]]


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
