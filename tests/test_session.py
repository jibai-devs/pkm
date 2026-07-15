import json
from pathlib import Path

import pytest

from pkm.types.obs import Observation
from pkm.tui.session import (
    Failed,
    Finished,
    HUMAN,
    Prompt,
    ThreadedEnvSession,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)
RAW_MAIN = FIXTURE["observations"]["0:0"]


def test_human_agent_returns_deck_when_select_is_none():
    session = ThreadedEnvSession(deck=[1] * 60, human_index=0, opponent="random")
    assert session.human_agent({"select": None}) == [1] * 60


def test_human_agent_blocks_then_returns_the_submitted_picks():
    session = ThreadedEnvSession(deck=[1] * 60, human_index=0, opponent="random")

    # Pre-load a pick, then call the agent: it should consume the prompt and
    # return our picks without ever touching the engine.
    session.submit([2])
    picks = session.human_agent(RAW_MAIN)

    assert picks == [2]
    event = session.next_event(timeout=1)
    assert isinstance(event, Prompt)
    assert isinstance(event.obs, Observation)
    assert event.obs.select is not None


def test_quit_aborts_the_blocked_agent():
    from pkm.tui.session import _Abort

    session = ThreadedEnvSession(deck=[1] * 60, human_index=0, opponent="random")
    session.quit()
    with pytest.raises(_Abort):
        session.human_agent(RAW_MAIN)


def test_abort_survives_kaggles_exception_handler():
    """_Abort must not subclass Exception.

    kaggle's Agent.act() does `except Exception as e: action = e`, so an
    Exception raised by the human agent gets swallowed, becomes the agent's
    "action", and marks the player ERROR — env.run then returns normally and we
    would write a replay for a game the human abandoned.
    """
    from pkm.tui.session import _Abort

    assert issubclass(_Abort, BaseException)
    assert not issubclass(_Abort, Exception)


@pytest.mark.slow
def test_quit_mid_game_unwinds_the_real_env_and_writes_nothing(tmp_path):
    """The quit path through the REAL engine, not just a direct agent call."""
    from pkm.data import Deck

    html = tmp_path / "result.html"
    replay = tmp_path / "replay.json"
    deck = Deck.from_csv("deck/02_dragapult.csv").card_ids
    session = ThreadedEnvSession(
        deck=deck,
        human_index=0,
        opponent="random",
        html_path=str(html),
        replay_path=str(replay),
    )
    session.start()

    for _ in range(3):  # play a few turns, then walk away
        event = session.next_event(timeout=60)
        assert isinstance(event, Prompt)
        session.submit([0])

    session.next_event(timeout=60)
    session.quit()
    session._thread.join(timeout=30)

    assert not session._thread.is_alive(), "quit must unwind env.run"
    assert not html.exists(), "an abandoned game must not write a replay"
    assert not replay.exists()


def test_worker_failure_surfaces_as_a_failed_event():
    session = ThreadedEnvSession(deck=[1] * 60, human_index=0, opponent="random")

    def boom():
        raise RuntimeError("engine exploded")

    session._run_env = boom  # simulate the engine dying
    session.start()
    event = session.next_event(timeout=5)

    assert isinstance(event, Failed)
    assert "engine exploded" in str(event.error)


@pytest.mark.slow
def test_full_game_against_random_through_the_real_engine():
    """Drive a real game with a scripted 'always pick the first options' human.

    This is the only test that can catch a genuine deadlock between the Textual
    thread and env.run.
    """
    from pkm.data import Deck

    deck = Deck.from_csv("deck/02_dragapult.csv").card_ids
    session = ThreadedEnvSession(
        deck=deck,
        human_index=0,
        opponent="random",
        html_path=None,
        replay_path=None,
    )
    session.start()

    prompts = 0
    while True:
        event = session.next_event(timeout=60)
        if isinstance(event, Finished):
            assert event.rewards[0] in (-1, 0, 1)
            break
        if isinstance(event, Failed):
            raise AssertionError(f"session failed: {event.error}")
        assert isinstance(event, Prompt)
        prompts += 1
        select = event.obs.select
        session.submit(list(range(select.minCount or 1))[: select.maxCount] or [0])

    assert prompts > 5, "a real game should ask the human more than a few questions"
    assert HUMAN == "human"


def test_win_rate_rejects_human():
    from pkm.rl.play import win_rate

    with pytest.raises(ValueError, match="human"):
        win_rate("human", "random", games=5)


def test_make_agent_by_name_rejects_human():
    # human needs a session to talk to; it cannot be built as a plain agent.
    from pkm.rl.play import make_agent_by_name

    with pytest.raises(ValueError, match="human"):
        make_agent_by_name("human", [1] * 60, None)
