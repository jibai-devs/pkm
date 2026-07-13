"""BattleApp — the human's screen.

Never call print() from this module or anything it touches: kaggle wraps the
agent call in redirect_stdout, which is active process-wide while the human is
thinking, so prints vanish into its capture buffer. Use textual.log.
"""

import queue

from textual import log
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header
from textual.worker import get_current_worker

from pkm.obs import Observation, OptionType
from pkm.tui.labels import log_label
from pkm.tui.session import Event, Failed, Finished, GameSession, Prompt
from pkm.tui.widgets import BoardPanel, ConfirmScreen, EventLog, HandBar, PromptPane

# How often the pump wakes up to check whether the app is shutting down.
# Worker.cancel() only cancels the asyncio Task wrapping the thread call — it
# cannot interrupt a thread already blocked inside next_event(). Without this
# poll, a session that runs dry (no more events, no Finished/Failed) leaves the
# pump thread parked forever, and asyncio's executor shutdown then blocks for
# up to THREAD_JOIN_TIMEOUT (300s) waiting for it to return.
PUMP_POLL_SECONDS = 0.1

IRREVERSIBLE = {OptionType.ATTACK, OptionType.END}


class BattleApp(App[None]):
    """Human vs agent. One prompt pane drives every decision."""

    CSS = """
    Screen { layout: vertical; }
    #board { height: 1fr; }
    #panels { width: 2fr; }
    BoardPanel { border: round $accent; padding: 0 1; height: 1fr; }
    EventLog { border: round $secondary; width: 1fr; }
    HandBar { border: round $accent; padding: 0 1; height: auto; }
    PromptPane { border: round $success; padding: 0 1; height: auto; min-height: 6; }
    #confirm-box { align: center middle; width: 50; height: auto;
                   border: thick $warning; background: $surface; padding: 1 2; }
    """

    BINDINGS = [
        ("q", "quit_game", "Quit"),
        ("enter", "submit", "Confirm"),
        *[(str(n), f"toggle({n - 1})", "") for n in range(1, 10)],
    ]

    def __init__(self, session: GameSession, confirm_irreversible: bool = True) -> None:
        super().__init__()
        self.session = session
        self.confirm_irreversible = confirm_irreversible
        self.result_text: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="board"):
            with Vertical(id="panels"):
                yield BoardPanel("AGENT", id="opponent")
                yield BoardPanel("YOU", id="me")
            yield EventLog(id="events", markup=False)
        yield HandBar(id="hand")
        yield PromptPane(id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#hand", HandBar).border_title = "YOUR HAND"
        self.session.start()
        self.run_worker(self._pump, thread=True, exclusive=True)

    # -- worker pump: blocks on the session, hands events to the UI thread --

    def _pump(self) -> None:
        worker = get_current_worker()
        while not worker.is_cancelled:
            try:
                event = self.session.next_event(timeout=PUMP_POLL_SECONDS)
            except queue.Empty:
                continue
            self.call_from_thread(self._handle, event)
            if isinstance(event, (Finished, Failed)):
                return

    def _handle(self, event: Event) -> None:
        if isinstance(event, Prompt):
            self._show(event.obs)
        elif isinstance(event, Finished):
            self._finish(event)
        elif isinstance(event, Failed):
            log.error(f"session failed: {event.error}")
            self.result_text = f"Error: {event.error}"
            self.query_one("#prompt", PromptPane).update(self.result_text)

    def _show(self, obs: Observation) -> None:
        self.query_one("#me", BoardPanel).show(obs.me)
        self.query_one("#opponent", BoardPanel).show(obs.opponent)
        self.query_one("#hand", HandBar).show(obs.me)

        # obs.logs is already a delta: the engine sends only what happened since
        # the last observation. Write all of it, every time.
        events = self.query_one("#events", EventLog)
        for entry in obs.logs:
            events.add(log_label(obs, entry))

        self.query_one("#prompt", PromptPane).show(obs)
        self.sub_title = f"turn {obs.current.turn}"

    def _finish(self, event: Finished) -> None:
        mine = event.rewards[self.session.human_index]
        if mine is None:
            self.result_text = "Game over (no result)"
        elif mine > 0:
            self.result_text = "You win!"
        elif mine < 0:
            self.result_text = "You lose."
        else:
            self.result_text = "Draw."
        artifacts = " · ".join(p for p in (event.html_path, event.replay_path) if p)
        suffix = f"\nwrote {artifacts}" if artifacts else ""
        self.query_one("#prompt", PromptPane).update(
            f"{self.result_text}{suffix}\n\npress q to exit"
        )

    # -- actions -----------------------------------------------------------

    def action_toggle(self, index: int) -> None:
        self.query_one("#prompt", PromptPane).toggle(index)

    def action_submit(self) -> None:
        prompt = self.query_one("#prompt", PromptPane)
        if not prompt.is_submittable():
            self.bell()
            return
        picks = list(prompt.picks)

        # NB: push_screen_wait() would raise NoActiveWorker here — it may only be
        # awaited inside a worker. Use the callback form instead.
        if self.confirm_irreversible and self._is_irreversible(prompt):
            self.push_screen(
                ConfirmScreen("This can't be undone. Confirm?"),
                lambda confirmed: self._send(picks) if confirmed else None,
            )
            return
        self._send(picks)

    def _send(self, picks: list[int]) -> None:
        self.query_one("#prompt", PromptPane).update("waiting for the agent…")
        self.session.submit(picks)

    def _is_irreversible(self, prompt: PromptPane) -> bool:
        if prompt.obs is None or prompt.obs.select is None:
            return False
        options = prompt.obs.select.option
        return any(options[i].kind in IRREVERSIBLE for i in prompt.picks)

    def action_quit_game(self) -> None:
        self.session.quit()
        self.exit()
