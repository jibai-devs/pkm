"""Dumb, stateless-ish widgets. They take models and render text."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Label, RichLog, Static

from pkm.obs import Observation, Player, PokemonRef
from pkm.tui.labels import card_name, energy_cost, option_label

HP_BAR_WIDTH = 16


def _hp_bar(pokemon: PokemonRef) -> str:
    if pokemon.maxHp <= 0:
        return ""
    filled = max(0, round(HP_BAR_WIDTH * pokemon.hp / pokemon.maxHp))
    return "█" * filled + "░" * (HP_BAR_WIDTH - filled)


def _pokemon_line(pokemon: PokemonRef | None, prefix: str) -> str:
    if pokemon is None:
        return f"{prefix} —"
    tools = ", ".join(card_name(t.id) for t in pokemon.tools)
    bits = [
        f"{prefix} {card_name(pokemon.id)}",
        f"{pokemon.hp}/{pokemon.maxHp}",
        _hp_bar(pokemon),
        energy_cost(pokemon.energies),
    ]
    if tools:
        bits.append(f"tool: {tools}")
    return "  ".join(bits)


class BoardPanel(Static):
    """One player's side of the board."""

    def __init__(self, title: str, **kwargs) -> None:
        # Rendered text (energy costs, tool lists) contains literal "[" / "]"
        # (e.g. "[P][P]"), which Rich markup would otherwise try to parse as
        # style tags and silently eat. This content is plain text, never markup.
        kwargs.setdefault("markup", False)
        super().__init__(**kwargs)
        self.border_title = title

    def show(self, player: Player) -> None:
        lines = [_pokemon_line(player.active_pokemon, "ACTIVE")]
        bench = [p for p in player.bench if p is not None]
        if bench:
            lines += [_pokemon_line(p, f"BENCH {i + 1}") for i, p in enumerate(bench)]
        else:
            lines.append("BENCH  —")
        # Prize entries are always None (face-down — you don't know your own
        # prizes either); the list shrinks as prizes are taken, so its length is
        # the count. Same convention as encoder.prize_potential.
        prizes_left = len(player.prize)
        lines.append(
            f"prizes {prizes_left}  deck {player.deckCount}  "
            f"hand {player.handCount}  discard {len(player.discard)}"
        )
        conditions = player.conditions
        if conditions:
            lines.append("status: " + ", ".join(conditions))
        self.update("\n".join(lines))


class HandBar(Static):
    """The human's hand, as names."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("markup", False)
        super().__init__(**kwargs)

    def show(self, player: Player) -> None:
        if not player.hand:
            self.update("hand: (empty)")
            return
        self.update(" · ".join(card_name(c.id) for c in player.hand))


class EventLog(RichLog):
    """Scrolling feed of translated log entries."""

    def add(self, line: str) -> None:
        self.write(line)


class PromptPane(Static):
    """The option list. Multi-select via toggling; Enter submits.

    Picks are indices into the *current* select.option list and are cleared on
    every new prompt — the engine has no rollback, so there is no cross-prompt
    state to keep.
    """

    picks: reactive[list[int]] = reactive(list, always_update=True)

    def __init__(self, **kwargs) -> None:
        # Option labels contain literal "[" / "]" (e.g. an attack's energy cost
        # "[P][P]") and the pick marker itself is rendered as "[x]" / "[ ]" —
        # Rich markup would parse these as style tags and eat the text. Plain
        # text only, never markup.
        kwargs.setdefault("markup", False)
        super().__init__(**kwargs)
        self.obs: Observation | None = None

    def show(self, obs: Observation) -> None:
        self.obs = obs
        self.picks = []
        self._redraw()

    def toggle(self, index: int) -> None:
        if self.obs is None or self.obs.select is None:
            return
        if not 0 <= index < len(self.obs.select.option):
            return
        picks = list(self.picks)
        if index in picks:
            picks.remove(index)
        elif len(picks) < self.obs.select.maxCount:
            picks.append(index)
        self.picks = picks
        self._redraw()

    def is_submittable(self) -> bool:
        if self.obs is None or self.obs.select is None:
            return False
        return self.obs.select.minCount <= len(self.picks) <= self.obs.select.maxCount

    def hint(self) -> str:
        if self.obs is None or self.obs.select is None:
            return ""
        select = self.obs.select
        missing = select.minCount - len(self.picks)
        if missing > 0:
            return f"pick {missing} more"
        if len(self.picks) > select.maxCount:
            return f"pick at most {select.maxCount}"
        return "Enter to confirm"

    def _redraw(self) -> None:
        # NB: this must not be named `_render` — that shadows Widget._render(),
        # Textual's internal hook for producing the layout Visual, and this
        # method returns None (its job is only the self.update() side effect).
        # Textual's layout engine then crashes trying to call .get_height() on
        # None when it recomputes content height (e.g. on toggle/resize).
        if self.obs is None or self.obs.select is None:
            self.update("waiting for the agent…")
            return
        select = self.obs.select
        span = (
            f"pick {select.minCount}"
            if select.minCount == select.maxCount
            else f"pick {select.minCount}-{select.maxCount}"
        )
        self.border_title = f"CHOOSE ({span}) — {self.hint()}"
        lines = []
        for i, option in enumerate(select.option):
            mark = "x" if i in self.picks else " "
            key = str(i + 1) if i < 9 else " "
            lines.append(f"[{mark}] {key}. {option_label(self.obs, option)}")
        self.update("\n".join(lines))


class ConfirmScreen(ModalScreen[bool]):
    """Confirm an irreversible choice (attack / end turn). There is no undo."""

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(self.question)
            yield Button("Confirm", variant="primary", id="yes")
            yield Button("Cancel", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")
