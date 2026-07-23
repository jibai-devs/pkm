"""The ``--deck`` runtime routing: the played 60-card list follows the choice,
while the (superset) vocabulary/network shape does not."""

from __future__ import annotations

import dataclasses

from pkm.new_agents.agent_000_dragapult import deck
from pkm.new_agents.agent_000_dragapult.agent import DragapultAgent
from pkm.new_agents.agent_000_dragapult.config import Config, RunConfig


def test_agent_submits_the_selected_deck() -> None:
    """At the deck-selection phase (select/current None) the agent returns its
    own deck's 60 IDs — not always the default."""
    sel_phase = {"select": None, "current": None}
    for name in deck.DECKS:
        agent = DragapultAgent(deck=name)
        assert agent(sel_phase) == deck.deck_60(name)


def test_default_deck_is_dragapult() -> None:
    assert DragapultAgent()({"select": None, "current": None}) == deck.deck_60("dragapult")
    assert RunConfig().deck == "dragapult"


def test_deck_is_part_of_config_identity() -> None:
    base = Config()
    other = dataclasses.replace(
        base, run=dataclasses.replace(base.run, deck="alakazam")
    )
    assert base.run.deck == "dragapult"
    assert base.hash() != other.hash(), "deck must be folded into the config hash"


def test_old_checkpoint_without_deck_backfills_default() -> None:
    d = Config().to_dict()
    del d["run"]["deck"]  # simulate a pre-deck checkpoint
    assert Config.from_dict(d).run.deck == "dragapult"
