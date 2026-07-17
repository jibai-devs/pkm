"""`pkm new_agents ...` — CLI group for the standalone next-gen agents.

Each agent under :mod:`pkm.new_agents` owns its full training stack and exposes
its own Typer app; they are mounted here so they surface as::

    pkm new_agents 000_dragapult smoke
    pkm new_agents 000_dragapult train --updates 200 --games 32 --workers 8
    pkm new_agents 000_dragapult eval

Importing this module is deliberately torch-free (the agent apps defer their
heavy imports into command bodies), so mounting it on the main `pkm` CLI does not
slow unrelated commands like `pkm deck list`.
"""

from __future__ import annotations

import typer

from pkm.new_agents.agent_000_dragapult.cli import app as dragapult_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Standalone next-generation agents (self-contained training stacks).",
)

app.add_typer(
    dragapult_app, name="000_dragapult", help="Dragapult ex specialist (PPO self-play)."
)


if __name__ == "__main__":
    app()
