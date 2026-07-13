"""Export a trained PolicyValueNet to .npz for torch-free inference.

The Kaggle submission bundle has a ~197 MiB cap, so we don't ship torch;
pkm/rl/numpy_policy.py replays the same forward pass in numpy.

Usage:
    pkm export checkpoints/ppo_latest.pt pkm/policy.npz
    pkm export --agent 01_psychic pkm/policy.npz
"""

from pathlib import Path

import numpy as np
import torch
import typer

from .model import PolicyValueNet

app = typer.Typer(help=__doc__)


def export_npz(model: PolicyValueNet, path: str) -> None:
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(path, **arrays)


def export_checkpoint(checkpoint_path: str, out_path: str) -> None:
    model = PolicyValueNet()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    export_npz(model, out_path)


@app.command()
def main(
    checkpoint: str = typer.Argument(
        "", help="path to .pt state_dict (omit with --agent)"
    ),
    out: str | None = typer.Argument(None, help="output .npz path"),
    agent: str | None = typer.Option(None, help="agent profile name"),
    phase: str = typer.Option(
        "ppo", help="which profile checkpoint to export: ppo or exit"
    ),
) -> None:
    from pkm.agents.profile import AgentProfile

    profile = AgentProfile(agent) if agent else None
    if profile is not None and not checkpoint:
        # An explicit checkpoint always wins; otherwise pick the requested phase.
        if phase not in ("ppo", "exit"):
            raise typer.BadParameter(f"unknown phase {phase!r}; expected ppo or exit")
        selected = profile.latest_checkpoint(phase)
        if selected is None:
            raise typer.BadParameter(
                f"agent {profile.name!r} has no {phase} checkpoint to export"
            )
        checkpoint = str(selected)
    if not checkpoint:
        raise typer.BadParameter("provide a checkpoint path or --agent")
    output = out or str(
        profile.exported_weights_path if profile else Path("pkm/policy.npz")
    )
    export_checkpoint(checkpoint, output)
    typer.echo(f"exported {checkpoint} -> {output}")


if __name__ == "__main__":
    app()
