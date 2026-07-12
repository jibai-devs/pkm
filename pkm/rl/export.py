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
) -> None:
    from pkm.agents.profile import AgentProfile

    profile = AgentProfile(agent) if agent else None
    if profile is not None:
        ckpt = profile.checkpoint_dir / "ppo_latest.pt"
        if not ckpt.is_file():
            ckpt = profile.checkpoint_dir / "ppo_iter0200.pt"
        checkpoint = str(ckpt)
    if not checkpoint:
        raise typer.BadParameter("provide a checkpoint path or --agent")
    output = out or str(
        profile.exported_weights_path if profile else Path("pkm/policy.npz")
    )
    export_checkpoint(checkpoint, output)
    typer.echo(f"exported {checkpoint} -> {output}")


if __name__ == "__main__":
    app()
