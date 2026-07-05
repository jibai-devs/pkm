"""Export a trained PolicyValueNet to .npz for torch-free inference.

The Kaggle submission bundle has a ~197 MiB cap, so we don't ship torch;
pkm/rl/numpy_policy.py replays the same forward pass in numpy.

Usage:
    python -m pkm.rl.export checkpoints/ppo_latest.pt pkm/policy.npz
    python -m pkm.rl.export --agent 01_psychic pkm/policy.npz
"""

import numpy as np
import torch

from .model import PolicyValueNet


def export_npz(model: PolicyValueNet, path: str) -> None:
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(path, **arrays)


def export_checkpoint(checkpoint_path: str, out_path: str) -> None:
    model = PolicyValueNet()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    export_npz(model, out_path)


if __name__ == "__main__":
    import typer

    from pkm.agents.profile import AgentProfile

    app = typer.Typer(help=__doc__)

    @app.command()
    def main(
        checkpoint: str = typer.Argument("", help="path to .pt state_dict (omit with --agent)"),
        out: str = typer.Argument("pkm/policy.npz", help="output .npz path"),
        agent: str | None = typer.Option(None, help="agent profile name"),
    ) -> None:
        if agent:
            profile = AgentProfile(agent)
            ckpt = profile.checkpoint_dir / "ppo_latest.pt"
            if not ckpt.is_file():
                ckpt = profile.checkpoint_dir / "ppo_iter0200.pt"
            checkpoint = str(ckpt)
        if not checkpoint:
            raise typer.BadParameter("provide a checkpoint path or --agent")
        export_checkpoint(checkpoint, out)
        typer.echo(f"exported {checkpoint} -> {out}")

    app()
