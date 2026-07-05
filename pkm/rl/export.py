"""Export a trained PolicyValueNet to .npz for torch-free inference.

The Kaggle submission bundle has a ~197 MiB cap, so we don't ship torch;
pkm/rl/numpy_policy.py replays the same forward pass in numpy.
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

    app = typer.Typer(help=__doc__)

    @app.command()
    def main(
        checkpoint: str = typer.Argument(help="path to .pt state_dict"),
        out: str = typer.Argument(help="output .npz path (e.g. pkm/policy.npz)"),
    ) -> None:
        export_checkpoint(checkpoint, out)
        typer.echo(f"exported {checkpoint} -> {out}")

    app()
