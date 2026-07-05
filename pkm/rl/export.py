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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to .pt state_dict")
    parser.add_argument("out", help="output .npz path (e.g. pkm/policy.npz)")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.out)
    print(f"exported {args.checkpoint} -> {args.out}")
