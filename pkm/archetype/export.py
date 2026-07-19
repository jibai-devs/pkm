"""Export a trained ArchetypeClassifier to .npz for torch-free inference.

Mirrors pkm/rl/export.py's pattern. Stamps the exported class count so
pkm/archetype/numpy_model.py can refuse to silently load a checkpoint whose
archetype list has since changed (same "fail loudly on mismatch" principle
as plan.md §2's checkpoint/feature-set stamping, applied to this separate
model's own class-count invariant).
"""

import numpy as np
import torch

from pkm.archetype.model import ArchetypeClassifier

_STAMP_KEY = "__num_archetypes__"


def export_npz(model: ArchetypeClassifier, path: str) -> None:
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(path, **{_STAMP_KEY: np.array(model.num_archetypes)}, **arrays)


def export_checkpoint(checkpoint_path: str, num_archetypes: int, out_path: str) -> None:
    model = ArchetypeClassifier(num_archetypes=num_archetypes)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    export_npz(model, out_path)
