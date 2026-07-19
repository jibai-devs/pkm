"""Torch-free forward pass for ArchetypeClassifier.

Mirrors pkm/rl/numpy_policy.py's pattern (and reuses its module-level
_relu/_linear helpers directly). Must stay in sync with
pkm/archetype/model.py. Used by the Kaggle submission agent so the bundle
doesn't need torch for this classifier either.
"""

import numpy as np

from pkm.rl.numpy_policy import _linear, _relu

_STAMP_KEY = "__num_archetypes__"


class NumpyArchetypeClassifier:
    def __init__(self, weights: dict[str, np.ndarray], num_archetypes: int):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.num_archetypes = num_archetypes

    @classmethod
    def load(cls, path: str) -> "NumpyArchetypeClassifier":
        from pkm.archetype.archetypes import get_archetypes

        with np.load(path) as z:
            if _STAMP_KEY not in z.files:
                raise ValueError(f"{path}: missing {_STAMP_KEY} stamp, cannot verify class count")
            stamped_num = int(z[_STAMP_KEY])
            current_num = len(get_archetypes())
            if stamped_num != current_num:
                raise ValueError(
                    f"{path}: exported for {stamped_num} archetypes, but "
                    f"staples.json/aliases.py now resolve {current_num} -- "
                    "retrain and re-export before using this checkpoint"
                )
            weights = {k: z[k] for k in z.files if k != _STAMP_KEY}
            return cls(weights, num_archetypes=stamped_num)

    def belief(self, card_ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """(K,) card_ids + (K,) counts -- a sparse bag of currently-visible
        opponent cards -- -> (num_archetypes + 1,) softmax belief."""
        w = self.w
        card_emb = w["card_emb.weight"]
        if len(card_ids) == 0:
            x = np.zeros(card_emb.shape[1], dtype=np.float32)
        else:
            e = card_emb[card_ids]
            x = (e * counts[:, None]).sum(0)
        h = _relu(_linear(w["fc1.weight"], w["fc1.bias"], x))
        h = _relu(_linear(w["fc2.weight"], w["fc2.bias"], h))
        logits = _linear(w["fc3.weight"], w["fc3.bias"], h)
        logits = logits - logits.max()
        p = np.exp(logits)
        return p / p.sum()
