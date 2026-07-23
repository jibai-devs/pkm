"""compute_belief(obs, classifier) -> np.ndarray

Builds the sparse bag of currently-visible opponent cards -- discard +
board (+ attachments) + revealed prizes, never hand (always None for the
opponent per pkm/types/obs.py's Player contract) -- and runs it through a
classifier's belief() method.

Reuses pkm.mcts.determinize._visible_counter for the "what's visible" logic
instead of re-deriving visibility rules in a second place.
"""

import numpy as np

from pkm.mcts.determinize import _visible_counter


def compute_belief(obs: dict, classifier) -> np.ndarray:
    """obs: raw observation dict (the pre-pydantic-validation seam, same
    convention pkm.mcts uses). classifier: anything exposing
    `.belief(card_ids, counts)` (pkm.archetype.numpy_model.NumpyArchetypeClassifier
    at inference)."""
    state = obs["current"]
    you = state["yourIndex"]
    visible = _visible_counter(state, 1 - you, include_hand=False)
    if not visible:
        card_ids = np.zeros(0, dtype=np.int64)
        counts = np.zeros(0, dtype=np.float32)
    else:
        card_ids = np.array(list(visible.keys()), dtype=np.int64)
        counts = np.array(list(visible.values()), dtype=np.float32)
    return classifier.belief(card_ids, counts)
