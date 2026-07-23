"""Kaggle submission entry point for agent_001_transformer.

Packed to ``main.py`` by :mod:`.pack`, alongside ``weights.pth`` (a
``{"state_dict", "dims"}`` checkpoint) and the ``pkm/`` package. Exposes a plain
``agent(obs) -> list[int]`` callable (a plain function, so kaggle's
``co_argcount`` check sees exactly one argument).

Inference loads the torch ``MyModel`` and runs the notebook's ``mcts_agent``
against Kaggle's own libcg via this repo's engine seam (``cabt``). Torch is NOT
bundled (size limit) — it relies on torch being present in the cabt sandbox,
same as agent_000.
"""

import os
from pathlib import Path

import torch

from pkm.new_agents.agent_001_transformer import net

_KAGGLE_DIR = Path("/kaggle_simulations/agent")
# kaggle runs this module via exec(), which does NOT define __file__.
_file = globals().get("__file__")
_DIR = Path(_file).resolve().parent if _file else _KAGGLE_DIR

# Inference search budget (kept modest for the per-move time limit; override via env).
_SIMS = int(os.environ.get("SEARCH_COUNT", "10"))


def _weights_path() -> Path | None:
    for candidate in (_DIR / "weights.pth", _KAGGLE_DIR / "weights.pth"):
        if candidate.is_file():
            return candidate
    return None


def _load_model() -> net.MyModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = _weights_path()
    if path is None:
        # untrained fallback (should not happen in a real bundle)
        return net.build_model(net.MODEL_DIMS, device)
    blob = torch.load(path, map_location=device, weights_only=False)
    dims = tuple(blob.get("dims", net.MODEL_DIMS))
    model = net.MyModel(*dims).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


_model = _load_model()
_deck = net.sample_deck


def agent(obs: dict) -> list[int]:
    # Deck-selection phase: no board / no choice list yet -> submit the deck.
    if obs.get("select") is None or obs.get("current") is None:
        return list(_deck)
    with torch.inference_mode():
        selected, _ = net.mcts_agent(obs, _deck, _model, _SIMS)
    return selected
