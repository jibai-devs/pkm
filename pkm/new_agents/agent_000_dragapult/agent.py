"""Inference agent for agent_000_dragapult.

The competition contract is a callable ``agent(obs_dict) -> list[int]`` returning
chosen option **indices** into ``obs["select"]["option"]``. This wraps the
`PolicyValueModel`:

    obs dict -> to_observation -> featurize -> collate([f]) -> model -> pick

Two special cases handled here:
  * **deck-selection phase** (`select`/`current` are ``None``): return the deck.
  * **selection count** (`minCount..maxCount`): the model emits per-option logits;
    this layer decides how many to pick (multi-select was deferred in the model,
    see README D4). Provisional rule: pick ``clamp(maxCount, minCount, n)`` by
    top-k (greedy) or sampling without replacement (stochastic).
"""

from __future__ import annotations

from typing import Any

import torch

from pkm.new_agents.agent_000_dragapult.cabt import to_observation
from pkm.new_agents.agent_000_dragapult.deck import DECK_60
from pkm.new_agents.agent_000_dragapult.features import featurize
from pkm.new_agents.agent_000_dragapult.model import PolicyValueModel, collate


class DragapultAgent:
    """Callable policy. Holds a (possibly trained) model; frozen at inference."""

    def __init__(
        self,
        model: PolicyValueModel | None = None,
        greedy: bool = False,
        seed: int | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = (model or PolicyValueModel()).to(self.device).eval()
        self.greedy = greedy
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator(device=self.device).manual_seed(seed)

    @classmethod
    def from_checkpoint(cls, path: str, **kw: Any) -> "DragapultAgent":
        # Always load on CPU — inference (Kaggle) runs on CPU regardless of the
        # device the weights were trained on.
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            # Bundle format: weights + the architecture they were trained with,
            # so a non-default (e.g. large / deeper) net rebuilds correctly.
            from pkm.new_agents.agent_000_dragapult.config import (
                ModelConfig,
                build_model,
            )

            mc = blob.get("model_config")
            model = build_model(ModelConfig(**mc)) if mc else PolicyValueModel()
            model.load_state_dict(blob["state_dict"])
        else:
            # Legacy format: a bare state_dict (default/small architecture).
            model = PolicyValueModel()
            model.load_state_dict(blob)
        return cls(model=model, **kw)

    @torch.no_grad()
    def __call__(self, obs_dict: dict[str, Any]) -> list[int]:
        # Deck-selection phase: no board / no choice list yet -> submit the deck.
        if obs_dict.get("select") is None or obs_dict.get("current") is None:
            return list(DECK_60)

        feats = featurize(to_observation(obs_dict))
        n = feats.n_options
        if n == 0:
            return []
        k = max(feats.min_count, min(feats.max_count, n))
        if k <= 0:
            return []

        batch = collate([feats])
        logits, _value = self.model(batch)
        probs = torch.softmax(logits[0, :n], dim=-1)  # only the n real options

        if self.greedy:
            idx = torch.topk(probs, k).indices
        else:
            idx = torch.multinomial(probs, k, replacement=False, generator=self.gen)
        return idx.tolist()


# Module-level convenience so `from ...agent import agent` plugs into the
# kaggle_environments harness (make("cabt").run([agent, agent])).
_default_agent: DragapultAgent | None = None


def agent(obs_dict: dict[str, Any]) -> list[int]:
    global _default_agent
    if _default_agent is None:
        _default_agent = DragapultAgent()
    return _default_agent(obs_dict)
