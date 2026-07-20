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

**Inference-time MCTS (optional).** When configured with an
:class:`InferenceConfig` of type ``"mcts"`` and a positive simulation budget
(``mcts_sims`` — the "K"), the acting decision is chosen by PUCT search
(`mcts.search`) guided by the net's priors + value, instead of the raw policy
head. ``mcts_sims == 0`` (or type ``"policy"``) disables search entirely and
falls back to the plain policy pick — so the two are one packed bundle apart,
and the deployment cost (search runs per decision, against Kaggle's own
``libcg.so`` search symbols) is opt-in. See `mcts.py` and `docs/ENGINE.md`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pkm.new_agents.agent_000_dragapult.cabt import to_observation
from pkm.new_agents.agent_000_dragapult.deck import DEFAULT_DECK, deck_60
from pkm.new_agents.agent_000_dragapult.features import featurize
from pkm.new_agents.agent_000_dragapult.model import PolicyValueModel, collate


@dataclass(frozen=True)
class InferenceConfig:
    """How the agent turns a state into a move at inference time.

    ``type == "mcts"`` with ``mcts_sims > 0`` enables PUCT search; anything else
    (``type == "policy"`` or ``mcts_sims == 0``) is the raw policy-head pick.
    This is a *deployment* choice, embedded in the packed bundle (`weights.pt`),
    so the same checkpoint can be submitted with or without search.
    """

    type: str = "policy"  # "policy" | "mcts"
    mcts_sims: int = 0  # the "K" search budget per decision; 0 disables MCTS
    mcts_worlds: int = 1  # IS-MCTS determinizations to average per decision (W)
    c_puct: float = 1.25
    temperature: float = 0.0  # 0 => pick the most-visited move (deterministic)
    determinization: str = "sample"  # key into mcts/determinize.DETERMINIZERS

    @property
    def use_mcts(self) -> bool:
        return self.type == "mcts" and self.mcts_sims > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InferenceConfig":
        # Backfill unknown/missing fields with defaults so old bundles load.
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**fields)


class DragapultAgent:
    """Callable policy. Holds a (possibly trained) model; frozen at inference."""

    def __init__(
        self,
        model: PolicyValueModel | None = None,
        greedy: bool = False,
        seed: int | None = None,
        device: str = "cpu",
        inference: InferenceConfig | None = None,
        deck: str = DEFAULT_DECK,
    ):
        self.device = torch.device(device)
        self.model = (model or PolicyValueModel()).to(self.device).eval()
        self.greedy = greedy
        self.inference = inference or InferenceConfig()
        # The 60-card list this agent submits at the deck-selection phase.
        self.deck_ids = deck_60(deck)
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator(device=self.device).manual_seed(seed)
        # Lightweight cfg view that `mcts.search` reads (only cfg.train.mcts_*
        # + cfg.train.determinization) — avoids building a full training Config.
        self._search_cfg = None
        if self.inference.use_mcts:
            from types import SimpleNamespace

            self._search_cfg = SimpleNamespace(
                train=SimpleNamespace(
                    mcts_simulations=self.inference.mcts_sims,
                    mcts_c_puct=self.inference.c_puct,
                    mcts_temperature=self.inference.temperature,
                    determinization=self.inference.determinization,
                ),
                # So mcts.search determinizes hidden zones from the played deck.
                run=SimpleNamespace(deck=deck),
            )

    @classmethod
    def from_checkpoint(
        cls, path: str, inference: InferenceConfig | None = None, **kw: Any
    ) -> "DragapultAgent":
        # Always load on CPU — inference (Kaggle) runs on CPU regardless of the
        # device the weights were trained on.
        blob = torch.load(path, map_location="cpu", weights_only=False)
        bundle_inference: InferenceConfig | None = None
        # Default the played deck from the checkpoint's own config unless the
        # caller overrides it (a bundle may record its deck at pack time).
        if isinstance(blob, dict) and "deck" not in kw:
            _cfg = blob.get("config")
            _deck = (
                _cfg.get("run", {}).get("deck") if isinstance(_cfg, dict) else None
            ) or blob.get("deck")
            if _deck:
                kw["deck"] = _deck
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
            # A packed bundle may also carry the deployment inference config
            # (policy vs mcts + the K budget) chosen at pack time.
            inf = blob.get("inference")
            if inf:
                bundle_inference = InferenceConfig.from_dict(inf)
        elif isinstance(blob, dict) and "model" in blob and "config" in blob:
            # Training checkpoint (TrainState blob): rebuild from its stored config
            # so any architecture loads. Lets eval point straight at a ckpt_N.pt.
            from pkm.new_agents.agent_000_dragapult.config import Config, build_model

            model = build_model(Config.from_dict(blob["config"]))
            model.load_state_dict(blob["model"])
        else:
            # Legacy format: a bare state_dict (default/small architecture).
            model = PolicyValueModel()
            model.load_state_dict(blob)
        # Explicit `inference=` (caller override) wins over the bundle's own.
        return cls(model=model, inference=inference or bundle_inference, **kw)

    @torch.no_grad()
    def __call__(self, obs_dict: dict[str, Any]) -> list[int]:
        # Deck-selection phase: no board / no choice list yet -> submit the deck.
        if obs_dict.get("select") is None or obs_dict.get("current") is None:
            return list(self.deck_ids)

        feats = featurize(to_observation(obs_dict))
        n = feats.n_options
        if n == 0:
            return []
        k = max(feats.min_count, min(feats.max_count, n))
        if k <= 0:
            return []

        if self.inference.use_mcts:
            return self._mcts_pick(obs_dict, n, k)

        batch = collate([feats])
        logits, _value = self.model(batch)
        probs = torch.softmax(logits[0, :n], dim=-1)  # only the n real options

        if self.greedy:
            idx = torch.topk(probs, k).indices
        else:
            idx = torch.multinomial(probs, k, replacement=False, generator=self.gen)
        return idx.tolist()

    def _mcts_pick(self, obs_dict: dict[str, Any], n: int, k: int) -> list[int]:
        """Choose ``k`` options from the MCTS root visit policy over ``n`` options."""
        from pkm.new_agents.agent_000_dragapult import mcts

        seat = obs_dict["current"]["yourIndex"]
        gen = self.gen or torch.Generator(device=self.device)
        pi = torch.from_numpy(
            mcts.search_worlds(
                obs_dict, seat, self.model, self._search_cfg, gen,
                n_worlds=self.inference.mcts_worlds,
            )
        )
        pi = pi[:n]
        if self.greedy:
            idx = torch.topk(pi, k).indices
        else:
            idx = torch.multinomial(pi, k, replacement=False, generator=self.gen)
        return idx.tolist()


# Module-level convenience so `from ...agent import agent` plugs into the
# kaggle_environments harness (make("cabt").run([agent, agent])).
_default_agent: DragapultAgent | None = None


def agent(obs_dict: dict[str, Any]) -> list[int]:
    global _default_agent
    if _default_agent is None:
        _default_agent = DragapultAgent()
    return _default_agent(obs_dict)
