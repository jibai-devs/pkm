"""Standalone opponent-archetype classifier network.

Deliberately NOT an auxiliary head on PolicyValueNet's shared trunk -- see
plan.md §8.4's 2026-07-19 note for why. Its own small card embedding table
(NOT pkm.rl.model.PolicyValueNet.card_emb), own training loop, own export --
fully decoupled so this network's architecture/class-count can change
without forcing a retrain of the policy/value trunk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pkm.types.obs import NUM_CARDS

EMB_CARD = 16
HIDDEN = 64


class ArchetypeClassifier(nn.Module):
    """Bag-of-cards embedding + count-weighted pooling -> MLP -> softmax
    over tracked archetypes + one reserved "Unknown" class.

    Input is a sparse bag (card_ids, counts) of whatever's currently visible
    of the opponent's deck -- discard + board (+ attached energy/tools) +
    revealed prizes. Never `hand`: pkm/types/obs.py's Player contract always
    reports the opponent's hand as None.
    """

    def __init__(
        self, num_archetypes: int, emb_card: int = EMB_CARD, hidden: int = HIDDEN
    ) -> None:
        super().__init__()
        self.num_archetypes = num_archetypes
        self.out_dim = num_archetypes + 1  # +1 for "Unknown"
        self.card_emb = nn.Embedding(NUM_CARDS, emb_card)
        self.fc1 = nn.Linear(emb_card, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, self.out_dim)

    def pool(self, card_ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """(B, K) ids + (B, K) counts -> (B, emb_card), count-weighted sum.
        Permutation-invariant over which slot holds which card -- same idea
        as PolicyValueNet._pool_deck, but this is its own embedding table."""
        e = self.card_emb(card_ids)
        return (e * counts.unsqueeze(-1)).sum(1)

    def logits(self, card_ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        x = self.pool(card_ids, counts)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def forward(self, card_ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(card_ids, counts), dim=-1)
