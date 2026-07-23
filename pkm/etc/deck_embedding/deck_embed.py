# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "numpy",
#     "polars",
# ]
# ///
"""Set-Transformer two-tower for deck embeddings + matchup prediction.

A deck is a *multiset* of cards: permutation-invariant, variable size, and its
meaning comes from card--card synergies. That is exactly what a Set Transformer
(Lee et al., ICML 2019) models -- self-attention over the cards captures
synergy, attention pooling (PMA) produces one deck vector. We wrap a shared deck
encoder in a two-tower matchup head that predicts P(A beats B) from real game
outcomes, so the learned embedding distance means "similar matchup behaviour",
not merely "similar card list".

This file is a **library**: a marimo notebook (or anything) imports the pieces::

    from deck_embed import (build_vocab, TwoTowerMatchup, DeckEncoder,
                            MatchupDataset, load_matchups_from_parquet,
                            train, embed_decks, nearest)

Run it standalone for a synthetic smoke test (no crawl data needed)::

    uv run --script deck_embed.py

Expected real input -- `deck_embedding/decks_with_outcomes.parquet`, long format,
one row per card per player per episode (produced by the `db -> decks` bridge,
HANDOFF #7.4)::

    episode_id: i64   player: i8   card_id: i64   count: i64   won: bool

`won` is that player's result in that episode; the two players of an episode form
one matchup. See `load_matchups_from_parquet`.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# A deck as handed around before tensorisation: (card_index, count) pairs, where
# card_index is already vocab-mapped (>=1; 0 is the PAD sentinel).
Deck = list[tuple[int, int]]

PAD_IDX = 0
MAX_COUNT = 32  # count embedding table size; real counts are clamped into [1, 31]


# --------------------------------------------------------------------------- #
# Set Transformer blocks (Lee et al. 2019). MAB uses nn.MultiheadAttention so
# key padding is handled correctly for variable-size decks.
# --------------------------------------------------------------------------- #

class MAB(nn.Module):
    """Multihead Attention Block: LN(Q + MHA(Q,K)) then LN(H + rFF(H))."""

    def __init__(self, dim: int, heads: int, ln: bool = True) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.ln0 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        a, _ = self.mha(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln0(q + a)
        return self.ln1(h + self.ff(h))


class SAB(nn.Module):
    """Set Attention Block: self-attention among set elements. MAB(X, X)."""

    def __init__(self, dim: int, heads: int, ln: bool = True) -> None:
        super().__init__()
        self.mab = MAB(dim, heads, ln)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.mab(x, x, key_padding_mask=key_padding_mask)


class ISAB(nn.Module):
    """Induced SAB: O(n*m) via m inducing points. Good for larger sets."""

    def __init__(self, dim: int, heads: int, m: int, ln: bool = True) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, m, dim))
        nn.init.xavier_uniform_(self.inducing)
        self.mab0 = MAB(dim, heads, ln)  # inducing points attend to the set
        self.mab1 = MAB(dim, heads, ln)  # set attends back to the summary

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.size(0)
        h = self.mab0(self.inducing.expand(b, -1, -1), x, key_padding_mask=key_padding_mask)
        return self.mab1(x, h)  # h has no padding


class PMA(nn.Module):
    """Pooling by Multihead Attention: k learned seeds attend to the set."""

    def __init__(self, dim: int, heads: int, k: int, ln: bool = True) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(1, k, dim))
        nn.init.xavier_uniform_(self.seeds)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.mab = MAB(dim, heads, ln)

    def forward(self, z: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = z.size(0)
        z = self.ff(z)
        return self.mab(self.seeds.expand(b, -1, -1), z, key_padding_mask=key_padding_mask)


# --------------------------------------------------------------------------- #
# Deck encoder: (card ids + counts) -> one embedding vector.
# --------------------------------------------------------------------------- #

class DeckEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64, heads: int = 4, m: int = 16,
                 n_blocks: int = 2, k_seeds: int = 1, emb_dim: int = 64) -> None:
        super().__init__()
        self.card_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD_IDX)
        self.count_emb = nn.Embedding(MAX_COUNT, dim, padding_idx=PAD_IDX)
        self.blocks = nn.ModuleList([ISAB(dim, heads, m) for _ in range(n_blocks)])
        self.pma = PMA(dim, heads, k_seeds)
        self.proj = nn.Linear(k_seeds * dim, emb_dim)

    def forward(self, card_idx: torch.Tensor, count: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # card_idx, count: (B, L) long; pad_mask: (B, L) bool, True where padded.
        x = self.card_emb(card_idx) + self.count_emb(count)
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)
        pooled = self.pma(x, key_padding_mask=pad_mask).flatten(1)  # (B, k*dim)
        return self.proj(pooled)


# --------------------------------------------------------------------------- #
# Two-tower matchup model. Antisymmetric logit => P(A>B) = 1 - P(B>A) by
# construction. W initialised to zero so training *starts* as a pure Bradley-
# Terry / Elo strength model and learns rock-paper-scissors interaction on top.
# --------------------------------------------------------------------------- #

class TwoTowerMatchup(nn.Module):
    def __init__(self, encoder: DeckEncoder, emb_dim: int = 64) -> None:
        super().__init__()
        self.encoder = encoder
        self.strength = nn.Linear(emb_dim, 1)
        self.interaction = nn.Parameter(torch.zeros(emb_dim, emb_dim))

    def encode(self, card_idx: torch.Tensor, count: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(card_idx, count, pad_mask)

    def matchup_logit(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        strength = self.strength(ea).squeeze(-1) - self.strength(eb).squeeze(-1)
        inter = (ea @ self.interaction * eb).sum(-1) - (eb @ self.interaction * ea).sum(-1)
        return strength + inter  # logit that A beats B

    def forward(self, a_idx, a_cnt, a_mask, b_idx, b_cnt, b_mask) -> torch.Tensor:
        ea = self.encode(a_idx, a_cnt, a_mask)
        eb = self.encode(b_idx, b_cnt, b_mask)
        return self.matchup_logit(ea, eb)


# --------------------------------------------------------------------------- #
# Vocab + tensorisation.
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class DeckVocab:
    idx2id: list[int]           # idx2id[0] == 0 is the PAD sentinel
    id2idx: dict[int, int]

    @property
    def size(self) -> int:
        return len(self.idx2id)


def build_vocab(card_ids: Iterable[int]) -> DeckVocab:
    """Map raw card_id -> contiguous index. Index 0 is reserved for PAD."""
    uniq = sorted({int(c) for c in card_ids})
    idx2id = [0, *uniq]                       # 0 sentinel; card ids in this game are >= 1
    return DeckVocab(idx2id=idx2id, id2idx={cid: i for i, cid in enumerate(idx2id)})


def deck_from_pairs(card_ids: Sequence[int], counts: Sequence[int], vocab: DeckVocab) -> Deck:
    return [(vocab.id2idx[int(c)], min(int(n), MAX_COUNT - 1)) for c, n in zip(card_ids, counts)]


def deck_to_tensors(deck: Deck) -> tuple[torch.Tensor, torch.Tensor]:
    """One deck -> (card_idx, count) 1-D long tensors, built once.

    Doing this per deck up front (in the dataset) instead of per element at
    batch time is what keeps the data pipeline off the Python hot path.
    """
    if deck:
        idx = torch.tensor([c for c, _ in deck], dtype=torch.long)
        cnt = torch.tensor([n for _, n in deck], dtype=torch.long)
    else:
        idx = cnt = torch.zeros(0, dtype=torch.long)
    return idx, cnt


# A tensorised matchup: (a_idx, a_cnt, b_idx, b_cnt, label).
TensorMatchup = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]


def collate(batch: list[TensorMatchup]) -> dict[str, torch.Tensor]:
    """Pad both decks of every matchup to the batch max — fully vectorised.

    ``pad_sequence`` pads in C over pre-built per-deck tensors, so there is no
    Python loop over cards. (The old collate did ~batch*cards scalar tensor
    assignments per batch, which starved the GPU.) Pad value 0 == ``PAD_IDX``
    and real card indices are >= 1, so the pad mask is exactly ``idx == PAD_IDX``.
    """
    a_idx = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=PAD_IDX)
    a_cnt = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=PAD_IDX)
    b_idx = pad_sequence([b[2] for b in batch], batch_first=True, padding_value=PAD_IDX)
    b_cnt = pad_sequence([b[3] for b in batch], batch_first=True, padding_value=PAD_IDX)
    y = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    return {"a_idx": a_idx, "a_cnt": a_cnt, "a_mask": a_idx == PAD_IDX,
            "b_idx": b_idx, "b_cnt": b_cnt, "b_mask": b_idx == PAD_IDX, "y": y}


class MatchupDataset(Dataset):
    """Pre-tensorises every matchup once so batching is only padding."""

    def __init__(self, matchups: list[tuple[Deck, Deck, int]]) -> None:
        self.items: list[TensorMatchup] = [
            (*deck_to_tensors(a), *deck_to_tensors(b), float(y)) for a, b, y in matchups
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> TensorMatchup:
        return self.items[i]


# --------------------------------------------------------------------------- #
# Data loading.
# --------------------------------------------------------------------------- #

def load_matchups_from_parquet(path: str) -> tuple[list[tuple[Deck, Deck, int]], dict[int, Deck], DeckVocab]:
    """Read decks_with_outcomes.parquet -> (matchups, decks_by_episode_player, vocab).

    Long format expected: episode_id, player, card_id, count, won. Each episode
    with exactly two players becomes one matchup, oriented (winner, loser) with
    label 1; we also add the mirror (loser, winner) with label 0 so the model
    sees both orientations.
    """
    import polars as pl

    df = pl.read_parquet(path)
    vocab = build_vocab(df["card_id"].to_list())

    decks: dict[tuple[int, int], Deck] = {}
    won: dict[tuple[int, int], bool] = {}
    for (eid, player), g in df.group_by(["episode_id", "player"]):
        key = (int(eid), int(player))
        decks[key] = deck_from_pairs(g["card_id"].to_list(), g["count"].to_list(), vocab)
        won[key] = bool(g["won"][0])

    by_episode: dict[int, list[int]] = {}
    for (eid, player) in decks:
        by_episode.setdefault(eid, []).append(player)

    matchups: list[tuple[Deck, Deck, int]] = []
    for eid, players in by_episode.items():
        if len(players) != 2:
            continue
        p0, p1 = sorted(players)
        d0, d1 = decks[(eid, p0)], decks[(eid, p1)]
        label = 1 if won[(eid, p0)] else 0
        matchups.append((d0, d1, label))
        matchups.append((d1, d0, 1 - label))     # mirror orientation

    flat = {eid * 10 + p: decks[(eid, p)] for (eid, p) in decks}
    return matchups, flat, vocab


def synthetic_matchups(n_archetypes: int = 6, cards_per_arch: int = 12, vocab_cards: int = 120,
                       n_decks: int = 300, matches_per_deck: int = 20, seed: int = 0
                       ) -> tuple[list[tuple[Deck, Deck, int]], list[tuple[int, Deck]], DeckVocab]:
    """Toy data with latent archetypes + a rock-paper-scissors matchup matrix, so
    the smoke test actually exercises clustering-worthy structure."""
    rng = np.random.default_rng(seed)
    vocab = build_vocab(range(1, vocab_cards + 1))
    arch_cards = [rng.choice(range(1, vocab_cards + 1), cards_per_arch, replace=False)
                  for _ in range(n_archetypes)]
    strength = rng.normal(0, 1, n_archetypes)
    rps = rng.normal(0, 1.5, (n_archetypes, n_archetypes))
    rps = rps - rps.T                                   # antisymmetric interaction

    deck_arch, decks = [], []
    for _ in range(n_decks):
        a = int(rng.integers(n_archetypes))
        chosen = rng.choice(arch_cards[a], size=cards_per_arch - 2, replace=False)
        pairs = [(vocab.id2idx[int(c)], int(rng.integers(1, 5))) for c in chosen]
        deck_arch.append(a)
        decks.append(pairs)

    matchups: list[tuple[Deck, Deck, int]] = []
    for i in range(n_decks):
        for _ in range(matches_per_deck):
            j = int(rng.integers(n_decks))
            ai, aj = deck_arch[i], deck_arch[j]
            logit = strength[ai] - strength[aj] + rps[ai, aj]
            label = int(rng.random() < 1 / (1 + np.exp(-logit)))
            matchups.append((decks[i], decks[j], label))

    labelled = list(enumerate(decks))                   # (deck_id, deck) for embedding/NN demo
    return matchups, labelled, vocab


# --------------------------------------------------------------------------- #
# Train / embed / nearest-neighbour.
# --------------------------------------------------------------------------- #

def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' if a GPU is available, else 'cpu'. Otherwise passthrough."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@torch.no_grad()
def evaluate(model: TwoTowerMatchup, matchups: list[tuple[Deck, Deck, int]], *,
             batch_size: int = 256, device: str = "cpu") -> dict[str, float]:
    """Mean BCE loss and matchup accuracy over a set of matchups."""
    loader = DataLoader(MatchupDataset(matchups), batch_size=batch_size, collate_fn=collate,
                        pin_memory=(device == "cuda"))
    model = model.to(device).eval()
    total, correct, loss_sum = 0, 0, 0.0
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        logit = model(b["a_idx"], b["a_cnt"], b["a_mask"], b["b_idx"], b["b_cnt"], b["b_mask"])
        loss = F.binary_cross_entropy_with_logits(logit, b["y"])
        loss_sum += loss.item() * len(b["y"])
        correct += ((logit > 0).float() == b["y"]).sum().item()
        total += len(b["y"])
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}


def train(model: TwoTowerMatchup, matchups: list[tuple[Deck, Deck, int]], *, epochs: int = 5,
          batch_size: int = 128, lr: float = 1e-3, device: str = "cpu",
          val_matchups: list[tuple[Deck, Deck, int]] | None = None,
          num_workers: int = 0,
          verbose: bool = True, on_epoch=None) -> tuple[TwoTowerMatchup, list[dict[str, float]]]:
    """Train the two-tower on matchups. Returns (model, history).

    history is a list of {"epoch", "loss", "acc"[, "val_loss", "val_acc"]} per
    epoch. Pass ``val_matchups`` to also evaluate a held-out set each epoch, and
    ``on_epoch`` (a callback receiving that dict) to stream progress into a UI.
    ``num_workers`` parallelises batch prep; with cuda, batches are pinned and
    copied non-blocking so host prep overlaps GPU compute.
    """
    loader = DataLoader(MatchupDataset(matchups), batch_size=batch_size, shuffle=True,
                        collate_fn=collate, num_workers=num_workers,
                        pin_memory=(device == "cuda"),
                        persistent_workers=num_workers > 0)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for b in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            logit = model(b["a_idx"], b["a_cnt"], b["a_mask"], b["b_idx"], b["b_cnt"], b["b_mask"])
            loss = F.binary_cross_entropy_with_logits(logit, b["y"])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(b["y"])
            correct += ((logit > 0).float() == b["y"]).sum().item()
            total += len(b["y"])
        rec = {"epoch": float(epoch), "loss": loss_sum / total, "acc": correct / total}
        if val_matchups:
            val = evaluate(model, val_matchups, batch_size=batch_size, device=device)
            rec["val_loss"], rec["val_acc"] = val["loss"], val["acc"]
        history.append(rec)
        if verbose:
            extra = f"  val_acc {rec['val_acc']:.3f}" if "val_acc" in rec else ""
            print(f"epoch {epoch}: loss {rec['loss']:.4f}  acc {rec['acc']:.3f}{extra}")
        if on_epoch is not None:
            on_epoch(rec)
    return model, history


@torch.no_grad()
def embed_decks(model: TwoTowerMatchup, decks: list[Deck], *, device: str = "cpu",
                batch_size: int = 256) -> np.ndarray:
    """Encode a list of decks -> (N, emb_dim) numpy array for clustering / NN."""
    model = model.to(device).eval()
    out: list[np.ndarray] = []
    for i in range(0, len(decks), batch_size):
        chunk = decks[i:i + batch_size]
        # second deck is ignored by encode(); reuse padding for a valid batch
        b = collate([(*deck_to_tensors(d), *deck_to_tensors(d), 0.0) for d in chunk])
        e = model.encode(b["a_idx"].to(device), b["a_cnt"].to(device), b["a_mask"].to(device))
        out.append(e.cpu().numpy())
    return np.concatenate(out, axis=0)


def nearest(query: np.ndarray, bank: np.ndarray, k: int = 5, metric: str = "cosine") -> tuple[np.ndarray, np.ndarray]:
    """Top-k nearest rows of `bank` to `query`. Returns (indices, distances)."""
    if metric == "cosine":
        q = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-9)
        b = bank / (np.linalg.norm(bank, axis=-1, keepdims=True) + 1e-9)
        dist = 1.0 - q @ b.T
    elif metric == "euclidean":
        dist = np.linalg.norm(bank[None] - query[:, None], axis=-1)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    idx = np.argsort(dist, axis=-1)[:, :k]
    return idx, np.take_along_axis(dist, idx, axis=-1)


# --------------------------------------------------------------------------- #
# Smoke test: synthetic archetypes -> train -> embed -> the model should rank a
# deck's own archetype as its nearest neighbours.
# --------------------------------------------------------------------------- #

def _smoke() -> None:
    torch.manual_seed(0)
    matchups, labelled, vocab = synthetic_matchups(n_decks=200, matches_per_deck=16)
    print(f"vocab={vocab.size}  decks={len(labelled)}  matchups={len(matchups)}")

    enc = DeckEncoder(vocab.size, dim=32, heads=4, m=8, n_blocks=2, emb_dim=32)
    model = TwoTowerMatchup(enc, emb_dim=32)
    model, _history = train(model, matchups, epochs=3, batch_size=128, lr=1e-3)

    decks = [d for _, d in labelled]
    emb = embed_decks(model, decks)
    print("embeddings:", emb.shape)
    idx, dist = nearest(emb[:3], emb, k=4)
    print("nearest-neighbour indices for decks 0..2:\n", idx)
    print("(each row's first entry is the deck itself, distance ~0)")


if __name__ == "__main__":
    _smoke()
