"""Pointer-style policy/value network.

Architecture overview:
    - **State encoder**: board card embeddings + hand bag-of-embeddings +
      scalar features -> 128-dim hidden vector `h`.
    - **Policy head** (lines 104-123): scores each legal option against `h`
      via a 2-layer MLP, producing logits over (options + STOP). The agent
      samples (or argmaxes) from the softmax of these logits. Multi-pick
      decisions are sequential: pick one, mask it, re-score, repeat.
    - **Value head** (lines 125-126): V(s) predictor. Two linear layers on
      `h`, tanh squashed to [-1, +1]. Predicts the expected final outcome
      (win=+1, loss=-1) from the current state, regardless of action taken.
      Used as a baseline in PPO (advantage = actual - predicted) and as
      leaf evaluation in MCTS.

The state is embedded into a vector ``h``; each option is embedded into a
vector ``o_i`` and scored against ``h`` (plus a summary of already-picked
options), giving a softmax over the variable-length legal option list.

Multi-select decisions are decomposed into sequential picks. A learned STOP
row is appended to the option list; it becomes legal once ``minCount`` picks
are made, and picking it ends the sequence. The decision's log-prob is the
sum of the per-pick log-probs.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    NUM_ATTACKS,
    NUM_CARDS,
    NUM_OPT_TYPES,
    OPT_FEATS,
    STATE_FEATS,
    EncodedDecision,
)
from .features import ARCHETYPE_OUT
from pkm.types.obs import N_POKEMON_SLOTS

EMB_CARD = 32
EMB_ATTACK = 16
EMB_OPT_TYPE = 8
OPT_ENC = 64
HIDDEN = 128

N_MY_SLOTS = (
    N_POKEMON_SLOTS // 2
)  # my active + bench; opponent's side is the other half

# Board split into my/opp pooled vectors + stadium, one EMB_CARD-wide block
# each, plus the pooled hand and the pooled deck-ledger (Task 7) -- five
# EMB_CARD-wide blocks total, independent of N_BOARD_SLOTS/bench size.
STATE_IN = 5 * EMB_CARD + STATE_FEATS
OPT_IN = 2 * EMB_CARD + EMB_ATTACK + EMB_OPT_TYPE + OPT_FEATS
SCORE_IN = HIDDEN + 2 * OPT_ENC

NEG_INF = -1e9


@dataclass
class ActResult:
    picks: list[int]
    stopped: bool
    logprob: float
    value: float
    belief: np.ndarray  # (ARCHETYPE_OUT,) detached opponent-archetype belief


class PolicyValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.card_emb = nn.Embedding(NUM_CARDS, EMB_CARD)
        self.attack_emb = nn.Embedding(NUM_ATTACKS, EMB_ATTACK)
        self.opt_type_emb = nn.Embedding(NUM_OPT_TYPES, EMB_OPT_TYPE)
        self.stop_vec = nn.Parameter(torch.zeros(OPT_ENC))

        # --- state encoder: board + hand + features -> 128-dim h ---
        self.state_fc1 = nn.Linear(STATE_IN, 512)
        self.state_fc2 = nn.Linear(512, HIDDEN)

        # --- policy head: scores each option against h ---
        self.opt_fc = nn.Linear(OPT_IN, OPT_ENC)
        self.score_fc1 = nn.Linear(SCORE_IN, 64)
        self.score_fc2 = nn.Linear(64, 1)

        # --- value head: V(s) predictor from h -> scalar in [-1, +1] ---
        self.value_fc1 = nn.Linear(HIDDEN, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # --- auxiliary head: opponent-archetype belief (Task 8) ---
        self.archetype_fc1 = nn.Linear(HIDDEN, 64)
        self.archetype_fc2 = nn.Linear(64, ARCHETYPE_OUT)

    # --- building blocks ---

    def _pool_cards(self, ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool a (B, K) card-id array over its non-empty (id > 0)
        entries -> (B, EMB_CARD). Used for hand and, since Task 7, each
        side of the board separately (my slots / opponent's slots)."""
        e = self.card_emb(ids)
        mask = (ids > 0).float().unsqueeze(-1)
        return (e * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def _pool_deck(self, ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Task 7 deck ledger: h_memory = sum_c unseen_count[c] * card_emb[c].
        (B, K) ids + (B, K) counts -> (B, EMB_CARD). Unlike _pool_cards this
        is a weighted sum, not a mean -- total magnitude legitimately grows
        with how much of the deck is still unseen."""
        e = self.card_emb(ids)
        return (e * counts.unsqueeze(-1)).sum(1)

    def encode_state(
        self,
        board_cards: torch.Tensor,
        hand_cards: torch.Tensor,
        state_feats: torch.Tensor,
        deck_card_ids: torch.Tensor,
        deck_card_counts: torch.Tensor,
    ) -> torch.Tensor:
        """(B, N_BOARD_SLOTS) board, (B, MAX_HAND) hand, (B, STATE_FEATS)
        feats, (B, K) deck ledger ids + counts -> (B, HIDDEN).

        Board is split into my slots / opponent's slots / stadium and each
        pooled separately (Task 7), instead of one flat positional
        concatenation across all N_BOARD_SLOTS.
        """
        my_board = self._pool_cards(board_cards[:, :N_MY_SLOTS])
        opp_board = self._pool_cards(board_cards[:, N_MY_SLOTS : 2 * N_MY_SLOTS])
        stadium = self.card_emb(board_cards[:, -1])
        hand = self._pool_cards(hand_cards)
        deck = self._pool_deck(deck_card_ids, deck_card_counts)
        x = torch.cat([my_board, opp_board, stadium, hand, deck, state_feats], dim=1)
        return F.relu(self.state_fc2(F.relu(self.state_fc1(x))))

    def encode_options(
        self,
        opt_type: torch.Tensor,
        opt_card: torch.Tensor,
        opt_card2: torch.Tensor,
        opt_attack: torch.Tensor,
        opt_feats: torch.Tensor,
    ) -> torch.Tensor:
        """(B, N) id arrays + (B, N, OPT_FEATS) -> (B, N, OPT_ENC)."""
        x = torch.cat(
            [
                self.card_emb(opt_card),
                self.card_emb(opt_card2),
                self.attack_emb(opt_attack),
                self.opt_type_emb(opt_type),
                opt_feats,
            ],
            dim=-1,
        )
        return F.relu(self.opt_fc(x))

    def option_logits(
        self,
        h: torch.Tensor,
        opts: torch.Tensor,
        picked_sum: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Policy head: score options (+ STOP as the last column).

        h: (B, HIDDEN); opts: (B, N, OPT_ENC); picked_sum: (B, OPT_ENC);
        mask: (B, N+1) bool, True = selectable. Returns (B, N+1) logits.
        """
        b, n, _ = opts.shape
        stop = self.stop_vec.expand(b, 1, OPT_ENC)
        rows = torch.cat([opts, stop], dim=1)  # (B, N+1, OPT_ENC)
        hx = h.unsqueeze(1).expand(b, n + 1, HIDDEN)
        px = picked_sum.unsqueeze(1).expand(b, n + 1, OPT_ENC)
        x = torch.cat([hx, rows, px], dim=-1)
        logits = self.score_fc2(F.relu(self.score_fc1(x))).squeeze(-1)
        return logits.masked_fill(~mask, NEG_INF)

    def value(self, h: torch.Tensor) -> torch.Tensor:
        """Value head: V(s) — expected final outcome from this state.

        Returns tanh-squashed scalar in [-1, +1]. Used as baseline in PPO
        (advantage = actual_return - V(s)) and as leaf eval in MCTS.
        Predicts from the state alone, not conditioned on any action.
        """
        return torch.tanh(self.value_fc2(F.relu(self.value_fc1(h)))).squeeze(-1)

    def archetype_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Auxiliary head (Task 8): opponent-archetype classification
        logits, width ARCHETYPE_OUT (tracked archetypes + "Other").

        Trained only against its own cross-entropy loss (pkm/rl/ppo.py) --
        must never receive gradient from the policy/value loss (plan.md
        §8.2 rule 1). This method returns raw logits for that training use;
        callers wanting the belief for GLOBAL-feature re-injection must use
        archetype_belief() instead, which enforces detachment.
        """
        return self.archetype_fc2(F.relu(self.archetype_fc1(h)))

    @torch.no_grad()
    def archetype_belief(self, h: torch.Tensor) -> torch.Tensor:
        """Detached softmax belief for GLOBAL-feature re-injection. The
        @torch.no_grad() here (not a bare .detach()) is deliberate: it
        prevents an autograd graph from ever being built for this call, so
        there is no live tensor for a caller to accidentally backprop
        through."""
        return F.softmax(self.archetype_logits(h), dim=-1)

    # --- acting (single decision, no grad) ---

    @torch.no_grad()
    def act(
        self, d: EncodedDecision, greedy: bool = False, temperature: float = 1.0
    ) -> ActResult:
        board = torch.from_numpy(d.board_cards).unsqueeze(0)
        hand = torch.from_numpy(d.hand_cards).unsqueeze(0)
        feats = torch.from_numpy(d.state_feats).unsqueeze(0)
        deck_ids = torch.from_numpy(d.deck_card_ids).unsqueeze(0)
        deck_counts = torch.from_numpy(d.deck_card_counts).unsqueeze(0)
        h = self.encode_state(board, hand, feats, deck_ids, deck_counts)
        v = float(self.value(h)[0])
        belief = self.archetype_belief(h)[0].numpy()

        n = len(d.opt_type)
        opts = self.encode_options(
            torch.from_numpy(d.opt_type).unsqueeze(0),
            torch.from_numpy(d.opt_card).unsqueeze(0),
            torch.from_numpy(d.opt_card2).unsqueeze(0),
            torch.from_numpy(d.opt_attack).unsqueeze(0),
            torch.from_numpy(d.opt_feats).unsqueeze(0),
        )

        picks: list[int] = []
        stopped = False
        logprob = 0.0
        picked_sum = torch.zeros(1, OPT_ENC)
        available = torch.ones(1, n + 1, dtype=torch.bool)

        while len(picks) < d.max_count:
            available[0, n] = len(picks) >= d.min_count
            logits = self.option_logits(h, opts, picked_sum, available)
            if temperature != 1.0:
                logits = logits / temperature
            logp = F.log_softmax(logits, dim=-1)
            if greedy:
                idx = int(logp.argmax(dim=-1)[0])
            else:
                idx = int(torch.multinomial(logp.exp(), 1)[0, 0])
            logprob += float(logp[0, idx])
            if idx == n:
                stopped = True
                break
            picks.append(idx)
            picked_sum = picked_sum + opts[0, idx]
            available[0, idx] = False

        return ActResult(
            picks=picks, stopped=stopped, logprob=logprob, value=v, belief=belief
        )

    # --- training (batched re-evaluation) ---

    @staticmethod
    def _batch_state_tensors(
        decisions: list[EncodedDecision],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(board, hand, feats, deck_ids, deck_counts) padded batch tensors,
        shared by evaluate() and evaluate_archetype()."""
        board = torch.from_numpy(np.stack([d.board_cards for d in decisions]))
        hand = torch.from_numpy(np.stack([d.hand_cards for d in decisions]))
        feats = torch.from_numpy(np.stack([d.state_feats for d in decisions]))

        b = len(decisions)
        k_deck_max = max((len(d.deck_card_ids) for d in decisions), default=0)
        deck_ids = np.zeros((b, k_deck_max), dtype=np.int64)
        deck_counts = np.zeros((b, k_deck_max), dtype=np.float32)
        for i, d in enumerate(decisions):
            k = len(d.deck_card_ids)
            deck_ids[i, :k] = d.deck_card_ids
            deck_counts[i, :k] = d.deck_card_counts

        return (
            board,
            hand,
            feats,
            torch.from_numpy(deck_ids),
            torch.from_numpy(deck_counts),
        )

    def evaluate_archetype(self, decisions: list[EncodedDecision]) -> torch.Tensor:
        """Batched archetype-head logits for a set of decisions (Task 8 aux
        loss, pkm/rl/ppo.py). Kept separate from evaluate() so the
        policy/value path is untouched by this -- callers combining this
        with pi_loss/v_loss.backward() are responsible for weighting it,
        never letting it dominate."""
        board, hand, feats, deck_ids, deck_counts = self._batch_state_tensors(decisions)
        h = self.encode_state(board, hand, feats, deck_ids, deck_counts)
        return self.archetype_logits(h)

    def evaluate(
        self, decisions: list[EncodedDecision]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute (logprobs, entropies, values) for a batch of decisions.

        Entropy is the mean per-pick entropy. Handles variable option counts
        and pick-sequence lengths by padding + masking.
        """
        b = len(decisions)
        n_max = max(len(d.opt_type) for d in decisions)
        # sequence length includes the STOP step when it was taken
        seq_lens = [len(d.picks) + (1 if d.stopped else 0) for d in decisions]
        k_max = max(max(seq_lens), 1)

        board, hand, feats, deck_ids, deck_counts = self._batch_state_tensors(decisions)

        def pad_ids(key: str) -> torch.Tensor:
            out = np.zeros((b, n_max), dtype=np.int64)
            for i, d in enumerate(decisions):
                arr = getattr(d, key)
                out[i, : len(arr)] = arr
            return torch.from_numpy(out)

        opt_feats = np.zeros((b, n_max, OPT_FEATS), dtype=np.float32)
        valid = np.zeros((b, n_max + 1), dtype=bool)
        for i, d in enumerate(decisions):
            n = len(d.opt_type)
            opt_feats[i, :n] = d.opt_feats
            valid[i, :n] = True

        h = self.encode_state(board, hand, feats, deck_ids, deck_counts)
        values = self.value(h)
        opts = self.encode_options(
            pad_ids("opt_type"),
            pad_ids("opt_card"),
            pad_ids("opt_card2"),
            pad_ids("opt_attack"),
            torch.from_numpy(opt_feats),
        )

        valid_t = torch.from_numpy(valid)
        picked_sum = torch.zeros(b, OPT_ENC)
        available = valid_t.clone()
        logprobs = torch.zeros(b)
        entropies = torch.zeros(b)
        steps_taken = torch.zeros(b)

        # per-step target index; n_max column = STOP
        targets = np.full((b, k_max), -1, dtype=np.int64)
        for i, d in enumerate(decisions):
            for k, p in enumerate(d.picks):
                targets[i, k] = p
            if d.stopped:
                targets[i, len(d.picks)] = n_max
        targets_t = torch.from_numpy(targets)

        for k in range(k_max):
            active = targets_t[:, k] >= 0
            if not bool(active.any()):
                break
            # STOP is legal once min_count picks have been made (k picks so far)
            stop_legal = torch.tensor([k >= d.min_count for d in decisions])
            mask = available.clone()
            mask[:, n_max] = stop_legal
            logits = self.option_logits(h, opts, picked_sum, mask)
            logp = F.log_softmax(logits, dim=-1)
            probs = logp.exp()
            step_ent = -(probs * logp).sum(dim=-1)

            tgt = targets_t[:, k].clamp(min=0)
            step_lp = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            logprobs = logprobs + step_lp * active
            entropies = entropies + step_ent * active
            steps_taken = steps_taken + active.float()

            # update picked_sum/available for samples that picked a real option
            picked_real = active & (targets_t[:, k] < n_max)
            if bool(picked_real.any()):
                idx = tgt.clamp(max=n_max - 1)
                chosen = opts.gather(
                    1, idx.view(b, 1, 1).expand(b, 1, OPT_ENC)
                ).squeeze(1)
                picked_sum = picked_sum + chosen * picked_real.unsqueeze(1)
                rows = torch.nonzero(picked_real).squeeze(1)
                available[rows, idx[rows]] = False

        entropies = entropies / steps_taken.clamp(min=1.0)
        return logprobs, entropies, values
