"""Task 8: opponent-archetype auxiliary head.

Per plan.md §8.2 rule 3, the head must be validated standalone (accuracy
clears a non-trivial threshold, trunk frozen) before its output is trusted
for re-injection. This file covers that standalone validation plus the
gradient-isolation guarantee from rule 1 -- PPO's own loss must never reach
archetype_fc1/archetype_fc2.
"""

import random

import torch
import torch.nn.functional as F

from pkm.data import Deck
from pkm.engine import battle_finish, battle_select, battle_start
from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.rl.encoder import EncodedDecision, encode_decision
from pkm.rl.features import ARCHETYPE_CLASSES, archetype_index
from pkm.rl.model import PolicyValueNet
from pkm.types.obs import Observation

MAX_DECISIONS_PER_GAME = 200


def _collect_labeled_decisions(
    deck_names: list[str], n_games: int, rng: random.Random
) -> list[EncodedDecision]:
    """Quick random-vs-random self-play across varied deck matchups,
    encoding every real (non-forced) decision with the true opponent
    archetype label. No network inference needed for acting -- moves are
    picked uniformly at random -- so this is cheap relative to a real
    training run."""
    decks = {name: Deck.from_csv(f"deck/{name}.csv").card_ids for name in deck_names}
    examples: list[EncodedDecision] = []

    for _ in range(n_games):
        name_a = rng.choice(deck_names)
        name_b = rng.choice(deck_names)
        deck_a, deck_b = decks[name_a], decks[name_b]
        labels = (archetype_index(name_b), archetype_index(name_a))

        obs, start = battle_start(list(deck_a), list(deck_b))
        if obs is None:
            continue
        contexts = (
            GameContext(list(deck_a), DeckTracker(deck_a)),
            GameContext(list(deck_b), DeckTracker(deck_b)),
        )
        count = 0
        try:
            while obs["current"]["result"] < 0 and count < MAX_DECISIONS_PER_GAME:
                p = obs["current"]["yourIndex"]
                tracker = contexts[p].tracker
                tracker.observe(obs)
                if tracker.is_search_reveal(obs):
                    tracker.record_search_reveal(obs)

                parsed = Observation.model_validate(obs)
                sel = parsed.select
                assert sel is not None
                forced = sel.forced_picks()
                if forced is not None:
                    obs = battle_select(forced)
                    count += 1
                    continue

                d = encode_decision(parsed, contexts[p])
                d.true_archetype = labels[p]
                examples.append(d)

                n = len(sel.option)
                picks = rng.sample(range(n), sel.maxCount)
                obs = battle_select(picks)
                count += 1
        finally:
            battle_finish()

    return examples


def _freeze_trunk_train_archetype_head_only(model: PolicyValueNet) -> None:
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("archetype_"))


def test_archetype_head_standalone_accuracy():
    """plan.md §8.2 rule 3: prove the head can learn the label at all,
    trunk frozen, before Step 5 wires the detached re-injection."""
    random.seed(0)
    torch.manual_seed(0)

    model = PolicyValueNet()
    model.eval()
    _freeze_trunk_train_archetype_head_only(model)

    rng = random.Random(0)
    examples = _collect_labeled_decisions(ARCHETYPE_CLASSES, n_games=40, rng=rng)
    assert len(examples) >= 100, "collected too few decisions to train/evaluate on"

    rng.shuffle(examples)
    split = int(len(examples) * 0.8)
    train_set, test_set = examples[:split], examples[split:]
    assert len(test_set) >= 20

    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if n.startswith("archetype_")],
        lr=1e-3,
    )
    for _ in range(15):
        rng.shuffle(train_set)
        for start in range(0, len(train_set), 32):
            batch = train_set[start : start + 32]
            labels = torch.tensor([d.true_archetype for d in batch], dtype=torch.long)
            logits = model.evaluate_archetype(batch)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        logits = model.evaluate_archetype(test_set)
        preds = logits.argmax(dim=-1)
        labels = torch.tensor([d.true_archetype for d in test_set], dtype=torch.long)
        accuracy = float((preds == labels).float().mean())

    random_baseline = 1.0 / len(ARCHETYPE_CLASSES)
    assert accuracy > random_baseline * 1.5, (
        f"archetype head accuracy {accuracy:.2f} did not clear "
        f"{random_baseline * 1.5:.2f} (1.5x random baseline {random_baseline:.2f})"
    )


def test_ppo_loss_gradient_never_reaches_archetype_head():
    """plan.md §8.2 rule 1, non-negotiable: the auxiliary head must never
    receive gradient from the policy/value loss."""
    torch.manual_seed(0)
    model = PolicyValueNet()
    model.train()

    rng = random.Random(1)
    examples = _collect_labeled_decisions(["00_basic"], n_games=2, rng=rng)
    assert len(examples) >= 1

    logprobs, entropies, values = model.evaluate(examples)
    pi_v_loss = -logprobs.mean() + values.pow(2).mean() - 0.01 * entropies.mean()

    model.zero_grad()
    pi_v_loss.backward()

    grad = model.archetype_fc1.weight.grad
    assert grad is None or torch.all(grad == 0)
    grad2 = model.archetype_fc2.weight.grad
    assert grad2 is None or torch.all(grad2 == 0)
