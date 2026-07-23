"""Training loop for the standalone opponent-archetype classifier.

Supervised cross-entropy on the synthetic partial-reveal dataset
(pkm/archetype/gen.py). No PPO/RL here -- this is a plain classifier
training loop, mirroring pkm/rl/train.py's CLI shape but much simpler.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pkm.archetype.archetypes import get_archetypes
from pkm.archetype.gen import Example, generate_dataset
from pkm.archetype.model import ArchetypeClassifier


def batch_examples(
    examples: list[Example],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length revealed-card bags into (ids, counts,
    labels) tensors. Padded slots use card_id=0, count=0.0, contributing
    nothing to the count-weighted pool."""
    b = len(examples)
    k_max = max((len(e.revealed) for e in examples), default=0)
    k_max = max(k_max, 1)
    ids = np.zeros((b, k_max), dtype=np.int64)
    counts = np.zeros((b, k_max), dtype=np.float32)
    labels = np.zeros(b, dtype=np.int64)
    for i, example in enumerate(examples):
        for j, (card_id, count) in enumerate(example.revealed.items()):
            ids[i, j] = card_id
            counts[i, j] = count
        labels[i] = example.label
    return torch.from_numpy(ids), torch.from_numpy(counts), torch.from_numpy(labels)


@dataclass
class EvalResult:
    overall_accuracy: float
    accuracy_by_reveal_bucket: dict[str, float]
    unknown_confidence: float  # mean max-prob on held-out "unknown" examples
    unknown_accuracy: float  # fraction of "unknown" examples predicted as Unknown
    unknown_misclassified_confidence: float  # mean confidence when confidently wrong


REVEAL_BUCKETS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.51)]


def evaluate(model: ArchetypeClassifier, examples: list[Example]) -> EvalResult:
    ids, counts, labels = batch_examples(examples)
    with torch.no_grad():
        logits = model.logits(ids, counts)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
    correct = (preds == labels).numpy()

    buckets: dict[str, float] = {}
    reveal_fracs = np.array([e.reveal_frac for e in examples])
    for lo, hi in REVEAL_BUCKETS:
        mask = (reveal_fracs >= lo) & (reveal_fracs < hi)
        if mask.sum() > 0:
            buckets[f"{lo}-{hi}"] = float(correct[mask].mean())

    unknown_label = model.num_archetypes
    unknown_mask = labels.numpy() == unknown_label
    unknown_probs = probs.numpy()[unknown_mask]
    unknown_preds = preds.numpy()[unknown_mask]
    unknown_conf = float(unknown_probs.max(axis=-1).mean()) if unknown_mask.any() else 0.0
    unknown_acc = (
        float((unknown_preds == unknown_label).mean()) if unknown_mask.any() else 0.0
    )
    misclassified = unknown_preds != unknown_label
    misclassified_conf = (
        float(unknown_probs[misclassified].max(axis=-1).mean())
        if misclassified.any()
        else 0.0
    )

    return EvalResult(
        overall_accuracy=float(correct.mean()),
        accuracy_by_reveal_bucket=buckets,
        unknown_confidence=unknown_conf,
        unknown_accuracy=unknown_acc,
        unknown_misclassified_confidence=misclassified_conf,
    )


def train(
    n_per_class: int = 200,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 5,
    label_smoothing: float = 0.1,
) -> tuple[ArchetypeClassifier, EvalResult]:
    """label_smoothing > 0 directly targets the "confidently wrong on
    off-meta decks" failure mode (plan's calibration gate): it caps how
    close the softmax can get to a one-hot target, so the model can't
    collapse to overconfident wrong predictions on distribution it hasn't
    seen, at a small cost to peak accuracy on examples it has."""
    archetypes = get_archetypes()
    model = ArchetypeClassifier(num_archetypes=len(archetypes))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_examples = generate_dataset(n_per_class=n_per_class, seed=seed)
    held_out = generate_dataset(n_per_class=max(n_per_class // 5, 10), seed=seed + 1)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        order = rng.permutation(len(train_examples))
        total_loss = 0.0
        for start in range(0, len(train_examples), batch_size):
            batch_idx = order[start : start + batch_size]
            batch = [train_examples[i] for i in batch_idx]
            ids, counts, labels = batch_examples(batch)
            logits = model.logits(ids, counts)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / len(train_examples)
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch}: train_loss={avg_loss:.4f}")

    result = evaluate(model, held_out)
    return model, result
