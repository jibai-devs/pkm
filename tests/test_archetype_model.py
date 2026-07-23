"""Tests for pkm.archetype.model/train/export/numpy_model:
training smoke test + torch/numpy parity (hard gate before Part 2, per
docs/opponent-archetype-classifier-plan.md)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pkm.archetype.export import export_npz
from pkm.archetype.gen import generate_dataset
from pkm.archetype.numpy_model import NumpyArchetypeClassifier
from pkm.archetype.train import batch_examples, evaluate, train


def test_training_loss_decreases():
    model, _ = train(n_per_class=15, epochs=1, batch_size=32, log_every=0, seed=1)
    examples = generate_dataset(n_per_class=15, seed=1)
    ids, counts, labels = batch_examples(examples)
    with torch.no_grad():
        loss_after_1_epoch = torch.nn.functional.cross_entropy(
            model.logits(ids, counts), labels
        ).item()

    model2, _ = train(n_per_class=15, epochs=15, batch_size=32, log_every=0, seed=1)
    with torch.no_grad():
        loss_after_15_epochs = torch.nn.functional.cross_entropy(
            model2.logits(ids, counts), labels
        ).item()

    assert loss_after_15_epochs < loss_after_1_epoch


def test_eval_result_shape():
    model, result = train(n_per_class=10, epochs=2, batch_size=32, log_every=0, seed=2)
    assert 0.0 <= result.overall_accuracy <= 1.0
    assert 0.0 <= result.unknown_confidence <= 1.0
    assert 0.0 <= result.unknown_accuracy <= 1.0
    assert 0.0 <= result.unknown_misclassified_confidence <= 1.0

    held_out = generate_dataset(n_per_class=10, seed=3)
    result2 = evaluate(model, held_out)
    assert 0.0 <= result2.overall_accuracy <= 1.0


def test_torch_numpy_parity(tmp_path):
    model, _ = train(n_per_class=10, epochs=3, batch_size=32, log_every=0, seed=4)
    npz_path = tmp_path / "archetype_test.npz"
    export_npz(model, str(npz_path))

    numpy_model = NumpyArchetypeClassifier.load(str(npz_path))

    examples = generate_dataset(n_per_class=2, seed=99)
    ids, counts, labels = batch_examples(examples)
    with torch.no_grad():
        torch_probs = model(ids, counts).numpy()

    for i, example in enumerate(examples):
        card_ids = np.array(list(example.revealed.keys()), dtype=np.int64)
        card_counts = np.array(list(example.revealed.values()), dtype=np.float32)
        numpy_probs = numpy_model.belief(card_ids, card_counts)
        assert np.abs(numpy_probs - torch_probs[i]).max() < 1e-4


def test_numpy_model_rejects_stale_stamp(tmp_path):
    from pkm.archetype.model import ArchetypeClassifier

    model = ArchetypeClassifier(num_archetypes=3)  # deliberately wrong count
    npz_path = tmp_path / "stale.npz"
    export_npz(model, str(npz_path))

    import pytest

    with pytest.raises(ValueError, match="retrain and re-export"):
        NumpyArchetypeClassifier.load(str(npz_path))
