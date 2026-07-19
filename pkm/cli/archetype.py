"""Opponent-archetype classifier CLI.

Usage:
    pkm archetype gen-data
    pkm archetype train --n-per-class 200 --epochs 20 --out checkpoints/archetype_latest.pt
    pkm archetype export checkpoints/archetype_latest.pt pkm/archetype.npz
    pkm archetype eval checkpoints/archetype_latest.pt
"""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help=__doc__)
console = Console()


@app.command(name="gen-data")
def gen_data(
    n_per_class: int = typer.Option(50, help="synthetic examples per archetype class"),
    seed: int = typer.Option(0, help="random seed"),
) -> None:
    """Report the alias-resolution status and a synthetic dataset's shape/
    class-balance -- diagnostic, doesn't persist a dataset file (generation
    is cheap and reproducible from a seed via generate_dataset())."""
    from pkm.archetype.archetypes import load_archetypes_with_report
    from pkm.archetype.gen import generate_dataset

    archetypes, report = load_archetypes_with_report()
    resolved_pct = (report.auto + report.alias) / report.total if report.total else 0.0

    console.print(
        f"Resolution: {report.auto} auto + {report.alias} alias = "
        f"{report.auto + report.alias}/{report.total} ({resolved_pct:.1%}), "
        f"{report.unresolved_count} raw unresolved occurrences"
    )

    examples = generate_dataset(n_per_class=n_per_class, seed=seed)
    table = Table(title=f"Synthetic dataset ({len(examples)} examples)")
    table.add_column("Archetype", style="cyan")
    table.add_column("Examples", justify="right")
    for archetype in archetypes:
        table.add_row(archetype.name, str(n_per_class))
    table.add_row("Unknown", str(n_per_class))
    console.print(table)


@app.command()
def train(
    n_per_class: int = typer.Option(200, help="synthetic training examples per class"),
    epochs: int = typer.Option(20, help="training epochs"),
    batch_size: int = typer.Option(64, help="batch size"),
    lr: float = typer.Option(1e-3, help="learning rate"),
    seed: int = typer.Option(0, help="random seed"),
    out: str = typer.Option(
        "checkpoints/archetype_latest.pt", help="output torch state_dict path"
    ),
) -> None:
    """Train the standalone opponent-archetype classifier."""
    from pathlib import Path

    import torch

    from pkm.archetype.train import train as _train

    model, result = _train(
        n_per_class=n_per_class, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed
    )

    console.print(f"held-out overall accuracy: {result.overall_accuracy:.1%}")
    for bucket, acc in sorted(result.accuracy_by_reveal_bucket.items()):
        console.print(f"  reveal {bucket}: {acc:.1%}")
    console.print(
        f"unknown-class accuracy: {result.unknown_accuracy:.1%} "
        f"(mean confidence {result.unknown_confidence:.1%}; when wrong, "
        f"mean confidence {result.unknown_misclassified_confidence:.1%})"
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    console.print(f"saved -> {out_path}")


@app.command()
def export(
    checkpoint: str = typer.Argument(help="path to .pt state_dict"),
    out: str = typer.Argument("pkm/archetype.npz", help="output .npz path"),
) -> None:
    """Export a trained checkpoint to .npz for torch-free inference."""
    from pkm.archetype.archetypes import get_archetypes
    from pkm.archetype.export import export_checkpoint

    num_archetypes = len(get_archetypes())
    export_checkpoint(checkpoint, num_archetypes, out)
    console.print(f"exported {checkpoint} -> {out}")


@app.command(name="eval")
def eval_(
    checkpoint: str = typer.Argument(help="path to .pt state_dict"),
    n_per_class: int = typer.Option(50, help="held-out examples per class"),
    seed: int = typer.Option(1234, help="random seed (should differ from training seed)"),
) -> None:
    """Evaluate a trained checkpoint on a fresh held-out synthetic dataset."""
    import torch

    from pkm.archetype.archetypes import get_archetypes
    from pkm.archetype.gen import generate_dataset
    from pkm.archetype.model import ArchetypeClassifier
    from pkm.archetype.train import evaluate

    num_archetypes = len(get_archetypes())
    model = ArchetypeClassifier(num_archetypes=num_archetypes)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    held_out = generate_dataset(n_per_class=n_per_class, seed=seed)
    result = evaluate(model, held_out)

    console.print(f"overall accuracy: {result.overall_accuracy:.1%}")
    for bucket, acc in sorted(result.accuracy_by_reveal_bucket.items()):
        console.print(f"  reveal {bucket}: {acc:.1%}")
    console.print(
        f"unknown-class accuracy: {result.unknown_accuracy:.1%} "
        f"(mean confidence {result.unknown_confidence:.1%}; when wrong, "
        f"mean confidence {result.unknown_misclassified_confidence:.1%})"
    )


if __name__ == "__main__":
    app()
