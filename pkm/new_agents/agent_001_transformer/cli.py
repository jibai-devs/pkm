"""Typer + rich command line for agent_001_transformer.

A self-contained entry point for training, evaluating and packing the
AlphaZero-style transformer agent — one command per verb, with ``--deck``
selecting which hard-coded 60-card list (see :mod:`.deck`) a run plays.

Commands::

    cli.py decks                       # list the hard-coded decks
    cli.py info                        # paths, dims, engine backend
    cli.py train --deck pult_munki     # self-play + MCTS training run
    cli.py eval  --deck pult_munki -c out/latest.pth
    cli.py pack  -c out/latest.pth     # bundle for Kaggle (deck baked in)

Because the transformer's encoder is a bag-of-features over a *shared* sparse
index space (not a per-deck learned vocabulary), adding or switching decks never
changes the network shape — a deck is purely the 60-card list a seat plays, and
it is baked into every checkpoint so packing/inference stay self-contained.

Run ``cli.py <command> --help`` for the full flag list. Heavy imports (torch via
``net``/``train``) are deferred into command bodies so ``--help`` stays fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pkm.new_agents.agent_001_transformer import deck as deck_registry

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Train / eval / pack the agent_001_transformer AlphaZero agent.",
)
console = Console()

# Default artifact root: <repo>/out_transformer
#   cli.py -> agent_001_transformer -> new_agents -> pkm -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "out_transformer"

_DECK_HELP = f"deck to play — one of {deck_registry.deck_names()} (default: {deck_registry.DEFAULT_DECK})"


def _validate_deck(name: str) -> str:
    if name not in deck_registry.DECKS:
        raise typer.BadParameter(
            f"unknown deck {name!r}; choose from {deck_registry.deck_names()}"
        )
    return name


# --------------------------------------------------------------------------- #
# decks
# --------------------------------------------------------------------------- #
@app.command()
def decks(
    name: Optional[str] = typer.Argument(
        None, help="show the full card list for one deck instead of the summary"
    ),
) -> None:
    """List the hard-coded decks (or the full card list for one deck)."""
    if name is not None:
        _validate_deck(name)
        table = Table(title=f"deck: {name}", box=None)
        table.add_column("id", justify="right", style="cyan")
        table.add_column("card")
        table.add_column("x", justify="right", style="magenta")
        total = 0
        for cid, cname, count in deck_registry.deck_def(name):
            table.add_row(str(cid), cname, str(count))
            total += count
        table.add_section()
        table.add_row("", "[bold]total[/bold]", f"[bold]{total}[/bold]")
        console.print(table)
        return

    table = Table(title="registered decks", box=None)
    table.add_column("name", style="green")
    table.add_column("cards", justify="right")
    table.add_column("headline pokémon")
    for dname, ddef in deck_registry.DECKS.items():
        # A deck's "headline" = its highest-count non-energy Pokémon-ish cards.
        pokemon = [n for _cid, n, _c in ddef if "Energy" not in n][:3]
        default = " [dim](default)[/dim]" if dname == deck_registry.DEFAULT_DECK else ""
        table.add_row(dname + default, str(sum(c for _, _, c in ddef)), ", ".join(pokemon))
    console.print(table)


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
@app.command()
def info() -> None:
    """Show architecture dims, paths and the engine backend."""
    import os

    from pkm.new_agents.agent_001_transformer import net

    body = [
        f"[bold]dims[/bold]          {net.MODEL_DIMS}  (d_model, heads, d_ff, n_enc, n_dec)",
        f"[bold]card vocab[/bold]    {net.card_count} ids, encoder space {net.encoder_size}",
        f"[bold]decoder size[/bold]  {net.decoder_size}",
        f"[bold]default sims[/bold]  {net.SEARCH_COUNT}",
        f"[bold]decks[/bold]         {deck_registry.deck_names()} (default: {deck_registry.DEFAULT_DECK})",
        f"[bold]engine[/bold]        PKM_ENGINE={os.environ.get('PKM_ENGINE', 'kaggle')}",
        f"[bold]default out[/bold]   {DEFAULT_OUT}",
    ]
    console.print(Panel("\n".join(body), title="agent_001_transformer", expand=False))


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
@app.command()
def train(
    deck: str = typer.Option(deck_registry.DEFAULT_DECK, "--deck", help=_DECK_HELP, callback=_validate_deck),
    iters: int = typer.Option(5, help="training iterations"),
    eval_games: int = typer.Option(50, "--eval-games", help="eval games per iteration"),
    selfplay_games: int = typer.Option(100, "--selfplay-games", help="self-play games per iteration"),
    sims: int = typer.Option(10, help="MCTS simulations per decision"),
    lr: float = typer.Option(3e-4, help="AdamW learning rate"),
    batch_size: int = typer.Option(128, "--batch-size", help="training batch size"),
    device: Optional[str] = typer.Option(None, help="cpu | cuda (default: auto)"),
    init: Optional[Path] = typer.Option(None, help="resume from a {state_dict,dims} checkpoint"),
    out: Path = typer.Option(DEFAULT_OUT, help="checkpoint output dir"),
) -> None:
    """Self-play + MCTS training run. Bakes the played deck into each checkpoint."""
    from pkm.new_agents.agent_001_transformer import train as trainer

    console.print(
        Panel(
            f"deck [green]{deck}[/green] · iters {iters} · sims {sims} · "
            f"selfplay {selfplay_games} · eval {eval_games} · lr {lr}",
            title="training",
            expand=False,
        )
    )
    latest = trainer.train_loop(
        deck_name=deck,
        iters=iters,
        eval_games=eval_games,
        selfplay_games=selfplay_games,
        sims=sims,
        lr=lr,
        batch_size=batch_size,
        device=device,
        init=init,
        out=out,
    )
    console.print(f"[bold green]done[/bold green] · latest -> {latest}")


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@app.command()
def eval(  # noqa: A001 - shadowing builtin `eval` is fine for a CLI verb
    checkpoint: Path = typer.Option(DEFAULT_OUT / "latest.pth", "--checkpoint", "-c", help="checkpoint to evaluate"),
    deck: Optional[str] = typer.Option(None, "--deck", help=_DECK_HELP + " [default: the checkpoint's baked deck]"),
    games: int = typer.Option(100, help="eval games vs random"),
    sims: int = typer.Option(10, help="MCTS simulations per decision"),
    device: Optional[str] = typer.Option(None, help="cpu | cuda (default: auto)"),
) -> None:
    """Win-rate of a checkpoint vs a random opponent (mirror deck)."""
    import torch

    from pkm.new_agents.agent_001_transformer import net
    from pkm.new_agents.agent_001_transformer import train as trainer

    if not checkpoint.is_file():
        raise typer.BadParameter(f"no checkpoint at {checkpoint}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blob = torch.load(checkpoint, map_location=dev, weights_only=False)
    dims = tuple(blob.get("dims", net.MODEL_DIMS))
    model = net.MyModel(*dims).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    # Deck: explicit --deck wins; else the checkpoint's baked deck; else sample.
    if deck is not None:
        _validate_deck(deck)
        deck_ids = deck_registry.deck_60(deck)
        deck_label = deck
    elif blob.get("deck"):
        deck_ids = list(blob["deck"])
        deck_label = blob.get("deck_name") or "baked"
    else:
        deck_ids = list(net.sample_deck)
        deck_label = "sample (fallback)"

    console.print(
        Panel(
            f"checkpoint [cyan]{checkpoint.name}[/cyan] · deck [green]{deck_label}[/green] · "
            f"dims {dims} · {games} games · sims {sims} · {dev}",
            title="eval",
            expand=False,
        )
    )
    with torch.inference_mode():
        win = trainer.evaluate(model, deck_ids, sims, games)
    console.print(f"[bold]win rate vs random: [green]{win}%[/green][/bold]")


# --------------------------------------------------------------------------- #
# pack
# --------------------------------------------------------------------------- #
@app.command()
def pack(
    checkpoint: Path = typer.Option(DEFAULT_OUT / "latest.pth", "--checkpoint", "-c", help="checkpoint to pack"),
    deck: Optional[str] = typer.Option(None, "--deck", help="override the deck baked in the bundle [default: keep as trained]"),
    out: Path = typer.Option(DEFAULT_OUT / "submissions", help="output dir for the .tar.gz"),
) -> None:
    """Bundle a checkpoint into a Kaggle submission (deck baked into weights.pth)."""
    import tarfile
    from datetime import datetime

    from pkm.new_agents.agent_001_transformer import pack as packer

    if not checkpoint.is_file():
        raise typer.BadParameter(f"no checkpoint at {checkpoint}")
    if deck is not None:
        _validate_deck(deck)

    template = Path(packer.__file__).with_name("submit_main.py")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = out / f"submission_{ts}.tar.gz"

    tmp = None
    if deck is not None:
        weights, tmp = packer._rebake_deck(checkpoint, deck)
        deck_label = deck
    else:
        weights = checkpoint
        deck_label = packer._baked_deck_name(checkpoint)

    try:
        with tarfile.open(bundle, "w:gz") as tar:
            tar.add(template, arcname="main.py")
            tar.add(weights, arcname="weights.pth")
            tar.add(REPO_ROOT / "pkm", arcname="pkm", filter=packer._bundle_filter)
    finally:
        if tmp is not None:
            tmp.cleanup()

    size_mib = bundle.stat().st_size / 1024 / 1024
    ok = size_mib <= packer._MAX_BUNDLE_MIB
    console.print(f"packed [cyan]{checkpoint.name}[/cyan] (deck: [green]{deck_label}[/green]) -> {bundle}")
    style = "green" if ok else "red"
    limit = f"<= {packer._MAX_BUNDLE_MIB} limit" if ok else "OVER LIMIT!"
    console.print(f"size: [{style}]{size_mib:.1f} MiB ({limit})[/{style}]")
    if not ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
