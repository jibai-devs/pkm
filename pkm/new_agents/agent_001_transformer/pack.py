"""Pack a trained agent_001_transformer checkpoint into a Kaggle bundle.

Mirrors agent_000's ``pack``: a ``.tar.gz`` whose top level is ``main.py``
(this agent's :mod:`.submit_main`), ``weights.pth`` (the checkpoint), and the
``pkm/`` package — flattened so kaggle extracts straight into
``/kaggle_simulations/agent/``.

Run (from repo root)::

    python -m pkm.new_agents.agent_001_transformer.pack \
        --checkpoint out_transformer/latest.pth
"""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

from pkm.new_agents.agent_001_transformer import deck as deck_registry

_MAX_BUNDLE_MIB = 197.7


def _rebake_deck(checkpoint: Path, deck_name: str) -> tuple[Path, tempfile.TemporaryDirectory]:
    """Rewrite ``checkpoint``'s baked deck to ``deck_name`` in a temp copy.

    Returns the temp weights path and the TemporaryDirectory that owns it (keep
    it alive until after the tar is written).
    """
    import torch  # deferred: only needed when overriding

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    blob["deck"] = deck_registry.deck_60(deck_name)
    blob["deck_name"] = deck_name
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "weights.pth"
    torch.save(blob, path)
    return path, tmp


def _baked_deck_name(checkpoint: Path) -> str:
    import torch  # deferred

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return blob.get("deck_name") or "sample (fallback)"


# Path segments that are build/run artifacts, not agent code. Excluding these
# keeps the bundle small: weights ride in as `weights.pth` (added separately), so
# checkpoint dirs like `out/` (hundreds of MiB of .pth) must never be swept into
# the `pkm/` payload — that would blow past Kaggle's 197.7 MiB limit.
_EXCLUDE_DIRS = {"__pycache__", "out", "out_transformer", "submissions", "logs", ".pytest_cache"}


def _bundle_filter(info: tarfile.TarInfo):
    parts = Path(info.name).parts
    base = Path(info.name).name
    if any(p in _EXCLUDE_DIRS for p in parts):
        return None
    if base.endswith((".pyc", ".pyo", ".pth", ".pt")):
        return None
    return info


def main():
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "out_transformer" / "latest.pth",
        help="checkpoint to pack ({state_dict, dims})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=repo_root / "out_transformer" / "submissions",
        help="output dir for the .tar.gz",
    )
    ap.add_argument(
        "--deck",
        default=None,
        choices=deck_registry.deck_names(),
        help="override the deck baked into the checkpoint (default: keep as trained)",
    )
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {args.checkpoint}")
    template = here.with_name("submit_main.py")
    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out / f"submission_{ts}.tar.gz"

    # Deck the bundle will submit: the checkpoint's baked deck, unless overridden.
    tmp = None
    if args.deck is not None:
        weights, tmp = _rebake_deck(args.checkpoint, args.deck)
        deck_label = args.deck
    else:
        weights = args.checkpoint
        deck_label = _baked_deck_name(args.checkpoint)

    try:
        with tarfile.open(out, "w:gz") as tar:
            tar.add(template, arcname="main.py")
            tar.add(weights, arcname="weights.pth")
            tar.add(repo_root / "pkm", arcname="pkm", filter=_bundle_filter)
    finally:
        if tmp is not None:
            tmp.cleanup()

    size_mib = out.stat().st_size / 1024 / 1024
    ok = size_mib <= _MAX_BUNDLE_MIB
    print(f"packed {args.checkpoint.name} (deck: {deck_label}) -> {out}")
    print(f"size: {size_mib:.1f} MiB " + (f"(<= {_MAX_BUNDLE_MIB} limit)" if ok else "(OVER LIMIT!)"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
