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
from datetime import datetime
from pathlib import Path

_MAX_BUNDLE_MIB = 197.7


def _no_pycache(info: tarfile.TarInfo):
    base = Path(info.name).name
    if "__pycache__" in info.name or base.endswith((".pyc", ".pyo")):
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
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {args.checkpoint}")
    template = here.with_name("submit_main.py")
    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out / f"submission_{ts}.tar.gz"

    with tarfile.open(out, "w:gz") as tar:
        tar.add(template, arcname="main.py")
        tar.add(args.checkpoint, arcname="weights.pth")
        tar.add(repo_root / "pkm", arcname="pkm", filter=_no_pycache)

    size_mib = out.stat().st_size / 1024 / 1024
    ok = size_mib <= _MAX_BUNDLE_MIB
    print(f"packed {args.checkpoint.name} -> {out}")
    print(f"size: {size_mib:.1f} MiB " + (f"(<= {_MAX_BUNDLE_MIB} limit)" if ok else "(OVER LIMIT!)"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
