"""Export agent_001_transformer's net to ONNX and serve it in Netron.

Loads a checkpoint (``out/latest.pth`` by default, else a fresh net), builds a
*real* pair of encoder/decoder sparse inputs from an actual initial observation
(so the traced graph has faithful shapes), exports to ONNX, then starts the
Netron web server on the resulting file.

Run::

    python -m pkm.new_agents.agent_001_transformer.scripts.netron_view
    # or the convenience wrapper:
    bash pkm/new_agents/agent_001_transformer/scripts/netron_view.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pkm.new_agents.agent_000_dragapult.cabt import (
    battle_finish,
    battle_start,
)
from pkm.new_agents.agent_001_transformer import net

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent


def _default_checkpoint() -> Path:
    """out/latest.pth, resolving to the primary checkout when run from a
    git worktree (checkpoints are gitignored, so they live only in the main tree)."""
    local = AGENT_DIR / "out" / "latest.pth"
    if local.exists():
        return local
    # .../<repo>/.claude/worktrees/<name>/pkm/... -> hop to the primary checkout
    parts = AGENT_DIR.parts
    if ".claude" in parts:
        i = parts.index(".claude")
        rel = Path(*parts[i + 3 :])  # strip .claude/worktrees/<name>
        primary = Path(*parts[:i]) / rel / "out" / "latest.pth"
        if primary.exists():
            return primary
    return local


def _real_inputs(deck: list[int]) -> tuple[net.SparseVector, net.SparseVector]:
    """Build encoder+decoder sparse vectors from a real initial observation.

    Mirrors ``net.create_node``'s action enumeration so the decoder input has a
    representative number of option-combination "words".
    """
    obs_dict, _ = battle_start(deck, deck)
    try:
        obs = net.to_observation_class(obs_dict)
        indices = list(range(obs.select.maxCount))
        actions: list[list[int]] = []
        for _ in range(64):
            actions.append(indices.copy())
            for i in range(len(indices)):
                index = len(indices) - i - 1
                if indices[index] < len(obs.select.option) - i - 1:
                    indices[index] += 1
                    for j in range(index + 1, len(indices)):
                        indices[j] = indices[j - 1] + 1
                    break
            else:
                break
        sv_enc = net.get_encoder_input(obs, deck)
        sv_dec = net.get_decoder_input(obs, actions)
        return sv_enc, sv_dec
    finally:
        battle_finish()


def _to_tensors(sv: net.SparseVector) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(sv.index, dtype=torch.int32),
        torch.tensor(sv.value, dtype=torch.float32),
        torch.tensor(sv.offset, dtype=torch.int32),
    )


def export_onnx(checkpoint: Path | None, onnx_path: Path) -> Path:
    # Netron only needs the graph; trace on CPU for a clean, portable export.
    device = torch.device("cpu")
    if checkpoint is not None and checkpoint.exists():
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        dims = tuple(blob.get("dims", net.MODEL_DIMS))
        model = net.MyModel(*dims).to(device)
        model.load_state_dict(blob["state_dict"])
        print(f"loaded {checkpoint} (dims={dims})", flush=True)
    else:
        dims = net.MODEL_DIMS
        model = net.build_model(dims, device)
        print(f"no checkpoint at {checkpoint}; exporting a fresh net (dims={dims})", flush=True)
    model.eval()

    sv_enc, sv_dec = _real_inputs(net.sample_deck)
    ei, ev, eo = _to_tensors(sv_enc)
    di, dv, do = _to_tensors(sv_dec)
    print(f"encoder words={len(sv_enc.offset)} nnz={len(sv_enc.index)}; "
          f"decoder words={len(sv_dec.offset)} nnz={len(sv_dec.index)}", flush=True)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (ei, ev, eo, di, dv, do),
        str(onnx_path),
        input_names=[
            "enc_index", "enc_value", "enc_offset",
            "dec_index", "dec_value", "dec_offset",
        ],
        output_names=["value", "policy"],
        opset_version=17,
        dynamo=False,
    )
    print(f"exported ONNX -> {onnx_path}", flush=True)
    return onnx_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_default_checkpoint(),
        help="checkpoint to visualize (default: out/latest.pth; fresh net if missing)",
    )
    ap.add_argument(
        "--onnx",
        type=Path,
        default=AGENT_DIR / "out" / "agent_001_transformer.onnx",
        help="ONNX output path",
    )
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-serve", action="store_true", help="export only, don't launch Netron")
    args = ap.parse_args()

    onnx_path = export_onnx(args.checkpoint, args.onnx)

    if args.no_serve:
        return

    import netron

    print(f"serving Netron at http://{args.host}:{args.port} (Ctrl-C to stop)", flush=True)
    netron.start(str(onnx_path), address=(args.host, args.port), browse=False)
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        netron.stop()
        print("\nnetron stopped.", flush=True)


if __name__ == "__main__":
    main()
