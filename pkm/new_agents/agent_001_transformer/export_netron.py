"""Export the agent_001_transformer net to ONNX and open it in Netron.

Netron renders a model graph from an on-disk file. PyTorch models aren't a file
format Netron reads directly, so we first trace the net to **ONNX** (a portable
graph format), then hand that file to Netron's built-in web server.

The net's ``forward`` takes six sparse tensors (the two ``EmbeddingBag`` inputs —
index / per-sample-value / offsets, for the encoder and decoder). We synthesize
a tiny but *shape-valid* example so the tracer can walk the whole graph:

  * encoder: 24 bags (``num_words_encoder``), one active feature each, so the
    ``reshape(-1, 24, d_model)`` sees batch=1;
  * decoder: 5 action "combination" bags, one active feature each.

Run via ``scripts/netron.sh`` (sets the NixOS libcuda path, uses the repo env) or
directly::

    python -m pkm.new_agents.agent_001_transformer.export_netron \
        --checkpoint pkm/new_agents/agent_001_transformer/out/latest.pth \
        --serve
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pkm.new_agents.agent_001_transformer import net


def _example_inputs(n_dec_combos: int = 5):
    """Shape-valid dummy inputs for the six-tensor forward().

    EmbeddingBag(offsets) convention: ``offsets[k]`` is the start of bag k in the
    flat ``index``/``value`` arrays, so ``len(offsets)`` == number of bags. We
    give every bag exactly one active feature (index k, value 1.0).
    """
    n_enc_bags = net.num_words_encoder  # 24 -> reshape sees batch=1

    def one_per_bag(n_bags: int, vocab: int):
        index = torch.arange(n_bags, dtype=torch.int32) % vocab
        value = torch.ones(n_bags, dtype=torch.float32)
        offset = torch.arange(n_bags, dtype=torch.int32)  # bag k starts at k
        return index, value, offset

    ei, ev, eo = one_per_bag(n_enc_bags, net.encoder_size)
    di, dv, do = one_per_bag(n_dec_combos, net.decoder_size)
    return (ei, ev, eo, di, dv, do)


def export(checkpoint: Path | None, out_path: Path, opset: int = 17) -> Path:
    device = torch.device("cpu")  # export is CPU-only; no GPU needed
    if checkpoint is not None and checkpoint.exists():
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        dims = tuple(blob.get("dims", net.MODEL_DIMS))
        model = net.MyModel(*dims).to(device)
        model.load_state_dict(blob["state_dict"])
        print(f"loaded checkpoint {checkpoint} (dims={dims})", flush=True)
    else:
        dims = net.MODEL_DIMS
        model = net.build_model(dims, device)
        print(f"no checkpoint — exporting a FRESH net (dims={dims})", flush=True)
    model.eval()

    args = _example_inputs()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    input_names = [
        "enc_index", "enc_value", "enc_offset",
        "dec_index", "dec_value", "dec_offset",
    ]
    torch.onnx.export(
        model,
        args,
        str(out_path),
        input_names=input_names,
        output_names=["value", "policy"],
        opset_version=opset,
        dynamo=False,  # legacy tracer handles EmbeddingBag + MHA more reliably
    )
    print(f"exported ONNX -> {out_path}", flush=True)
    return out_path


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=here / "out" / "latest.pth")
    ap.add_argument("--out", type=Path, default=here / "out" / "model.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--serve", action="store_true", help="open the graph in Netron after export")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    onnx_path = export(args.checkpoint, args.out, args.opset)

    if args.serve:
        import netron

        print(f"serving {onnx_path} at http://{args.host}:{args.port}", flush=True)
        netron.start(str(onnx_path), address=(args.host, args.port), browse=False)
        print("Ctrl-C to stop.", flush=True)
        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            netron.stop()
            print("\nstopped.", flush=True)


if __name__ == "__main__":
    main()
