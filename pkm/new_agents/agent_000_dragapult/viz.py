"""Visualize the PolicyValueModel five ways: torchinfo summary, torchview,
torchviz, ONNX (for Netron), and TensorBoard add_graph.

Run:  python -m pkm.new_agents.agent_000_dragapult.viz [outdir]

The model's forward() takes a dict, but torchview/torchviz/ONNX want positional
tensor args, so we wrap it in a module with an explicit tensor signature that
rebuilds the dict internally. All five tools consume the *same* synthetic batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pkm.new_agents.agent_000_dragapult.features import F, G, O, _VOCAB
from pkm.new_agents.agent_000_dragapult.model import PolicyValueModel

# --- fixed shapes for the dummy batch ---
B = 2  # batch
E = 12  # board entities
L = 5  # legal options (variable at runtime; fixed here for a static graph)

# The exact key order the wrapper's positional args map to (and back to a dict).
KEYS = [
    "entity_id_row",  # long  [B,E]
    "entity_card_id",  # long  [B,E]
    "entity_feat",  # float [B,E,F]
    "entity_mask",  # float [B,E]
    "hand_hist",  # float [B,27]
    "discard_hist",  # float [B,27]
    "globals",  # float [B,G]
    "option_type",  # long  [B,L]
    "option_feat",  # float [B,L,O]
    "option_mask",  # float [B,L]
    "select_type",  # long  [B]
    "select_context",  # long  [B]
]


def synthetic_batch() -> dict[str, torch.Tensor]:
    """A shape-correct, index-safe dummy batch (no engine needed)."""
    g = torch.Generator().manual_seed(0)

    def randint(hi: int, *shape: int) -> torch.Tensor:
        return torch.randint(0, hi, shape, generator=g)

    return {
        "entity_id_row": randint(_VOCAB, B, E),
        "entity_card_id": torch.zeros(B, E, dtype=torch.long),  # 0 = safe attr row
        "entity_feat": torch.rand(B, E, F, generator=g),
        "entity_mask": torch.ones(B, E),
        "hand_hist": torch.rand(B, _VOCAB, generator=g),
        "discard_hist": torch.rand(B, _VOCAB, generator=g),
        "globals": torch.rand(B, G, generator=g),
        "option_type": randint(17, B, L),  # N_OPTION_TYPES
        "option_feat": torch.rand(B, L, O, generator=g),
        "option_mask": torch.ones(B, L),
        "select_type": randint(11, B),  # N_SELECT_TYPES
        "select_context": randint(49, B),  # N_SELECT_CTX
    }


class TensorWrapper(nn.Module):
    """Positional-tensor facade over the dict-taking PolicyValueModel."""

    def __init__(self, model: PolicyValueModel):
        super().__init__()
        self.model = model

    def forward(self, *args: torch.Tensor):
        b = dict(zip(KEYS, args, strict=True))
        logits, value = self.model(b)
        return logits, value


def main(outdir: str = "viz_out") -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model = PolicyValueModel().eval()
    wrapper = TensorWrapper(model).eval()
    batch = synthetic_batch()
    args = tuple(batch[k] for k in KEYS)

    # 1) torchinfo — text table of layers + param counts
    from torchinfo import summary

    info = summary(
        wrapper,
        input_data=list(args),
        depth=5,
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    (out / "summary.txt").write_text(str(info))
    print(f"[1/5] torchinfo   -> {out / 'summary.txt'}")

    # 2) torchview — module-level architecture graph (Graphviz)
    from torchview import draw_graph

    gv = draw_graph(
        wrapper,
        input_data=list(args),
        graph_name="PolicyValueModel",
        depth=4,
        expand_nested=True,
        save_graph=False,
    )
    gv.visual_graph.render(filename=str(out / "torchview"), format="svg", cleanup=True)
    print(f"[2/5] torchview   -> {out / 'torchview.svg'}")

    # 3) torchviz — autograd op graph from the policy loss
    from torchviz import make_dot

    logits, value = wrapper(*args)
    loss = logits.sum() + value.sum()
    dot = make_dot(loss, params=dict(wrapper.named_parameters()))
    dot.render(filename=str(out / "torchviz"), format="svg", cleanup=True)
    print(f"[3/5] torchviz    -> {out / 'torchviz.svg'}")

    # 4) ONNX export — open the .onnx in Netron (netron file.onnx / netron.start)
    onnx_path = out / "model.onnx"
    # Legacy TorchScript exporter (dynamo=False) at opset 17 — opset 17 supports the
    # MultiheadAttention transpose/unsqueeze the encoder uses; do NOT pre-trace (a
    # pre-trace loses rank info and the exporter then rejects the transpose).
    torch.onnx.export(
        wrapper,
        args,
        str(onnx_path),
        input_names=KEYS,
        output_names=["option_logits", "value"],
        opset_version=17,
        dynamo=False,
    )
    print(f"[4/5] onnx/netron -> {onnx_path}   (view: netron {onnx_path})")

    # 5) TensorBoard — graph tab (tensorboard --logdir runs/viz)
    from torch.utils.tensorboard import SummaryWriter

    tb_dir = out / "tb"
    # add_graph traces internally with check_trace=True and exposes no way to turn
    # it off (its use_strict_trace maps to trace's *strict*, a different flag). The
    # re-trace comparison spuriously fails on this model — module __torch_mangle
    # renaming + value renumbering, not a numeric divergence — so force
    # check_trace=False on the trace it runs.
    orig_trace = torch.jit.trace

    def _trace_no_check(*a, **k):
        k["check_trace"] = False
        return orig_trace(*a, **k)

    torch.jit.trace = _trace_no_check
    try:
        with SummaryWriter(log_dir=str(tb_dir)) as w:
            w.add_graph(wrapper, args)
    finally:
        torch.jit.trace = orig_trace
    print(f"[5/5] tensorboard -> {tb_dir}   (view: tensorboard --logdir {tb_dir})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "viz_out")
