#!/usr/bin/env bash
# Export agent_001_transformer to ONNX and view it in Netron (in-browser graph).
#
# Usage (from anywhere):  bash scripts/netron.sh [CHECKPOINT] [PORT]
#   CHECKPOINT  {state_dict,dims} .pth to visualize (default ./out/latest.pth;
#               falls back to a fresh net if it doesn't exist)
#   PORT        Netron server port (default 8080)
#
# Writes ./out/model.onnx, then serves it at http://localhost:PORT. Ctrl-C stops.
set -euo pipefail

# NixOS: same libcuda shim as train.sh. Export runs on CPU, but importing torch
# still probes cuda; harmless to expose the driver path either way.
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$AGENT_DIR/../../.." && pwd)"

CKPT="${1:-$AGENT_DIR/out/latest.pth}"
PORT="${2:-8080}"

cd "$REPO_ROOT"
echo "exporting + serving agent_001_transformer via Netron (ckpt=$CKPT port=$PORT)" >&2
python -m pkm.new_agents.agent_001_transformer.export_netron \
    --checkpoint "$CKPT" \
    --out "$AGENT_DIR/out/model.onnx" \
    --port "$PORT" \
    --serve
