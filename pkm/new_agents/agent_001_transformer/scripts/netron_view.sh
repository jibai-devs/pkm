#!/usr/bin/env bash
# Export agent_001_transformer's net to ONNX and serve it in Netron.
# Usage: bash scripts/netron_view.sh [PORT] [-- <extra python args>]
set -euo pipefail

# NixOS libcuda path (harmless elsewhere); export runs on CPU anyway.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PORT="${1:-8080}"
shift || true

cd "$REPO_ROOT"
exec uv run python -m pkm.new_agents.agent_001_transformer.scripts.netron_view \
    --port "$PORT" "$@"
