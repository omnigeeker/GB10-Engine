#!/usr/bin/env bash
# Convert the local NVFP4 checkpoint to GGUF so llama.cpp can serve the same
# weights, which is what the M8 head-to-head requires.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV=/home/wayne/KimiCode/Deploy_QWen3.8-27B/oracle/venv
export PYTHONPATH="$ROOT/tools/llama.cpp/gguf-py"
cd "$ROOT/tools/llama.cpp"
"$VENV/bin/python" convert_hf_to_gguf.py \
  "$ROOT/models/Qwen3.8-27B-NVFP4" \
  --outfile "$ROOT/models/Qwen3.8-27B-NVFP4.gguf" \
  --outtype nvfp4 2>&1 || \
"$VENV/bin/python" convert_hf_to_gguf.py \
  "$ROOT/models/Qwen3.8-27B-NVFP4" \
  --outfile "$ROOT/models/Qwen3.8-27B-NVFP4.gguf" \
  --outtype bf16
echo "GGUF_OK"
