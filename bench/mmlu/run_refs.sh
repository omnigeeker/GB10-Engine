#!/usr/bin/env bash
# The two transformers references, run one after the other on purpose.
#
# Each holds a ~52 GB bf16 model in unified memory. Running one of these
# alongside the engine (~20 GB) is what OOM-killed the bf16 perplexity job in
# round 2, so they are serialised deliberately rather than in parallel.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/wayne/KimiCode/Deploy_QWen3.8-27B/oracle/venv/bin/python

echo "=== 1/2 bf16 base model (total quantization cost) ==="
$PY bench/mmlu/ref_mmlu.py \
    --mode bf16 --model-path models/Qwen3.8-27B-BF16 \
    --jsonl bench/mmlu/mmlu.jsonl \
    --engine-json bench/mmlu/engine-mmlu.json \
    --out bench/mmlu/ref-bf16.json 2>&1 | tee bench/mmlu/ref-bf16.log

echo
echo "=== 2/2 NVFP4 dequantized (same weights as the engine) ==="
GB10_MODEL_DIR=models/Qwen3.8-27B-NVFP4 $PY bench/mmlu/ref_mmlu.py \
    --mode dequant --model-path models/Qwen3.8-27B-NVFP4 \
    --jsonl bench/mmlu/mmlu.jsonl \
    --engine-json bench/mmlu/engine-mmlu.json \
    --out bench/mmlu/ref-nvfp4-dequant.json 2>&1 | tee bench/mmlu/ref-nvfp4-dequant.log

echo
echo "both references done"
