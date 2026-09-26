#!/usr/bin/env bash
# Score MMLU through llama.cpp's server, the other NVFP4 implementation here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=tools/llama.cpp/build/bin
PY=/home/wayne/KimiCode/Deploy_QWen3.8-27B/oracle/venv/bin/python
PORT=8899
# n_probs must be wide enough that all four answer letters are always in the
# returned candidate list. At 100 they were not (130/3240 questions, mostly
# philosophy); at 1000 they are. OUT is parameterised so a run at one width
# never silently overwrites a run at another.
NPROBS=${NPROBS:-1000}
OUT=${OUT:-bench/mmlu/llamacpp.json}

cleanup() { kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true; }
trap cleanup EXIT

echo "starting llama-server on :$PORT"
tools/llama.cpp/build/bin/llama-server \
    -m models/Qwen3.8-27B-NVFP4.gguf \
    --host 127.0.0.1 --port "$PORT" -c 2048 -np 1 \
    > bench/mmlu/llamacpp-server.log 2>&1 &
SRV=$!

for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "server up after $((i * 2))s"
        break
    fi
    if ! kill -0 "$SRV" 2>/dev/null; then
        echo "server died; last log lines:" >&2
        tail -20 bench/mmlu/llamacpp-server.log >&2
        exit 1
    fi
    sleep 2
done

if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "server never became healthy" >&2
    tail -20 bench/mmlu/llamacpp-server.log >&2
    exit 1
fi

$PY bench/mmlu/llamacpp_mmlu.py \
    --url "http://127.0.0.1:$PORT" \
    --jsonl bench/mmlu/mmlu.jsonl \
    --engine-json bench/mmlu/engine-mmlu.json \
    --n-probs "$NPROBS" \
    --out "$OUT" 2>&1 | tee "${OUT%.json}.log"
