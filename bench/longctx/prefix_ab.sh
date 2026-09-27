#!/usr/bin/env bash
# Does the prefix cache change the answer?
#
# Runs one identical request sequence against two servers -- one that resumes
# from the cached prefix, one that refuses every resume -- and diffs the text.
# The outputs are unlabelled, so equality here is the cache's whole correctness
# contract; nothing else tests it, because the gates in gb10-verify drive the
# model directly and never go through the server's cache at all.
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=/home/wayne/KimiCode/Deploy_QWen3.8-27B/oracle/venv/bin/python
REPS=${REPS:-20}
CTX=${CTX:-4096}
PORT=8080
MODEL=models/Qwen3.8-27B-NVFP4

run_one() {
    local flag="$1" tag="$2"
    pkill -x gb10-server 2>/dev/null
    sleep 3
    # shellcheck disable=SC2086
    ./target/release/gb10-server --model "$MODEL" --port "$PORT" --ctx "$CTX" $flag \
        > "/tmp/longctx/gb10-$tag.log" 2>&1 &
    for _ in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 3
    done
    "$PY" bench/longctx/prefix_check.py --port "$PORT" --reps "$REPS" >/dev/null
    cp "/tmp/longctx/prefix-$PORT.json" "/tmp/longctx/ab-$tag.json"
}

echo "=== caching server ==="
run_one "" cache
echo "=== non-caching server (--no-prefix-cache) ==="
run_one "--no-prefix-cache" nocache
pkill -x gb10-server 2>/dev/null

if diff -q /tmp/longctx/ab-cache.json /tmp/longctx/ab-nocache.json >/dev/null; then
    echo
    echo "PASS: cache and no-cache produce identical output"
    "$PY" - <<'EOF'
import json
d = json.load(open("/tmp/longctx/ab-cache.json"))
for k in ("a1", "b", "c"):
    print(f"  {k}: {d[k][:60]!r}")
EOF
    exit 0
else
    echo
    echo "FAIL: outputs differ"
    diff /tmp/longctx/ab-cache.json /tmp/longctx/ab-nocache.json | head -20
    exit 1
fi
