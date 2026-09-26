#!/bin/bash
# Long-context validation driver: start a 256K-context server, then run
# needle-in-a-haystack at 32K, 128K and 256K.
#
# Runs *detached* on purpose. The session's managed background jobs are cleaned
# up at goal-round boundaries, which kills anything longer than a round; the
# 256K leg alone is ~12 h, so this has to own its own lifecycle and report
# through files instead.
#
#   setsid nohup bench/longctx/run_validation.sh >/dev/null 2>&1 &
#
# Progress:  tail -f /tmp/longctx/results.log
# Finished:  grep -q '^DONE' /tmp/longctx/results.log
set -u
cd "$(dirname "$0")/../.." || exit 1

LOG=/tmp/longctx
mkdir -p "$LOG"
: > "$LOG/server.log"
: > "$LOG/results.log"
: > "$LOG/driver.log"

pkill -x gb10-server 2>/dev/null
sleep 3

# The checkpoint's native window; concurrency follows from the KV budget.
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --port 8080 --ctx 262144 \
    >> "$LOG/server.log" 2>&1 &
echo "server pid $!" >> "$LOG/driver.log"

for _ in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "server up" >> "$LOG/driver.log"
        break
    fi
    sleep 5
done

PY=/home/wayne/KimiCode/Deploy_QWen3.8-27B/oracle/venv/bin/python
N="$PWD/bench/longctx/needle.py"

# ~31 tokens per filler repeat.
run() {
    echo "=== $1 ===" >> "$LOG/results.log"
    DEPTHS="$2" "$PY" "$N" "$3" >> "$LOG/results.log" 2>&1
    echo "(exit $?)" >> "$LOG/results.log"
}

run "32K, depths 10/50/90%" "0.1,0.5,0.9" 1054
run "128K, depth 50%"     "0.5"           4214
run "256K, depth 50%"     "0.5"           8429

echo "DONE" >> "$LOG/results.log"
