#!/bin/bash
# Needle-in-a-haystack validation at 32K / 128K / 256K, against a live server.
#
# Launch it as a systemd *user* unit, not as a session subprocess:
#
#   systemd-run --user --unit=gb10-longctx --collect \
#       --working-directory=/home/wayne/dsh/QWen3.8-27B-GB10 \
#       /home/wayne/dsh/QWen3.8-27B-GB10/bench/longctx/run_validation.sh
#
# Why: the first attempt ran under `setsid nohup` and still died 96 min in. The
# harness tracks a subprocess *scope*, not a session, and reaped it --
#
#   dsh-subprocess-191061-e0db7c338b98.scope: Sending signal SIGTERM to
#   process 193058 (gb10-server) on client request.
#
# -- which is long enough for the 32K leg but nowhere near the 128K or 256K
# legs. A user unit lives in the user manager's own scope, so the reaping does
# not reach it. Progress goes to files, since this outlives any single turn.
set -u
cd /home/wayne/dsh/QWen3.8-27B-GB10 || exit 1

LOG=/tmp/longctx
PY=/home/wayne/KimiCode/Deploy_QWen3.8-27B/oracle/venv/bin/python
NEEDLE="$PWD/bench/longctx/needle.py"
SERVER="$PWD/target/release/gb10-server"

mkdir -p "$LOG"
: > "$LOG/results.log"

say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG/results.log"; }

# One server per leg, sized just past that leg's prompt. The KV cache is 128 KB
# per token, so a 256K server reserves 34 GB whether or not the leg needs it;
# sizing each leg separately keeps the early ones cheap and makes the tight
# 68 GB startup happen once instead of three times.
leg() {
    local name="$1" ctx="$2" reps="$3" depths="$4"

    say "=== $name  (ctx $ctx, reps $reps, depths $depths) ==="
    pkill -x gb10-server 2>/dev/null
    sleep 8

    "$SERVER" --model models/Qwen3.8-27B-NVFP4 --port 8080 --ctx "$ctx" \
        > "$LOG/server-$name.log" 2>&1 &
    local srv=$!

    local up=0
    for _ in $(seq 1 180); do
        if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then up=1; break; fi
        sleep 5
    done
    if [ "$up" = 0 ]; then
        say "!! $name: server never became ready; see server-$name.log"
        return 1
    fi
    say "$name: server ready"
    grep -E '^context ' "$LOG/server-$name.log" >> "$LOG/results.log" 2>/dev/null

    local t0=$SECONDS
    DEPTHS="$depths" "$PY" "$NEEDLE" "$reps" >> "$LOG/results.log" 2>&1
    say "$name: needle exit $? after $((SECONDS - t0))s"

    kill "$srv" 2>/dev/null
    wait "$srv" 2>/dev/null
    sleep 8
}

# ~31 tokens per filler repeat, so 1054 -> 32K, 4214 -> 128K, 8429 -> 256K.
# 32K gets all three depths because it is cheap; the longer legs get the middle
# depth only, which is the one a recency effect cannot carry.
leg 32K  36864  1054 "0.1,0.5,0.9"
leg 128K 139264 4214 "0.5"
leg 256K 262144 8429 "0.5"

say "DONE"
