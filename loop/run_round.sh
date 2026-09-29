#!/usr/bin/env bash
# GB10-Engine round driver.
#
# One "round" = build + test + correctness gate + (optionally) benchmark, then
# commit and push. A round that fails a gate is still recorded, but is marked
# FAIL and does not advance the milestone.
#
# Usage:
#   loop/run_round.sh                 # full round
#   loop/run_round.sh --no-push       # skip the git push
#   loop/run_round.sh --quick         # build + test only (no bench)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STATE="$ROOT/loop/state.json"
ROUNDS="$ROOT/loop/rounds"
RESULTS="$ROOT/bench/results"
PUSH=1
QUICK=0

for arg in "$@"; do
  case "$arg" in
    --no-push) PUSH=0 ;;
    --quick)   QUICK=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$ROUNDS" "$RESULTS" "$ROOT/loop/logs"

ROUND=$(python3 -c "import json;print(json.load(open('$STATE'))['round'])" 2>/dev/null || echo 0)
ROUND=$((ROUND + 1))
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/loop/logs/round-$(printf '%03d' "$ROUND")-$STAMP.log"
REPORT="$ROUNDS/$(printf '%03d' "$ROUND")-round.md"

say() { echo "[round $ROUND] $*" | tee -a "$LOG"; }

GATE_BUILD=skip; GATE_TEST=skip; GATE_CORRECT=skip; GATE_GENERATE=skip; GATE_LONGCTX=skip; GATE_TCPARITY=skip; GATE_DECODE=skip; GATE_PREFIX=skip; GATE_BENCH=skip
STATUS=FAIL

{
  echo "# Round $ROUND — $STAMP"
  echo
} > "$REPORT"

# ---------------------------------------------------------------- 1. build ---
say "cargo build --release"
if cargo build --release --workspace >>"$LOG" 2>&1; then
  GATE_BUILD=pass; say "build OK"
else
  GATE_BUILD=fail; say "build FAILED (see $LOG)"
fi

# ----------------------------------------------------------------- 2. test ---
if [ "$GATE_BUILD" = pass ]; then
  say "cargo test --workspace"
  if cargo test --workspace --release >>"$LOG" 2>&1; then
    GATE_TEST=pass; say "tests OK"
  else
    GATE_TEST=fail; say "tests FAILED (see $LOG)"
  fi
fi

# ------------------------------------------------- 3. correctness vs oracle ---
if [ "$GATE_TEST" = pass ]; then
  if [ -x "$ROOT/target/release/gb10-verify" ]; then
    say "gb10-verify (token-exact vs HF oracle)"
    if "$ROOT/target/release/gb10-verify" --oracle "$ROOT/fixtures/oracle" \
         --model "$ROOT/models/Qwen3.8-27B-NVFP4" >>"$LOG" 2>&1; then
      GATE_CORRECT=pass; say "correctness OK"
    else
      GATE_CORRECT=fail; say "correctness FAILED (see $LOG)"
    fi
  else
    GATE_CORRECT=missing; say "gb10-verify not built yet — gate pending"
  GATE_BATCH=missing
  fi
fi

# ------------------------------------- 3b. end-to-end 64-layer generate gate ---
# The layer gate proves each block; this proves the stack (embedding gather,
# 64 layers wired in order, final norm, NVFP4 lm_head, argmax) against a
# full-model oracle.
if [ "$GATE_TEST" = pass ]; then
  if [ -x "$ROOT/target/release/gb10-verify" ] && [ -f "$ROOT/fixtures/oracle/greedy_tokens.json" ]; then
    say "gb10-verify generate (64-layer greedy decode vs full-model oracle)"
    # --repeat makes the same prefill+decode run 8 times from fresh state and
    # fails the gate if any repeat differs. One run cannot distinguish a
    # deterministic path from a lucky one; this can, and it costs ~18s.
    if "$ROOT/target/release/gb10-verify" generate --n 16 --repeat 8 \
         --oracle "$ROOT/fixtures/oracle" --model "$ROOT/models/Qwen3.8-27B-NVFP4" >>"$LOG" 2>&1; then
      GATE_GENERATE=pass; say "generate OK"
    else
      GATE_GENERATE=fail; say "generate FAILED (see $LOG)"
    fi
  else
    GATE_GENERATE=missing; say "no full-model oracle at fixtures/oracle/greedy_tokens.json — gate pending"
  fi
fi

# --------------------------------- 3b-ii. long-context follow gate ---
# Every gate above is short-prompt, and that is exactly why round 282's
# tensor-core prefill GEMM could pass all of them (generate 16/16, batch-parity
# 16/16, PPL within 0.023%) while breaking the engine completely: past ~970
# prompt tokens, through the chat template, one flipped argmax made the model
# emit EOS as its first token and answer nothing at all, for 126 rounds.
# This gate is deliberately a long prompt through the real template; it asserts
# only that the model generates SOMETHING. See bench/longctx/longctx_gate.sh.
if [ "$GATE_TEST" = pass ]; then
  if [ -x "$ROOT/bench/longctx/longctx_gate.sh" ]; then
    say "longctx-follow (long prompt + chat template; catches immediate EOS)"
    if "$ROOT/bench/longctx/longctx_gate.sh" "$ROOT" >>"$LOG" 2>&1; then
      GATE_LONGCTX=pass; say "longctx-follow OK"
    else
      GATE_LONGCTX=fail; say "longctx-follow FAILED (see $LOG)"
    fi
  else
    GATE_LONGCTX=missing; say "no bench/longctx/longctx_gate.sh — gate pending"
  fi

  # `longctx-follow` above catches the SYMPTOM of a broken prefill GEMM; this
  # catches the arithmetic itself. The tensor-core prefill GEMM was silently
  # truncating every element past 16,776,960 because eight element-wise kernels
  # had no grid-stride loop under a 65535-block cap -- invisible at t=512
  # (mathematically exact), fatal from t=964. A long prompt caught that; this
  # gate localises it without a prompt, a template or sampling in the way.
  # FAILS if the tensor-core path and the fp32 reference disagree on real
  # weights at a length past the cliff.
  say "tc-parity (tensor-core prefill GEMM vs fp32 reference, real weights)"
  if ./target/release/gb10-bench tc-parity --model "$ROOT/models/Qwen3.8-27B-NVFP4" \
       >>"$LOG" 2>&1; then
    GATE_TCPARITY=pass; say "tc-parity OK"
  else
    GATE_TCPARITY=fail; say "tc-parity FAILED (see $LOG)"
  fi
fi


# The generate gate above is single-sequence. This one proves that N sequences
# decoded together in one pass give token-exact results, which is what the
# concurrency target depends on. It uses prompts of *different* lengths: with
# equal lengths every sequence sits at the same position, so a per-sequence
# addressing bug would read equivalent data and pass.
if [ "$GATE_TEST" = pass ]; then
  if [ -x "$ROOT/target/release/gb10-verify" ]; then
    say "gb10-verify batch-parity (16 sequences, token-exact vs one-at-a-time)"
    if "$ROOT/target/release/gb10-verify" batch-parity --n-seq 16 --n 16 \
         --model "$ROOT/models/Qwen3.8-27B-NVFP4" >>"$LOG" 2>&1; then
      GATE_BATCH=pass; say "batch-parity OK"
      grep -h "batched decode:" "$LOG" | tail -1
    else
      GATE_BATCH=fail; say "batch-parity FAILED (see $LOG)"
    fi
  else
    GATE_BATCH=missing; say "gb10-verify not built yet — batch gate pending"
  fi
fi

# The server's prefix cache is invisible to every gate above: they drive the
# model directly and never go through the server's resume path. So this runs
# one request sequence twice -- once against a caching server, once against a
# server started with --no-prefix-cache -- and requires byte-identical output.
if [ "$GATE_BUILD" = pass ] && [ -x "$ROOT/target/release/gb10-server" ]; then
  say "prefix cache A/B (caching vs --no-prefix-cache)"
  if "$ROOT/bench/longctx/prefix_ab.sh" >>"$LOG" 2>&1; then
    GATE_PREFIX=pass; say "prefix cache OK"
  else
    GATE_PREFIX=fail; say "prefix cache FAILED (see $LOG)"
  fi
fi

# `decode-bench` checks the warp-parallel decode kernel against the serial
# reference and times both. It is not redundant with `batch-parity`: that gate
# compares batched against one-at-a-time, and both of those now take the warp
# path, so a bug inside the warp kernel would cancel out of it.
if [ "$GATE_TEST" = pass ]; then
  if [ -x "$ROOT/target/release/gb10-verify" ]; then
    say "gb10-verify decode-bench (warp vs serial reference)"
    if "$ROOT/target/release/gb10-verify" decode-bench --n-seq 4 \
         --kv-keys 8192 --model "$ROOT/models/Qwen3.8-27B-NVFP4" >>"$LOG" 2>&1; then
      GATE_DECODE=pass; say "decode-bench OK"
      grep -h "8192" "$LOG" | tail -1
    else
      GATE_DECODE=fail; say "decode-bench FAILED (see $LOG)"
    fi
  else
    GATE_DECODE=missing; say "gb10-verify not built yet — decode gate pending"
  fi
fi

# ------------------------------------------------------------ 4. benchmark ---
if [ "$QUICK" = 0 ] && [ "$GATE_BUILD" = pass ] && [ -x "$ROOT/target/release/gb10-bench" ]; then
  say "gb10-bench"
  if "$ROOT/target/release/gb10-bench" stream --out "$RESULTS/$STAMP.json" >>"$LOG" 2>&1; then
    GATE_BENCH=pass; say "bench OK -> $RESULTS/$STAMP.json"
  else
    GATE_BENCH=fail; say "bench FAILED (see $LOG)"
  fi
fi

if [ "$GATE_BUILD" = pass ] && [ "$GATE_TEST" = pass ] &&
   { [ "$GATE_CORRECT" = pass ] || [ "$GATE_CORRECT" = missing ]; } &&
   { [ "$GATE_GENERATE" = pass ] || [ "$GATE_GENERATE" = missing ]; } &&
   { [ "$GATE_LONGCTX" = pass ] || [ "$GATE_LONGCTX" = missing ]; } &&
   { [ "$GATE_DECODE" = pass ] || [ "$GATE_DECODE" = missing ]; } &&
   { [ "$GATE_PREFIX" = pass ] || [ "$GATE_PREFIX" = missing ]; } &&
   { [ "$GATE_BENCH" = pass ] || [ "$GATE_BENCH" = skip ]; }; then
  STATUS=PASS
fi

# --------------------------------------------------------------- 5. report ---
{
  echo "## Gates"
  echo
  echo "| gate | result |"
  echo "|---|---|"
  echo "| build | $GATE_BUILD |"
  echo "| test | $GATE_TEST |"
  echo "| correctness (layers) | $GATE_CORRECT |"
  echo "| correctness (64-layer) | $GATE_GENERATE |"
  echo "| longctx-follow (long prompt, no silent EOS) | $GATE_LONGCTX |"
  echo "| tc-parity (prefill GEMM vs fp32, real weights) | $GATE_TCPARITY |"
  echo "| decode-bench (warp vs serial) | $GATE_DECODE |"
  echo "| prefix cache A/B | $GATE_PREFIX |"
  echo "| benchmark | $GATE_BENCH |"
  echo
  echo "**status: $STATUS**"
  echo
  echo "## Log"
  echo
  echo '```'
  tail -n 40 "$LOG"
  echo '```'
} >> "$REPORT"

# ---------------------------------------------------------------- 6. state ---
python3 - "$STATE" "$ROUND" "$STATUS" "$STAMP" "$GATE_BUILD" "$GATE_TEST" "$GATE_CORRECT" "$GATE_GENERATE" "$GATE_LONGCTX" "$GATE_TCPARITY" "$GATE_DECODE" "$GATE_PREFIX" "$GATE_BENCH" <<'PY'
import json, sys
state_path, rnd, status, stamp, b, t, c, g, lc, tp, dc, pf, bm = sys.argv[1:14]
s = json.load(open(state_path))
s["round"] = int(rnd)
s["last_round_at"] = stamp
s["last_status"] = status
s.setdefault("history", []).append(
    {"round": int(rnd), "at": stamp, "status": status,
     "gates": {"build": b, "test": t, "correctness": c, "generate": g,
               "longctx": lc, "tcparity": tp, "decode": dc, "prefix": pf, "benchmark": bm}}
)
json.dump(s, open(state_path, "w"), indent=2)
print(f"state: round={rnd} status={status}")
PY

# ------------------------------------------------------------- 7. git push ---
if [ "$PUSH" = 1 ] && [ -d "$ROOT/.git" ]; then
  say "committing and pushing"
  git add -A >>"$LOG" 2>&1
  if ! git diff --cached --quiet; then
    git commit -q -m "round $ROUND: $STATUS

gates: build=$GATE_BUILD test=$GATE_TEST correctness=$GATE_CORRECT generate=$GATE_GENERATE longctx=$GATE_LONGCTX tcparity=$GATE_TCPARITY bench=$GATE_BENCH

Automated round by loop/run_round.sh" >>"$LOG" 2>&1
  fi
  if git remote get-url origin >/dev/null 2>&1; then
    if git push -q origin HEAD >>"$LOG" 2>&1; then
      say "pushed"
    else
      say "PUSH FAILED (see $LOG)"
    fi
  else
    say "no origin remote; skipped push"
  fi
fi

say "round $ROUND finished: $STATUS"
echo "$REPORT"
[ "$STATUS" = PASS ]
