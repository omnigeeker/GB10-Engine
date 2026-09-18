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

GATE_BUILD=skip; GATE_TEST=skip; GATE_CORRECT=skip; GATE_BENCH=skip
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
  fi
fi

# ------------------------------------------------------------ 4. benchmark ---
if [ "$QUICK" = 0 ] && [ "$GATE_BUILD" = pass ] && [ -x "$ROOT/target/release/gb10-bench" ]; then
  say "gb10-bench"
  if "$ROOT/target/release/gb10-bench" --out "$RESULTS/$STAMP.json" >>"$LOG" 2>&1; then
    GATE_BENCH=pass; say "bench OK -> $RESULTS/$STAMP.json"
  else
    GATE_BENCH=fail; say "bench FAILED (see $LOG)"
  fi
fi

if [ "$GATE_BUILD" = pass ] && [ "$GATE_TEST" = pass ] &&
   { [ "$GATE_CORRECT" = pass ] || [ "$GATE_CORRECT" = missing ]; } &&
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
  echo "| correctness | $GATE_CORRECT |"
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
python3 - "$STATE" "$ROUND" "$STATUS" "$STAMP" "$GATE_BUILD" "$GATE_TEST" "$GATE_CORRECT" "$GATE_BENCH" <<'PY'
import json, sys
state_path, rnd, status, stamp, b, t, c, bm = sys.argv[1:9]
s = json.load(open(state_path))
s["round"] = int(rnd)
s["last_round_at"] = stamp
s["last_status"] = status
s.setdefault("history", []).append(
    {"round": int(rnd), "at": stamp, "status": status,
     "gates": {"build": b, "test": t, "correctness": c, "benchmark": bm}}
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

gates: build=$GATE_BUILD test=$GATE_TEST correctness=$GATE_CORRECT bench=$GATE_BENCH

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
