#!/bin/bash
# Long-context instruction-following gate.
#
# WHY THIS EXISTS
# ---------------
# Round 282 turned the tensor-core prefill GEMM on by default (`GB10_TC_GEMM`)
# for a 2.48x cold-TTFT win. It passed every gate the round loop had:
#
#   * `generate` 16/16 token-exact against the frozen HF oracle
#   * `batch-parity` 16/16 over 16 sequences
#   * perplexity within 0.023% of fp32 at a 512-token window
#
# ...and it still broke the engine completely. All of those gates use SHORT
# prompts. Past roughly 970 prompt tokens, and only through the chat template,
# the reduced-precision GEMM flipped one greedy argmax at the first generated
# position to EOS: the model then answered nothing at all, for every long
# prompt, while llama.cpp answered the same prompts correctly. It cost the
# long-context needle validation and went unnoticed for 126 rounds.
#
# So this gate is deliberately the opposite of the other gates: a LONG prompt,
# through the real chat template, asserting only that the model generates
# SOMETHING. It does not check the answer -- "ignore the document, say hello"
# is trivially checkable, but the bug it catches is not a wrong answer, it is a
# model that refuses to speak. Anything non-empty passes; immediate EOS fails.
#
# It is server-free (no port, no scheduler) and takes one model load plus one
# prefill, so it is cheap enough to run every round.
#
# Usage: bench/longctx/longctx_gate.sh [root]
# Exit: 0 = the model generated tokens, 1 = immediate EOS (regression), 2 = setup error.

set -u
ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="$ROOT/target/release/gb10-verify"

if [ ! -x "$BIN" ]; then
  echo "longctx-follow: gb10-verify not built — gate pending"
  exit 2
fi

# ~1000 tokens of filler, then the instruction. The length is the point: below
# ~970 tokens this bug is invisible, which is exactly why the short gates missed
# it. Do not shorten this to make the gate faster.
FILLER="The archive room contains many boxes of old records. Each box is labelled with a number and a date, and the shelves are dusted every second Tuesday. "
PROMPT=""
for _ in $(seq 1 30); do PROMPT="$PROMPT$FILLER"; done
PROMPT="$PROMPT

Ignore the document. Reply with the single word: hello"

# `--fixtures` deliberately points at a directory that does not exist: when
# fixtures/oracle/greedy_tokens.json is present it OVERRIDES --prompt with the
# oracle's own (short) prompt, which would silently turn this back into a short
# gate that cannot see the bug.
EMPTY_FIX="$(mktemp -d)"
trap 'rm -rf "$EMPTY_FIX"' EXIT

OUT="$("$BIN" generate --model "$ROOT/models/Qwen3.8-27B-NVFP4" \
        --fixtures "$EMPTY_FIX" --max-seq 2048 \
        --prompt "$PROMPT" --n 4 2>&1)"
RC=$?
echo "$OUT" | sed -n 's/^prompt: /longctx-follow: prompt /p'

if [ $RC -ne 0 ]; then
  echo "longctx-follow: FAILED to run (exit $RC)"
  echo "$OUT" | tail -5
  exit 2
fi

# `generate` prints the greedy ids it produced; an empty list is the bug.
IDS="$(echo "$OUT" | sed -n 's/^ids:  \[\(.*\)\]$/\1/p')"
if [ -z "${IDS// /}" ]; then
  echo "longctx-follow: FAIL — model generated 0 tokens (immediate EOS at long context)"
  echo "$OUT" | tail -4
  exit 1
fi

echo "longctx-follow: OK — $(echo "$IDS" | tr ',' ' ' | wc -w) tokens generated at long context"
exit 0
