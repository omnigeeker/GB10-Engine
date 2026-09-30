#!/usr/bin/env bash
# Acceptance gates + memory-safety checks for the elementwise.ptx currently
# installed in OUT_DIR.
#
# Usage:  bench/longctx/fa2_gates.sh <tag> [outdir]
#
# `tag` names the arm (e.g. armA_control, armB_sr) so results from different
# kernels land in different files and can be diffed afterwards.
#
# Baseline values (documented, committed on main):
#   generate      EXACT 16/16 ids
#                 [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338,
#                  7285, 8722, 57879, 3296, 13, 21134]
#   perplexity --ctx 512  --chunks 60 -> mean NLL 1.875067  (ppl 6.52126)
#   perplexity --ctx 4096 --chunks 60 -> mean NLL 1.877331  (ppl 6.53604)
#   attn-tile returns rc=1 BY DESIGN (fp16 P*V accumulator) -- not a veto.
#
# The `attn-tile` stdout is compared byte-for-byte between arms: the staging
# change is meant to be bit-identical, so a diff there is a real regression,
# whereas a matching table is strong evidence of exactness.
set -u
ROOT=/home/wayne/dsh/QWen3.8-27B-GB10
cd "$ROOT" || exit 2
TAG=${1:?usage: fa2_gates.sh <tag> [outdir]}
OUT=${2:-/tmp/gates/$TAG}
mkdir -p "$OUT"
MODEL=models/Qwen3.8-27B-NVFP4
export GB10_FA2=1

# Record which kernel actually ran.
sha256sum target/release/build/gb10-cuda-*/out/elementwise.ptx | tee "$OUT/ptx.sha256"
cp target/release/build/gb10-cuda-*/out/elementwise.ptx "$OUT/elementwise.ptx"

run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "--- $name"
  "$@" >"$OUT/$name.txt" 2>&1
  echo "    rc=$? -> $OUT/$name.txt"
}

run generate ./target/release/gb10-verify generate --model "$MODEL" \
    --prompt "What is the capital of France?" --n 24
run ppl512 ./target/release/gb10-verify perplexity --model "$MODEL" \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw \
    --ctx 512 --chunks 60
run ppl4096 ./target/release/gb10-verify perplexity --model "$MODEL" \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw \
    --ctx 4096 --chunks 60
# rc=1 by design.
./target/release/gb10-verify attn-tile --model "$MODEL" >"$OUT/attn_tile.txt" 2>&1
echo "--- attn-tile rc=$? (1 is expected: fp16 P*V accumulator)"

# ---- memory safety: the staging loop is indexing code --------------------
run sanitizer_memcheck compute-sanitizer --tool memcheck --error-exitcode 9 \
    ./target/release/gb10-verify attn-tile --model "$MODEL"
run sanitizer_racecheck compute-sanitizer --tool racecheck --error-exitcode 9 \
    ./target/release/gb10-verify attn-tile --model "$MODEL"
run sanitizer_memcheck_generate compute-sanitizer --tool memcheck --error-exitcode 9 \
    ./target/release/gb10-verify generate --model "$MODEL" \
    --prompt "What is the capital of France?" --n 24

echo
echo "=== summary for $TAG ==="
grep -h "mean NLL\|ppl\|ids" "$OUT"/generate.txt "$OUT"/ppl512.txt "$OUT"/ppl4096.txt 2>/dev/null | head
for f in sanitizer_memcheck sanitizer_racecheck sanitizer_memcheck_generate; do
  printf '%-32s %s\n' "$f" "$(grep -cE 'ERROR SUMMARY: [1-9]|errors? found|Race' "$OUT/$f.txt" 2>/dev/null)"
done
