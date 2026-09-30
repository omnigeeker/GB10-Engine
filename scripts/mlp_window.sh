#!/bin/bash
# One-shot GPU batch for the MLP workstream.
#
# RUN ONLY WHEN THE ATTENTION AGENT'S RUN IS FINISHED -- not merely when
# `pgrep -x gb10-verify` is empty. Its A/B is a sequence of separate
# invocations with 30-60 s gaps between them, so "no process right now" cannot
# distinguish "finished" from "between runs". Starting in a gap contaminates the
# reproducible half of its result. Use the log test:
#
#   tail -6 /tmp/fa2_ab2.log | grep -q total && echo DONE
#
# Everything here is same-binary, same-session: the dequant A/B is selected by
# GB10_DEQ_2D rather than by swapping binaries, so the pair is comparable.
cd /home/wayne/dsh/QWen3.8-27B-GB10 || exit 1
M=models/Qwen3.8-27B-NVFP4
P="--tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw"

echo "########## 0. sanity: nothing else on the GPU ##########"
pgrep -x gb10-verify && { echo "REFUSING: another gb10-verify is running"; exit 1; }
pgrep -x gb10-server && { echo "REFUSING: a gb10-server is running"; exit 1; }
pkill -x gb10-server

# ---- PRIORITY 1: where is the 22.8 s actually going? -----------------------
# The three GEMMs are 22.8 s of the 25.6 s MLP, so this is the measurement that
# decides whether anything is recoverable at all.
echo "########## 1. tc-mlp: cuBLAS rate at the real shapes + algo sweep ##########"
./target/release/gb10-bench tc-mlp

# ---- PRIORITY 2: the dequant fix, measured once ----------------------------
echo "########## 2. 32K prefill, OLD dequant, no events ##########"
GB10_DEQ_2D=0 ./target/release/gb10-verify prefill-shape --model $M --limit 32768 --max-seq 32768 2>&1 | tail -3
echo "########## 3. 32K prefill, NEW dequant, no events ##########"
GB10_DEQ_2D=1 ./target/release/gb10-verify prefill-shape --model $M --limit 32768 --max-seq 32768 2>&1 | tail -3

# ---- PRIORITY 3: the MLP component itself, before/after --------------------
echo "########## 4. 32K prefill + MLP/GEMM events, OLD dequant ##########"
GB10_DEQ_2D=0 GB10_MLP_EVENTS=1 GB10_GEMM_EVENTS=1 \
  ./target/release/gb10-verify prefill-shape --model $M --limit 32768 --max-seq 32768 2>&1 | tail -6
echo "########## 5. 32K prefill + MLP/GEMM events, NEW dequant ##########"
GB10_DEQ_2D=1 GB10_MLP_EVENTS=1 GB10_GEMM_EVENTS=1 \
  ./target/release/gb10-verify prefill-shape --model $M --limit 32768 --max-seq 32768 2>&1 | tail -6

# ---- PRIORITY 4: correctness, including a LONG-context gate ----------------
# ctx 512 is the number everyone already trusts; keep it. But ctx 512 is exactly
# the regime where a long-context bug does not appear, so add ctx 4096 -- a
# change to a shared dequantise path needs a gate that moves if any logit is
# wrong anywhere in a 4096-token window. Baseline is taken with the change OFF
# so the comparison is same-session rather than against someone else's number.
echo "########## 6. perplexity ctx 4096, OLD dequant (BASELINE) ##########"
GB10_DEQ_2D=0 ./target/release/gb10-verify perplexity --model $M $P --ctx 4096 --chunks 8
echo "########## 7. perplexity ctx 4096, NEW dequant (must match) ##########"
GB10_DEQ_2D=1 ./target/release/gb10-verify perplexity --model $M $P --ctx 4096 --chunks 8

echo "########## 8. generate: must be an EXACT 16/16 id match ##########"
./target/release/gb10-verify generate --model $M --prompt "What is the capital of France?" --n 24
echo "########## 9. perplexity ctx 512: target 6.5212 ##########"
./target/release/gb10-verify perplexity --model $M $P --ctx 512 --chunks 60

echo "########## DONE ##########"
