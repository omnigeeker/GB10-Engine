# Roadmap

Ordered by dependency. Each milestone ends with a gate that must pass before
the next begins; `loop/run_round.sh` enforces build + test + correctness on
every round.

## M0 — Foundation (done)

- [x] Cargo workspace, five crates
- [x] `ModelConfig` parsing the real `qwen3_5` config, incl. MTP fields
- [x] modelopt quantization resolution (NVFP4 group-16 / FP8 / none, honouring
      the `ignore: ["mtp*"]` list)
- [x] mmap safetensors reader, sharded, header-only open of 21.9 GB
- [x] tokenizer via the checkpoint's own `tokenizer.json`
- [x] chat template rendered from `chat_template.jinja` with `minijinja`
- [x] **gate: rendered prompts byte-identical and token ids exact vs the HF
      oracle** on 4 prompts (`cargo test -p gb10-core`)

## M1 — CUDA backend and the roofline

The whole engine lives or dies on streaming 17.6 GB/token at ~230 GB/s.

- [x] `gb10-cuda`: device init, PTX module loading via `cudarc`, typed launch
      wrappers, bandwidth-derived grid/block config
- [x] NVFP4 dequant-and-GEMV kernel: packed E2M1 + group-16 E4M3 scales, fp32
      accumulate
- [x] FP8 (E4M3) GEMV kernel with per-tensor scale
- [x] bf16 GEMV
- [x] **gate PASSED: 249.9 GB/s over the real 17.56 GB of quantized weights,
      projecting 14.23 tok/s — above the 12.95 tok/s conservative roofline**
      (`bench/results/stream-m1.json`)
- [x] correctness: all three kernels verified against an independent CPU
      reference on real weights, error ~1e-6 (fp32 rounding)
- [ ] RMSNorm, SiLU/SwiGLU, residual, fused where it saves traffic (moved to M2)

Design note: GB10 caps dynamic shared memory at 99 KB/block and 100 KB/SM, so
staging activations in shared memory would cap occupancy at one block per SM.
The kernels instead keep the activation tile in registers and hoist it out of
the row loop, amortising activation traffic over `ROWS` output rows with no
shared memory at all.

## M2 — Qwen3.5 layer kernels (done)

- [x] Gated DeltaNet: `in_proj_qkv`, `in_proj_z`, conv1d (kernel 4), gated
      delta rule recurrence in fp32 state, `out_proj`
- [x] Full attention: q (with output gate) / k / v projections, partial RoPE
      (0.25), GQA 24:4, causal softmax, sigmoid output gate
- [x] MLP: gate/up/down NVFP4 with SiLU, `post_attention_layernorm` applied
- [x] **gate: 24 stage checks across layer 0 (DeltaNet) and layer 3
      (attention), all `err/scale < 2e-3`, typically ~1e-7**

The gate compares every intermediate the reference can expose, not just the
layer output — see "Debugging methodology" in `docs/ARCHITECTURE.md`. Three
semantic bugs were found this way: the head_dim^-0.5 factor applied to the key
as well as the query, a missing `post_attention_layernorm`, and a fixture that
had been generated with non-causal attention.

## M3 — Full forward and greedy decode (done)

- [x] 64-layer forward, bf16 embedding gather, fp32 norms, NVFP4 `lm_head`
- [x] KV cache for the 16 full-attention layers; fp32 recurrent state for the
      48 DeltaNet layers
- [x] greedy decode loop
- [x] freeze oracle traces into `fixtures/oracle/` (`tools/full_oracle.py`)
- [x] **gate: 16/16 greedy tokens match `Qwen3_5ForCausalLM` exactly, and the
      59-token chat prompt tokenises identically**

Measured: 17.60 GB streamed per token (predicted 17.608), TTFT 6.89 s for a
59-token prompt (prefill is currently 59 sequential decode steps), decode
8.52 tok/s against a 12.95 tok/s roofline — i.e. 66% of achievable bandwidth.
Closing that gap is M7.

## M4 — MTP speculative decoding

The only lever that beats the bandwidth roofline.

- [ ] MTP layer forward (1 layer, bf16, unquantized)
- [ ] chained drafting: reuse the MTP hidden state to propose k tokens
- [ ] single-pass verification of the k+1 candidates against the target model
- [ ] acceptance-rate instrumentation
- [ ] **gate (T2): >= 18 tok/s single-stream, i.e. >= 2.0 accepted tokens per
      weight read**

## M5 — Paged KV cache and continuous batching

- [ ] paged KV blocks, block table, prefix sharing
- [ ] continuous batching scheduler, 16 concurrent sequences
- [ ] chunked prefill so decode is not starved by long prompts
- [ ] **gate (T3): >= 140 tok/s aggregate at 16 concurrent**

## M6 — Endpoints

- [ ] OpenAI: `/v1/models`, `/v1/completions`, `/v1/chat/completions`
      (streaming SSE and non-streaming)
- [ ] Anthropic: `/v1/messages` (streaming and non-streaming)
- [ ] sampling: temperature, top-p, top-k, seed, stop sequences
- [ ] **gate (T8, T9): 16 simultaneous requests served correctly over both
      protocol shapes**

## M5b — Batched prefill (NEW, highest priority)

Measured against llama.cpp on identical NVFP4 weights, decode already wins
(9.73 vs 7.63 tok/s) but TTFT loses by ~85x: 6273 ms for a 59-token prompt,
because `Model::prefill` calls `step` once per token. llama.cpp does 798.91
tok/s of prompt processing. See `bench/results/llamacpp-baseline.json`.

This is the largest single win available and it is independent of decode.

**Done:** the six batched kernels and their `Ops` launchers are in place and
compile -- `l2norm_scale_batched`, `delta_gate_batched`,
`conv1d_prefill_silu` (threads the 3-token conv history in and out so a prompt
can be chunked), `rope_neox_batched`, `kv_cache_append_batched`, and
`gated_delta_rule_chunk` (the whole T-loop in one launch with the 3.1 MB
recurrent state held in shared memory, rather than one launch plus a state
round trip per token).

`rmsnorm_zero_centered`, `rmsnorm_gated` and `add` already took a row count, and
`Linear::forward` already takes `batch`, so the projections and MLP need no new
kernels.

**Wired and verified (round 12):** `Model::prefill` now embeds all T tokens at
once, runs all 64 layers through `forward_prefill`, and computes logits only
for the final prompt row (`copy_last_row`), so `lm_head`'s 715 MB is read once
rather than once per prompt token. The 64-layer oracle gate still passes
**exactly, 16/16**, which is what makes this safe to build on. TTFT went
6273 -> 4904 ms.

**But that is only 22%, and the reason matters:** `Linear::forward(dev, x, y,
batch=B)` launches `grid_dim = (ceil(N/rows_per_block), B)`, i.e. it runs B
*independent* matrix-vector products. Each one re-reads the entire weight
matrix. So batching the existing GEMV parallelises across tokens but reduces
weight traffic by **zero** -- the 22% is launch-overhead and occupancy, not
bandwidth.

Real prefill needs a **GEMM** that reads each weight tile once and multiplies
it against all T activations, which is the opposite tiling from the decode
GEMV: decode blocks over N and streams W; prefill must block over (N, T) and
reuse the W tile T times.

Roofline for that: at T=59 the arithmetic intensity is
`2*17408*5120 / 44.6e6 = 118` FLOP/byte against a crossover of ~87, so prefill
becomes *compute*-bound rather than memory-bound. llama.cpp's 798.91 tok/s is
1.25 ms/token, i.e. ~75 ms for 59 tokens, which is consistent with being
compute-bound at a few TFLOPS effective. That is the target.

**GEMM landed (round 14):** `kernels/gemm.cu` has NVFP4/FP8/bf16 (N,T)-tiled
GEMMs. Each warp owns one output row, each block covers 8 prompt tokens, the x
chunk is staged in shared memory, and the T-wide partials are reduced once at
the end. All three entry points are `extern "C"` wrappers over a templated
`__device__` body, because NVRTC looks symbols up by name and a template
instantiation mangles.

TTFT **4904 -> 2054 ms**, still 16/16 oracle-exact. Total so far: 6273 -> 2054.

A regression to watch: the first wiring pass used a bulk `sed` that also
rewrote the *decode* call sites to `forward_prefill(..., 1)`, dropping decode
from 9.73 to 6.13 tok/s. Reverted; decode is back to 8.59 in this gate.

**Profiled, and two hypotheses died (round 15).** `nsys` on the prefill path
gives the breakdown unambiguously:

| kernel | launches | ms |
|---|---|---|
| `nvfp4_gemm` | 192 | 1536.9 |
| `fp8_gemm` | 208 | 1051.8 |
| `bf16_gemm` | 96 | 16.5 |
| `gated_delta_rule_chunk` | 48 | **17.6** |
| `rmsnorm_zero_centered` | 2737 | 30.2 |

So prefill is 98% GEMM, and the chunked recurrence costs 17.6 ms -- the part I
expected to be hard is essentially free. But the GEMM moves 17.6 GB in 2605 ms
= **6.8 GB/s**, 26x below the GEMV's 178 GB/s.

*Hypothesis 1: shared-memory-bound.* Plausible (each FMA pair costs 2 shared
loads), but register-blocking to NR=4 made it **worse** (2054 -> 2650 ms):
80 registers drops occupancy to 50%, and the kernel is latency-bound, not
shared-throughput-bound. Reverted to NR=1.

*Hypothesis 2: weights re-read once per T-tile.* With `grid.y = ceil(59/8) = 8`
every grid.y block sweeps the full N and K range, so W is read 8x. Setting
TILE_T=64 should read it once -- but it needs `xv0[64], xv1[64]` = 128
registers on top of a 64-wide accumulator, so it **spilled** and got slower
again (2838 ms). The experiment was invalid, so this hypothesis is still open.

**Solved (round 16) by changing the tiling.** `kernels/gemm.cu` now stages
*both* operands in shared and gives each thread a 2D register tile:

* block covers TILE_N=64 rows x TILE_T=64 tokens
* `Wtile`/`xtile` are stored **transposed** as `[k][row]` -- stored as
  `[row][k]` every inner-loop read is a 32-way bank conflict, because the
  stride between threads is exactly the tile width
* each thread owns a 4x4 sub-tile: 4 W + 4 x shared reads per 16 FMAs (1:2
  instead of 1:1), with a 16-register accumulator
* TILE_T covers the whole prompt, so each weight is read exactly once

64 registers, no spills, 16.6 KB shared, ~67% occupancy.

**TTFT 2054 -> 778 ms.** The profile confirms the shift:

| kernel | before | after |
|---|---|---|
| `nvfp4_gemm` | 1536.9 | 473.9 |
| `fp8_gemm` | 1051.8 | 255.3 |
| `bf16_gemm` | 16.5 | 73.5 -> 16.5 |

`bf16_gemm` regressed because `in_proj_a/b` are [48, 5120] and TILE_N=64, so
the entire grid collapsed to **one block on one SM**. Fixed with a
`self.n < 256` threshold in `Linear::forward_prefill` that falls back to the
batched GEMV: for a matrix that small, re-reading it per token beats starving
47 of 48 SMs.

KC=64 was also tried (fewer barriers) and was worse -- 78-90 registers, less
occupancy, 1025 ms against 847. Reverted to KC=32.

**Further 2054 -> 588 ms (round 17)**, all three steps measured and all
keeping the 64-layer gate exact:

1. **float4 shared reads.** The inner loop did four LDS.32 per operand per k;
   padding the tile strides to a multiple of 4 floats makes each row 16-byte
   aligned and lets one LDS.128 fetch a whole 4-wide sub-tile. 778 -> 741 ms.
2. **De-duplicated the group scales.** The element-at-a-time NVFP4 staging
   issued one scale load *per element*, so each group byte was read 16 times:
   713 MB of staging traffic for a 44.6 MB matrix. One thread per (row, 8-wide
   k segment) now reads the packed weights as a `uint32` and the scale once --
   2 loads per 8 elements instead of 16. 741 -> 627 ms.
3. **Same treatment for FP8**, reading a `uint2` per segment and hoisting the
   per-tensor scale out of the loop entirely. 627 -> 588 ms.

**Where the remaining 8x is.** The GEMM still runs well below both rooflines.
Remaining measured candidates: only ~1.4 waves of 272 blocks
(`grid.x = 17408/64`), so the tail wastes most of the last wave; and 320
`__syncthreads` per block serialising staging against compute, whose standard
fix is double-buffering the shared tiles.

**The design that should work** is a 2D register tile with *both* operands
staged in shared: block covers 64 rows x 64 tokens, `Wtile[64][KC]` and
`xtile[64][KC]` in shared, each thread owning a 4x4 sub-tile. That reads 4 W
and 4 x values from shared per 16 FMAs (ratio 1:2 instead of 1:1), keeps the
accumulator at 16 registers, and reads each weight exactly once because the
block's T extent covers the whole prompt.

**Why TTFT is still 2054 ms and not ~100 ms.** 17.6 GB read once at 178 GB/s
is 99 ms, so we are 20x off the bandwidth bound -- i.e. compute-bound, as the
roofline predicted. But 1.5e12 MACs / 2054 ms = **0.73 TFLOPS**, against ~20
TFLOPS of fp32 peak. The kernel is shared-memory-bound, not FMA-bound: every
one of the 8 warps in a block re-reads the same `xs[i][k]` from shared for its
own row `n`, so each FMA pair costs 2 shared loads. At 128 B/cycle of shared
throughput that is ~4x the FMA time.

The fix is register blocking: have each lane own several `n` rows, load
`xs[i][k]` once into a register, and reuse it across them. With 4 rows/lane the
shared traffic per FMA drops 4x. That costs `TILE_T * rows_per_lane = 32`
accumulator registers, which fits.

**Remaining wiring:**
- [ ] size `Scratch` and `ModelState::{a,b,normed}` by `max_seq` instead of 1
      token (~300 MB at 512, which is fine)
- [ ] `deinterleave_heads` needs a batched variant (it has no row stride today)
- [ ] `DeltaNetLayer::forward_prefill` / `FullAttnLayer::forward_prefill`
- [ ] `Model::prefill`: embed all T tokens, run all 64 layers batched, then
      norm + lm_head + argmax on the last row only
- [ ] replace the O(T^2)-per-thread `attn_prefill_kernel` with a tiled
      flash-attention kernel

## M7 — Roofline optimization (in progress)

**Where we are:** the GEMV kernels achieve **173 GB/s**, which is 76% of the
228 GB/s `bench/hw/bw4.cu` sustains on an incompressible streaming read. Three
independent contexts agree on 173 (`stream`, `store-stream`, and the engine's
own `nsys` kernel time), so the model graph itself costs only ~5% and the whole
remaining gap is inside the GEMV kernel. See `docs/PHYSICS.md`.

**Leading hypothesis:** the fp32 activation tile is 2 KB per warp per k-tile
against only 1 KB of NVFP4 weights (`ROWS=4`), so the kernel issues twice as
many bytes of L2 activation loads as DRAM weight loads, plus the
`weight_scale` byte loads. `ROWS=8` halves that ratio but was measured slower
because it halves the block count.

- [ ] stage the activation k-tile in shared memory once per block instead of
      once per warp (8x less activation traffic), with a transposed layout so
      the per-lane reads are bank-conflict-free
- [ ] make `ROWS` adaptive to `N`, so large-`N` matrices amortise the
      activation tile without starving small-`N` ones of blocks
- [ ] fix `in_proj_a/b` (bf16, `gridX=2`, 17 GB/s) and `k/v_proj`
      (fp8, `gridX=32`, 90 GB/s) with a split-K or small-N kernel
- [ ] widen the `weight_scale` load: it is one byte per lane per row-tile, a
      32-byte transaction against a 256-byte weight transaction

Original plan:



- [ ] profile per-kernel bandwidth utilization, close the gap to 200 GB/s
- [ ] kernel fusion to remove intermediate traffic
- [ ] CUDA graphs to remove launch overhead at batch 1
- [ ] **gate: T1 >= 9.0 tok/s single-stream, T4 >= 280 tok/s at 16 concurrent
      with MTP**

## M8 — Ollama head-to-head

- [ ] user-local Ollama with the same checkpoint
- [ ] identical prompt set, measure TTFT and tok/s for both
- [ ] publish `bench/results/ollama-comparison.md`
- [ ] **gate (T5, T6): TTFT better than Ollama at 1 and at 16 concurrent**
