# GB10-Engine — performance contract

This is the **definition of done**. It supersedes the original request, whose
numeric targets were shown to be physically unreachable in `docs/PHYSICS.md`
(agreed with the owner before implementation began).

## Hardware

DGX Spark GB10, `sm_121`, 48 SMs, 121 GiB unified LPDDR5X, CUDA 13.0.
Read bandwidth measured at **228 GB/s** (incompressible probe) and **250 GB/s**
demonstrated on the real weight set.

## Model

`nv-community/Qwen3.8-27B-NVFP4` — `qwen3_5` hybrid, 64 layers
(48 Gated-DeltaNet linear-attention + 16 full-attention), hidden 5120,
vocab 248320, **1 MTP layer**, modelopt MIXED_PRECISION
(NVFP4 group-16 MLP + lm_head, FP8 attention projections).
**17.608 GB of dense text weights read per token.**

## Targets (all must be measured, not estimated)

| # | Metric | Target | Rationale |
|---|---|---|---|
| T1 | Single-stream decode | **>= 12.5 tok/s** | 95 % of the 12.95 tok/s conservative roofline |
| T2 | Single-stream with MTP | **>= 25.0 tok/s** | >= 2.0 accepted tokens per weight read |
| T3 | 16-concurrent aggregate | **>= 195 tok/s** | 16 x 12.95 tok/s roofline, minus overhead |
| T4 | 16-concurrent with MTP | **>= 390 tok/s** | MTP amortisation at batch 16 |
| T5 | TTFT @ 1 concurrent, 512-token prompt | **better than Ollama** | prefill is compute-bound, not BW-bound |
| T6 | TTFT @ 16 concurrent | **better than Ollama** | chunked prefill + continuous batching |
| T7 | Correctness | **top-1 token match vs the NVFP4 reference** on the fixed prompt set, >= 32 greedy tokens | no quality regression vs reference |
| T8 | Endpoints | OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/models`; Anthropic `/v1/messages`; streaming and non-streaming | drop-in local replacement |
| T9 | Concurrency | 16 simultaneous requests served without error | required by owner |

### Note on T7

The HF oracle in `/home/wayne/KimiCode/Deploy_QWen3.8-27B/oracle` runs the
**BF16** base model, because `transformers` cannot load a pre-quantized
`modelopt` NVFP4 checkpoint. This engine runs the **NVFP4** weights, so
token-exact agreement with a BF16 reference is not expected: 4-bit
quantization perturbs logits by design.

The correctness gate is therefore two-tiered:

1. **Exact**: top-1 token match against an NVFP4 reference produced by
   dequantizing this checkpoint and running it in bf16 (to be generated in M3).
2. **Agreement**: top-1 agreement rate against the BF16 oracle, reported as a
   percentage, with a floor of 85 % on the frozen prompt set.

Both numbers are reported; neither is presented as the other.

### Explicit non-goals

- **100 tok/s single-stream on this model is not a target** — it requires
  1.76 TB/s, 7.7x the measured hardware limit. It is recorded as impossible.
- Metal/macOS backend — the owner selected CUDA-on-GB10 only.
- Vision/multimodal inference. The vision tower is loaded but not exercised;
  text-only is the contract.

## Measured: 16-way batched decode

Correctness: `gb10-verify batch-parity --n-seq 16 --n 16` reports **16/16
sequences token-exact** against decoding them one at a time, over prompts of
deliberately *different* lengths. Equal-length prompts are not a valid gate:
every sequence then sits at the same position, so a per-sequence addressing bug
reads equivalent data and passes.

Throughput, same kernel set, measured this round:

| n_seq | ms/step | aggregate tok/s | per-sequence tok/s |
|---|---|---|---|
| 1 | 115.1 | 8.69 | 8.69 |
| 2 | 124.9 | 16.01 | 8.00 |
| 4 | 163.2 | 24.51 | 6.13 |
| 8 | 207.1 | 38.63 | 4.83 |
| 16 | 380.3 | **42.08** | 2.63 |

T3 asks for >=195 tok/s aggregate; T1 asks for >=12.5 tok/s single-stream
(roofline ceiling 12.95).

Three steps got here, each measured:

1. The GEMV originally launched with `gridDim.y = batch`, so every block
   streamed the whole weight matrix for its own sequence: 16 sequences cost 16x
   the weight traffic (17.6 GB x 16 = 281 GB ~ 1233 ms at 228 GB/s, against
   1453 ms measured). 11.01 tok/s.
2. Multi-sequence kernels loop over the batch inside the block and load each
   weight tile once. 25.52 tok/s.
3. Profiling showed the batch GEMV was still 545 of ~645 ms/step, ~11x off the
   weight roofline, because with one row per warp each warp pulled all B
   x-vectors through L2 for its own row -- x traffic was ~8x the weight traffic.
   Staging x in `__shared__` once per block made it *worse* (837 ms): two
   `__syncthreads()` per k-tile cost more than the L2 traffic they saved.
   Templating on `ROWS` so one x load feeds ROWS rows worked. On NVFP4:

   | ROWS | ms/step | tok/s |
   |---|---|---|
   | 1 | 624.0 | 25.64 |
   | 2 | 474.7 | 33.70 |
   | 4 | **447.9** | **35.72** |
   | 6 | 470.2 | 34.03 |
   | 8 | 476.5 | 33.58 |

   ROWS=4 is the optimum; beyond it `acc[ROWS][BMAX]` spills. Applying the same
   to FP8 gave 380.3 ms and **42.08 tok/s**.

### Where the remaining time actually is (profiled, not guessed)

Re-profiling after the ROWS tuning, per batched step:

| kernel | ms/step | share |
|---|---|---|
| `nvfp4_gemv_batch` | 202.4 | 53% |
| `fp8_gemv_batch` | 104.3 | 27% |
| `gated_delta_rule_step_multi` | 27.8 | 7% |
| `bf16_gemv_batch` | 8.3 | 2% |
| `attn_decode_multi` | 1.7 | <1% |

The Gated DeltaNet recurrence was **not** the bottleneck -- it is 7%. The GEMV
is 80%, and it runs at 17.6 GB / 380 ms = **46 GB/s, only 20% of the 228 GB/s
roofline**, where the single-sequence kernel reaches 76%.

### It is a register/occupancy problem, not a traffic problem

`ptxas -Xptxas -v` on the batch kernels:

| variant | registers | stack frame |
|---|---|---|
| single-sequence `nvfp4_gemv_kernel` | 40 | 0 |
| batch, ROWS=4, `lo`/`hi` materialised | 128 | 256 B |
| batch, ROWS=4, weights kept packed | **210** | 256 B |
| batch, ROWS=1 + shared x staging | 40 | 0 |

At ROWS=4 the kernel needs `acc[4][16]` + `lo[4][8]` + `hi[4][8]` = 128 live
floats, which exactly exhausts the register file: ptxas spills 256 bytes to
local memory and occupancy falls to 2 blocks/SM. Keeping the weights packed and
re-running the dequant per sequence was *worse* (210 registers -- the compiler
hoists more aggressively without the arrays).

The x-vectors are only 328 KB for batch 16 and stay L2-resident, so x traffic
is **not** the limiter: staging x in `__shared__` (with float4 loads, at 40
registers) still measured 582 ms, worse than 448 ms, because the two
`__syncthreads()` per k-tile are paid against only 8 rows of work per block.

So the remaining fix is to give each block many more rows of work per staged
x-tile -- roughly 64-128 rows -- with the per-row partial sums held in
`__shared__` rather than registers, since 64 rows x 16 sequences cannot fit in
registers at all. Shared budget: `xs[16][512]` = 32 KB plus `part[64][16]` =
4 KB = 36 KB, inside the 48 KB static limit.

### Attempted and NOT yet working (round 37)

That design was implemented as `GB10_BATCH_RB = 64`: each block takes 64 rows,
stages x per k-tile with float4 loads, keeps `part[64][16]` in shared, and
reduces with `warp_reduce_sum` before a lane-0 shared add. It compiles to **109
registers, 0 spill, 36864 B shared**, which is the profile the analysis calls
for.

It is **wrong and slower**: `batch-parity` reports 0/16 with the first mismatch
at token 1, and the step is 628 ms against 377 ms. The launcher grid was
initially left at `cdiv(N, warps)` instead of `cdiv(N, RB)` (1904 of 2176
blocks idle); fixing that changed nothing, so the fault is inside the kernel.

Verified while debugging, and therefore *not* the cause: the staging index math
(`reinterpret_cast<float4*>(xs)[idx]` with `idx = b*(kTile/4) + j` maps to float
`b*kTile + 4j`, matching the source at `x + b*K + i*kTile`); `kTile/kVec/kWarp`
= 512/16/32; `warp_reduce_sum` is a down-shuffle so lane 0 holds the sum and the
`if (lane == 0)` guard is right; `rl = rr*nwarps + warp` covers 0..RB-1 exactly
once; the `__syncthreads()` placement (after zeroing, after staging, after the
row loop, after write-out) has no race; and the shared budget is exact.

Reverted to the ROWS=4 kernels, which remain the verified state at **42.44
tok/s, 16/16 exact, 377.0 ms/step**. The next step is to isolate the new kernel
against `nvfp4_gemv` for the same input at a fixed batch, rather than reasoning
about it further.

## Optimisation order (roofline-driven)

1. **Reach the bandwidth roofline.** At batch 1 every kernel must be a
   streaming GEMV that saturates ~230 GB/s. *Done for the three weight formats:
   the M1 benchmark measures 234.9 – 249.9 GB/s over the real 17.56 GB weight
   set, projecting 13.4 – 14.2 tok/s against a 12.95 tok/s conservative
   roofline.*
2. **MTP speculative decoding.** Reduce bytes-per-token by accepting multiple
   tokens per weight read. This is the only lever that beats the roofline.
3. **Prefill/TTFT.** Chunked prefill, FP8/NVFP4 tensor-core GEMM, prefix
   caching. This is where we beat Ollama decisively.
4. **Continuous batching + paged KV.** Aggregate throughput at 16 concurrent.

## Verification gates

Every round runs `loop/run_round.sh`, which fails the round unless:

- `cargo build --release` succeeds,
- `cargo test` passes,
- the correctness gate passes against the frozen reference traces,
- any perf claim is backed by a fresh benchmark artifact under `bench/results/`.

Numbers are only ever reported from artifacts, never from estimates.
