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

Still ~5x above the ~77 ms/step weight floor (17.6 GB / 228 GB/s). Next
suspect: the Gated DeltaNet recurrence. `gated_delta_rule_step_multi` declares
`__shared__ float S[128][129]` = 66048 B, so only **one block fits per SM**
(102400 B available); at n_seq=16 the grid is (48, 16) = 768 blocks over 48
SMs, i.e. 16 sequential waves.

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
