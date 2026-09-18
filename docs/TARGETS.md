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
