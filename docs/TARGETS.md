# GB10-Engine — performance contract

This is the **definition of done**. It supersedes the original request, whose
numeric targets were shown to be physically unreachable in `docs/PHYSICS.md`
(agreed with the owner before implementation began).

## Hardware

DGX Spark GB10, `sm_121`, 48 SMs, 121 GiB unified LPDDR5X, CUDA 13.0.
Usable read bandwidth measured at **200 GB/s**.

## Model

`nv-community/Qwen3.8-27B-NVFP4` — `qwen3_5` hybrid, 64 layers
(48 Gated-DeltaNet linear-attention + 16 full-attention), hidden 5120,
vocab 248320, **1 MTP layer**, modelopt MIXED_PRECISION
(NVFP4 group-16 MLP + lm_head, FP8 attention projections).
**21.0 GB of text weights read per token.**

## Targets (all must be measured, not estimated)

| # | Metric | Target | Rationale |
|---|---|---|---|
| T1 | Single-stream decode | **>= 9.0 tok/s** | 95 % of the 9.52 tok/s bandwidth roofline |
| T2 | Single-stream with MTP | **>= 18.0 tok/s** | >= 2.0 accepted tokens per weight read |
| T3 | 16-concurrent aggregate | **>= 140 tok/s** | 16 x 9.5 tok/s roofline, minus overhead |
| T4 | 16-concurrent with MTP | **>= 280 tok/s** | MTP amortisation at batch 16 |
| T5 | TTFT @ 1 concurrent, 512-token prompt | **better than Ollama** | prefill is compute-bound, not BW-bound |
| T6 | TTFT @ 16 concurrent | **better than Ollama** | chunked prefill + continuous batching |
| T7 | Correctness | **top-1 token match vs HF oracle** on the fixed prompt set, >= 32 tokens greedy | no quality regression vs reference |
| T8 | Endpoints | OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/models`; Anthropic `/v1/messages`; both streaming and non-streaming | drop-in local replacement |
| T9 | Concurrency | 16 simultaneous requests served without error | required by owner |

### Explicit non-goals

- **100 tok/s single-stream on this model is not a target** — it requires
  2.10 TB/s, 10.5x the measured hardware limit. It is recorded as impossible.
- Metal/macOS backend — the owner selected CUDA-on-GB10 only.
- Vision/multimodal inference. The vision tower is loaded but not exercised;
  text-only is the contract.

## Optimisation order (roofline-driven)

1. **Get to the bandwidth roofline.** At batch 1 every kernel must be a
   streaming GEMV that saturates 200 GB/s. Target: single-stream decode within
   10 % of 9.5 tok/s.
2. **MTP speculative decoding.** Reduce bytes-per-token by accepting multiple
   tokens per weight read. This is the only lever that beats the roofline.
3. **Prefill/TTFT.** Chunked prefill, FP8/NVFP4 tensor-core GEMM, prefix
   caching. This is where we beat Ollama decisively.
4. **Continuous batching + paged KV.** Aggregate throughput at 16 concurrent.

## Verification gates

Every round must run `loop/run_round.sh`, which fails the round unless:

- `cargo build --release` succeeds,
- `cargo test` passes,
- the correctness gate (T7) passes against the frozen oracle traces,
- any perf claim is backed by a fresh benchmark artifact under `bench/results/`.

Numbers are only ever reported from artifacts, never from estimates.
