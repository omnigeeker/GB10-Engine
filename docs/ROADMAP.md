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
