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

## M3 — Full forward and greedy decode

- [ ] 64-layer forward, bf16 embeddings, fp32 norms, NVFP4 `lm_head`
- [ ] KV cache for the 16 full-attention layers; fp32 recurrent state for the
      48 DeltaNet layers
- [ ] greedy decode loop
- [ ] freeze oracle traces into `fixtures/oracle/` and build `gb10-verify`
- [ ] **gate (T7): top-1 token id matches the HF oracle for >= 32 greedy
      tokens on the frozen prompt set**

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

## M7 — Roofline optimization

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
