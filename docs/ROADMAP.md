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

## M7 — Roofline optimization (in progress)

**Lead:** the GEMV kernels achieve 245.5 GB/s back-to-back but only 162 GB/s
inside the model, and the GPU is ~100% busy in both cases — so the loss is in
how kernels are separated, not in any one kernel. Full data and the list of
already-ruled-out hypotheses in `docs/PHYSICS.md`. Closing this gap alone is
worth ~1.5x, i.e. ~13.5 tok/s and target T1.

Immediate defects:
- [ ] `bf16_gemv` for `in_proj_a`/`in_proj_b` runs at gridX=2 (2 SMs of 48)
      for 2.8 ms/step, 2.5% of the token for 0.27% of the bytes
- [ ] bandwidth tracks grid size (193 GB/s at gridX=384 vs 155 at gridX=160)
- [ ] fuse the small per-layer kernels to shorten the dependency chain between
      GEMVs

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
