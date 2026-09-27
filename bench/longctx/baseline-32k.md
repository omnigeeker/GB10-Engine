# Long-context baseline: gb10-server vs llama.cpp at 32K

Measured on the same box, same model (NVFP4), same prompt, same harness
(`bench/longctx/ttft.py`), sequential runs so neither contends for the GPU.
Prompt is repeated filler (~31 tokens/repeat) plus a request to list integers
1..300, which makes the model decode for the full 200 tokens so OTPS is computed
over ~198 intervals rather than 3.

| metric | gb10-server | llama.cpp | ratio |
|---|---|---|---|
| prompt tokens | 32,747 | 32,785 | — |
| **cold TTFT** | **821.0 s** | **44.3 s** | 18.5× slower |
| **warm TTFT** | **821.3 s** | **0.27 s** | 3042× slower |
| **OTPS** | **1.18** | **7.0** | 5.9× slower |

llama.cpp cold/warm were 3 trials (43.55 / 44.85 / 44.46 and 0.29 / 0.28 /
0.25); gb10 one trial, because a single cold+warm pair already costs 27 minutes.

## What each gap actually is

**Warm TTFT is a missing feature, not a slow kernel.** `run_group` calls
`state.reset()` on every request (`crates/gb10-server/src/main.rs`), which wipes
the KV cache, so gb10's warm time is identical to its cold time by construction.
llama.cpp caches the slot's prompt and answers from it. This one is fixable.

**Cold TTFT and OTPS are kernel-quality gaps**, and they are bigger than the
attention alone:

* Prefill at short context is 7.5 ms/token ≈ **7.2 TFLOPS** of the model's
  ~54 GFLOP/token. llama.cpp's 44 s for 32 K is ~745 tok/s ≈ **40 TFLOPS**.
  So even the projection/MLP path is ~5.5× behind, before attention is involved.
* Attention is ~646 s of the 821 s — the `5.93e-7·T²` term. At 32 K the tiled
  kernel reads ~25.7 GB per token... per *step* in decode, and moves data at
  ~78 GB/s against ~228 GB/s measured read bandwidth.
* Two structural wastes in both attention kernels: every one of the 24 query
  heads reads its group's K/V independently (**6× redundant traffic** for a
  GQA group of 6), and the KV cache is f32 where f16 would halve it. Together
  that is 12× more traffic than the floor.
* Decode falls from ~8.5 tok/s at short context to 1.18 at 32 K, i.e. decode
  attention is O(context) with a large constant.

## Honest read

Warm TTFT is achievable — prefix caching is a bounded feature.

Cold TTFT needs ~18× and OTPS ~6×. The attention fixes above (GQA-aware tiling,
f16 KV, pipelining) are worth roughly 12× on attention traffic in the ideal
case, which by itself does **not** close cold TTFT, because the GEMM path is
also ~5.5× behind and contributes ~175 s of the 821 s. Closing all of it means
bringing both the matmul and attention kernels to llama.cpp's level, which is a
large, high-risk effort rather than a tuning pass.

The number worth re-checking before concluding: my GEMM at 7.2 TFLOPS is far
below what NVFP4 tensor cores should do on this part, so the projection path may
have a discrete bug or a missing fast path rather than merely being unpolished.
That is the cheapest thing to investigate next and the one with the most upside.
