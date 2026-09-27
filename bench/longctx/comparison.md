# Long-context comparison against llama.cpp

Same box, same NVFP4 weights, same prompt, same harness (`bench/longctx/ttft.py`),
runs sequential so neither contender has the GPU to itself. llama.cpp is
`tools/llama.cpp/build/bin/llama-server` on `models/Qwen3.8-27B-NVFP4.gguf` with
`-fa on -np 1`; gb10-server is this repo's build.

Two definitions, because "TTFT" hides a real question:

* **cold TTFT** — the prefix has never been seen, so the whole prompt is prefilled.
* **warm TTFT** — the identical request again, so a prefix cache can skip the prefill.

The prompt is repeated filler plus "list the integers 1 to 300", which stops the
model answering in two tokens and makes OTPS an average over ~198 intervals.

## 8K — with the new decode kernel

| metric | gb10-server | llama.cpp | ratio |
|---|---|---|---|
| prompt tokens | 8,225 | 8,263 | — |
| **cold TTFT** | **89.0 s** | **10.45 s** | 8.5× slower |
| **warm TTFT** | **89.0 s** | **0.24 s** | 371× slower |
| **OTPS** | **2.41** | **7.43** | 3.1× slower |

## 32K

| metric | gb10-server | llama.cpp | ratio |
|---|---|---|---|
| prompt tokens | 32,747 | 32,785 | — |
| **cold TTFT** | **821.0 s** | **44.3 s** | 18.5× slower |
| **warm TTFT** | **821.3 s** | **0.27 s** | 3042× slower |
| **OTPS** | **1.18** | **7.0** | 5.9× slower |

gb10's 32K row is the serial kernel; llama.cpp's is the mean of three trials
(43.55 / 44.85 / 44.46 cold, 0.29 / 0.28 / 0.25 warm).

## What the new decode kernel bought

`attn_decode_multi_kernel` used to give every thread one `head_dim` lane and walk
the keys serially, calling `block_reduce_sum` once per key — two
`__syncthreads()` inside the loop, so the block advanced exactly one key per
barrier and no two keys were ever in flight. It now gives each *warp* a strided
subset of the keys, reduces a key's score with `__shfl_xor` inside the warp, and
merges the eight warps' partials once.

`gb10-verify decode-bench` times both, in the same binary, so this is an A/B and
not a comparison across builds:

| keys | serial | warp | speedup | rms rel |
|---|---|---|---|---|
| 2,048 | 2.405 ms | 0.399 ms | 6.02× | 1.18e-6 |
| 8,192 | 9.629 ms | 1.579 ms | 6.10× | 2.07e-6 |
| 32,768 | 37.831 ms | 6.452 ms | 5.86× | 4.31e-6 |
| 65,536 | 74.216 ms | 12.514 ms | 5.93× | 6.36e-6 |

RMS-relative because a per-element relative error is meaningless on values that
cross zero. Agreement is ~1e-6; a tiling or indexing bug would show up at O(1).
End to end, at 8K with everything else held constant, **OTPS goes 1.82 → 2.41
(+32%)**. Correctness is unchanged: `generate` is still 16/16 token-exact against
the bf16 oracle and identical across 8 repeats.

The 6× does not become 6× end to end because attention is not where decode
spends its time. Working back from the measured numbers at 8K:

* all 21.9 GB of weights, streamed once per token at the measured 228 GB/s, is
  **96 ms** — a hard floor of ~10.4 tok/s;
* attention, at the warp kernel's 1.58 ms per layer × 16, is **~25 ms**;
* measured is 415 ms/token.

So ~300 ms, most of the decode step, is neither weights-at-peak nor attention.
That is the number to chase next, and it is why OTPS is 3.1× off and not 6×.

## Warm TTFT is a missing feature, not a slow kernel

`run_group` calls `state.reset()` on every request
(`crates/gb10-server/src/main.rs`), which wipes the KV cache, so gb10's warm time
is its cold time by construction — 89.0 s versus 89.0 s at 8K. llama.cpp caches
the slot's prompt and answers from it. Until that is implemented, this column
cannot move no matter what the kernels do.

## Cold TTFT

Prefill cost is `≈5.11 ms·T + 5.93e-7·T²`, so the 8.5× at 8K and 18.5× at 32K
are both dominated by kernels that are behind llama.cpp's, in the projections as
well as in attention: short-context prefill runs at ~7.2 TFLOPS against
llama.cpp's ~40. The tiled attention kernel is the other half, and it still
reads K/V once per query head and keeps the cache in f32.
