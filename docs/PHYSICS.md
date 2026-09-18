# GB10 Physics: why the original throughput targets are unreachable

This document is the **evidence base** for the performance contract in
`docs/TARGETS.md`. Everything here was measured on the target machine, not
copied from a datasheet.

Machine: NVIDIA DGX Spark, GB10 (Blackwell), `sm_121`, 48 SMs, 121 GiB unified
LPDDR5X, driver 580.142, CUDA 13.0.

---

## 1. Measured memory bandwidth

`bench/hw/bw.cu` (8 GiB buffer, 5 launch configs, `__ldcs` streaming loads):

| Pattern | Achieved |
|---|---|
| Pure read (8 GiB) | **195 – 199 GB/s** |
| Pure write (8 GiB) | 161 – 170 GB/s |
| Triad read+write | 120 – 124 GB/s |
| `cudaMemcpy` D2D | 189 – 193 GB/s |

Theoretical: 8533 MT/s x 256 bit = 273 GB/s. Measured streaming read is
**~200 GB/s = 73 % of peak**, which is normal for LPDDR5X.

A 24 GiB buffer measured *slower* (44–158 GB/s) due to page/TLB pressure from
the 100 GiB page cache; 8 GiB is the representative figure for steady-state
decode.

> **Working number: 200 GB/s of usable weight-read bandwidth.**

## 2. Measured per-token weight traffic

Decode is memory-bound: for a dense model *every* weight must be read from
HBM/LPDDR once per generated token. Parsing the safetensors headers of
`nv-community/Qwen3.8-27B-NVFP4` (2194 tensors) gives the exact bytes:

| Module | GB | tensors |
|---|---|---|
| `mlp` (NVFP4, U8-packed) | 9.626 | 768 |
| `linear_attn` Gated DeltaNet (FP8) | 5.588 | 720 |
| `embed_tokens` (BF16) | 2.543 | 1 |
| `full_attn` (FP8) | 1.678 | 224 |
| `vision_tower` (not used for text) | 0.921 | 333 |
| `mtp` (BF16, 1 layer) | 0.849 | 15 |
| `lm_head` (NVFP4) | 0.715 | 4 |
| norms/misc | 0.001 | 129 |
| **total checkpoint** | **21.921** | 2194 |

**Text-only traffic per token: 21.0 GB.**

## 3. Roofline

```
time_per_token = weight_bytes / bandwidth = 21.0 GB / 200 GB/s = 105 ms
=> single-stream ceiling = 9.5 tok/s
```

| Variant | GB/token | ms/token | tok/s ceiling |
|---|---|---|---|
| full checkpoint | 21.92 | 109.6 | 9.12 |
| text-only | 21.00 | 105.0 | **9.52** |
| text w/o vision+mtp | 20.15 | 100.8 | 9.93 |

Compute is **not** the constraint: 100 tok/s needs only
`2 x 27.8e9 x 100 = 5.6 TFLOPS`, while GB10 offers roughly 250 dense FP8
TFLOPS. There is ~45x of spare compute and ~10x too little bandwidth.

## 4. Consequence for the requested targets

| Requested | Required | Measured hardware | Verdict |
|---|---|---|---|
| 100 tok/s, 1 stream | 2.10 TB/s | 0.20 TB/s | **impossible, 10.5x short** |
| 50 tok/s/stream @ 16 | 2.10 TB/s | 0.20 TB/s | **impossible** (per-stream) |
| 150 tok/s aggregate @ 16 | 0.20 TB/s (weights shared) | 0.20 TB/s | **at the roofline** |

Batching does not raise per-stream speed; it amortises the weight read across
sequences. At batch 16 the step still costs ~105 ms, so all 16 streams advance
one token per 105 ms => **9.5 tok/s each, ~150 tok/s aggregate**.

The only lever that reduces bytes-per-token below 21 GB is **speculative
decoding with the MTP head** (`mtp_num_hidden_layers: 1` is present in this
checkpoint): one weight read produces several accepted tokens.

With MTP acceptance `a` tokens/step:

| accepted/step | effective tok/s, 1 stream | aggregate @ batch 16 |
|---|---|---|
| 1.0 (no MTP) | 9.5 | ~150 |
| 2.0 | 19.0 | ~300 |
| 2.5 | 23.8 | ~380 |

Acceptance is bounded by the single MTP layer; 2–2.5 is the realistic design
point.

## 5. What this means for "beat Ollama"

Ollama on GB10 runs the same checkpoint and is subject to the same roofline.
It is therefore winnable on:

- **TTFT** — prefill is compute-bound, so FP8/NVFP4 GEMM efficiency, chunked
  prefill and prefix caching matter. This is where a hand-written engine can
  win big.
- **Aggregate throughput at batch 16** — continuous batching + paged KV cache
  vs Ollama's scheduler.
- **Single-stream tok/s** — only marginally; both are pinned to ~9.5 tok/s.

It is **not** winnable on single-stream tok/s by a large margin, because both
implementations are against the same wall.

## 6. Reproduce

```bash
nvcc -arch=sm_121 -O3 -o bench/hw/bw  bench/hw/bw.cu  && ./bench/hw/bw
nvcc -arch=sm_121 -O3 -o bench/hw/bw2 bench/hw/bw2.cu && ./bench/hw/bw2
python3 scripts/weight_traffic.py            # tensor-by-tensor traffic table
```
