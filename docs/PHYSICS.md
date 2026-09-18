# GB10 Physics: what the hardware can actually do

This document is the **evidence base** for the performance contract in
`docs/TARGETS.md`. Everything here was measured on the target machine, not
copied from a datasheet.

Machine: NVIDIA DGX Spark, GB10 (Blackwell), `sm_121`, 48 SMs, 121 GiB unified
LPDDR5X, driver 580.142, CUDA 13.0.

---

## 1. Read bandwidth

Datasheet: 8533 MT/s x 256 bit = **273 GB/s**.

Measured with three independent probes. Earlier revisions of this document
reported 200 GB/s; that figure was **wrong** and the correction matters, so the
mistake is recorded here.

| Probe | Method | Result |
|---|---|---|
| `bench/hw/bw.cu` | 8 GiB, 1536 blocks, `__ldcs` | 199 GB/s |
| `bench/hw/bw3.cu` | 8 GiB, 3072 blocks, 4 load policies | 207 – 233 GB/s |
| `bench/hw/bw4.cu` | **16 GiB incompressible random**, 6 configs | **228 GB/s** |
| real workload | NVFP4/FP8/bf16 GEMV over the actual 17.6 GB checkpoint | **235 – 250 GB/s** |

**Two methodology errors produced the original 200 GB/s figure:**

1. *Under-parallelised.* The first probe launched 1536 blocks; the kernel needs
   ~3072 to saturate. Going to 3072 blocks alone moved 8 GiB reads from 199 to
   233 GB/s.
2. *Compressible fill.* The probe filled its buffer with `cudaMemset`, a
   repeating byte pattern that Blackwell's memory compression can compress.
   `bw4.cu` fills with pseudo-random data from a GPU kernel instead.

The real-weight GEMV measuring *above* the synthetic probe (250 vs 228 GB/s) is
consistent: 401 separate matrices have better page/row-buffer locality than one
16 GiB buffer walked with a 12.6 MB stride.

The real-workload figure varies by ~6 % between runs (measured 234.9 and
249.9 GB/s on two consecutive invocations). The best run is **not** quoted as
the headline: both are reported.

> **Working number: 228 GB/s conservative floor, 235 – 250 GB/s demonstrated
> on the real weight set.**

## 2. Per-token weight traffic

Decode is memory-bound: for a dense model every weight is read once per token.
Parsing the safetensors headers of `nv-community/Qwen3.8-27B-NVFP4` (2194
tensors) gives exact bytes:

| Group | GB | Note |
|---|---|---|
| `mlp` NVFP4 (U8-packed) | 9.626 | read every token |
| `linear_attn` Gated DeltaNet (FP8) | 5.588 | read every token |
| `full_attn` (FP8) | 1.678 | read every token |
| `lm_head` (NVFP4) | 0.715 | read every token |
| norms / conv1d / a,b projections | ~0.001 | read every token |
| **dense text total** | **17.608** | **this is the per-token traffic** |
| `embed_tokens` (bf16) | 2.543 | **gather**, ~10 KB per token — *not* 2.5 GB |
| `mtp` (bf16, 1 layer) | 0.849 | read only when speculative drafting |
| `vision_tower` | 0.921 | unused for text |
| checkpoint total | 21.921 | |

**Per-token dense traffic: 17.608 GB.** An earlier revision of this document
used 21.0 GB, which wrongly counted the embedding table as fully read; the
embedding is a row gather, so at batch 1 it costs one 5120-element row.

## 3. Roofline

```
time_per_token = 17.608 GB / bandwidth
```

| Bandwidth | ms/token | single-stream ceiling |
|---|---|---|
| 228 GB/s (conservative probe) | 77.2 | **12.95 tok/s** |
| 235 GB/s (measured GEMV, low) | 74.9 | **13.35 tok/s** |
| 250 GB/s (measured GEMV, high) | 70.4 | **14.20 tok/s** |

Compute is **not** the constraint: 100 tok/s needs
`2 x 27.8e9 x 100 = 5.6 TFLOPS`, while GB10 offers roughly 250 dense FP8
TFLOPS. There is ~45x spare compute and ~7.7x too little bandwidth.

## 4. Consequence for the originally requested targets

| Requested | Required | Measured hardware | Verdict |
|---|---|---|---|
| 100 tok/s, 1 stream | 1.76 TB/s | 0.23 TB/s | **impossible, 7.7x short** |
| 50 tok/s/stream @ 16 | 1.76 TB/s | 0.23 TB/s | **impossible** (per-stream) |
| ~200 tok/s aggregate @ 16 | 0.23 TB/s (weights shared) | 0.23 TB/s | **at the roofline** |

Batching does not raise per-stream speed; it amortises the weight read across
sequences. At batch 16 the step still costs ~70 ms, so all 16 streams advance
one token per step => **~13-14 tok/s each, ~210-225 tok/s aggregate**.

The only lever that reduces bytes-per-token below 17.6 GB is **speculative
decoding with the MTP head** (`mtp_num_hidden_layers: 1` is present in this
checkpoint): one weight read produces several accepted tokens.

| accepted/step | effective tok/s, 1 stream | aggregate @ batch 16 |
|---|---|---|
| 1.0 (no MTP) | ~13-14 | ~210-225 |
| 2.0 | ~26-28 | ~420-450 |
| 2.5 | ~32-35 | ~525-560 |

Acceptance is bounded by the single MTP layer; 2-2.5 is the realistic design
point.

## 5. What this means for "beat Ollama"

Ollama on GB10 runs the same checkpoint against the same wall, so it is
winnable on:

- **TTFT** — prefill is compute-bound, not bandwidth-bound. FP8/NVFP4 GEMM
  efficiency, chunked prefill and prefix caching all matter here.
- **Aggregate throughput at batch 16** — continuous batching + paged KV cache
  versus Ollama's scheduler.
- **Single-stream tok/s** — only marginally; both are pinned near the roofline.

It is **not** winnable on single-stream tok/s by a large margin.

## 6. Reproduce

```bash
nvcc -arch=sm_121 -O3 -o bench/hw/bw  bench/hw/bw.cu  && ./bench/hw/bw
nvcc -arch=sm_121 -O3 -o bench/hw/bw3 bench/hw/bw3.cu && ./bench/hw/bw3
nvcc -arch=sm_121 -O3 -o bench/hw/bw4 bench/hw/bw4.cu && ./bench/hw/bw4
python3 scripts/weight_traffic.py

# the real thing: streams the actual checkpoint through the GEMV kernels
./target/release/gb10-bench stream --out bench/results/stream-m1.json
```
