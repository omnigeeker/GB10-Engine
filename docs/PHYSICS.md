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

## M7: the GEMV kernels run at 173 GB/s, 76% of the DRAM roofline

> **This section replaces an earlier and wrong version of itself.** The first
> draft claimed the GEMV reached 245 GB/s in isolation but only 162 GB/s inside
> the model, and pointed at kernel scheduling. That was a measurement bug in
> the benchmark, not a property of the engine -- see "The 2x that wasn't" below.
> The corrected picture is simpler and the target is unchanged.

`gb10-bench stream` pushes all 401 text-decoder matrices through the GEMV
kernels back to back and reports the achieved bandwidth:

```
per-token weight traffic : 17.555 GB
time per token           : 101.58 ms
achieved bandwidth       : 172.8 GB/s
projected decode         : 9.84 tok/s (single stream)
bandwidth utilisation    : 75.8% of measured 228 GB/s
```

Stable across 30 sustained iterations, so this is not thermal throttling.

Three independent contexts agree, which is what makes the number trustworthy:

| context | loader | GB/s |
|---|---|---|
| `gb10-bench stream` | benchmark's own uploader | 172.8 |
| `gb10-bench store-stream` | `gb10_model::Store` (the engine's path) | 173.2 |
| the engine itself, `nsys` GPU kernel time | `gb10_model::Store` | ~169 |

So the engine's own decode path is within ~5% of what the kernels do when
nothing else is running. **There is no structural loss in the model graph.**
The remaining gap is entirely inside the GEMV kernel: 173 GB/s achieved against
the 228 GB/s that `bench/hw/bw4.cu` sustains on an incompressible 16 GiB
streaming read.

Per-shape numbers from `nsys` (model run, before the two fixes below):

| shape | kind | gridX | GB/s |
|---|---|---|---|
| 12288x5120 (q_proj) | fp8 | 384 | 193 |
| 6144x5120 (in_proj_z) | fp8 | 192 | 184 |
| 10240x5120 (in_proj_qkv) | fp8 | 320 | 178 |
| 5120x6144 (o_proj) | fp8 | 160 | 164 |
| 17408x5120 (gate/up) | nvfp4 | 544 | 163 |
| 5120x17408 (down) | nvfp4 | 160 | 155 |
| 1024x5120 (k/v_proj) | fp8 | 32 | 90 |
| 48x5120 (in_proj_a/b) | bf16 | 2 | 17 |

### Occupancy was the limit, not activation traffic

The original hypothesis here was that the fp32 activation tile (2 KB per warp
per k-tile, against 1 KB of NVFP4 weights at `ROWS=4`) was the cost, and that
staging it in shared memory would help. Sweeping `ROWS` falsifies that
directly:

| rows/warp | registers | occupancy | GB/s |
|---|---|---|---|
| 8 | - | - | 157.8 |
| 4 | 60 | 50% (4x256 of 1536) | 172.8 |
| 2 | - | - | 174.5 |
| **1** | **39** | **100% (6x256 = 1536)** | **178.6** |

Bandwidth rises monotonically as `ROWS` falls -- i.e. as the kernel issues
*more* activation traffic and *more* blocks. So activation traffic is close to
free (it is L2-resident and reused by every block), and what actually matters is
resident warps. `ROWS=1` drops register pressure enough to fit the device's
hard cap of `maxThreadsPerMultiProcessor = 1536` (6 blocks of 256), taking
occupancy from 50% to 100%.

End to end that is 108.2 -> 102.7 ms/token, i.e. **9.24 -> 9.73 tok/s**, with
the 64-layer oracle agreement still exact at 16/16.

Remaining gap: 178.6 of 228 GB/s is 78%. With occupancy maxed, the next
suspects are the `weight_scale` byte load (a 32-byte transaction per warp per
row-tile against a 256-byte weight transaction) and the row-strided DRAM access
pattern itself, which reads 256 contiguous bytes per warp per row with rows
2560 bytes apart -- nothing like the purely sequential 16 GiB stream that
`bw4.cu` measures 228 GB/s on. A probe that reads the real weights with the
GEMV's access pattern but no arithmetic would separate those two.

### The 2x that wasn't (kept as a warning)

The original claim was that `nvfp4_gemv_kernel` ran exactly 2x slower inside
the model than inside `stream` while `fp8_gemv_kernel` was identical, and that
this pointed at the NVFP4 kernel's execution. Both halves of that observation
were real; the inference was not.

`stream` derived `n` and `k` from `info.shape`, but an NVFP4 `weight` tensor is
stored **packed** as `[N, K/2]`. So `stream` called the NVFP4 GEMV with
`k = 2560` instead of `5120`, reading half of every row, while still crediting
`raw.len()` -- the full byte count -- to its bandwidth total. The reported
figure was inflated by exactly 2x, and only for NVFP4, because FP8 and bf16 are
stored unpacked and their `shape[1]` really is K. That is a perfect match for
the observed "2.09x", and it is why FP8 looked innocent.

`gemv_parity` was never affected: it hardcodes `k = 5120` and validates against
an independent CPU reference, so the correctness gate was sound throughout.
The lesson is that a bandwidth benchmark which derives its own shapes has no
way to notice it is measuring a different problem than the engine solves; the
fixed version now asserts `raw.len() == n * k / 2` so the two can never drift
apart silently again.

### Ruled out, with evidence

- **`exp2f` in the E2M1 decode.** The original decode used
  `(m ? 1.5f : 1.0f) * exp2f(e - 1)`, and `exp2f` is MUFU.EX2 (SFU, ~1/4 FMA
  throughput). Replacing it with integer bit-pattern assembly
  (`kernels/gemv_common.cuh`) is strictly better and is kept, but it only moved
  the token from 113.0 to 110.2 ms -- the decode was hidden by memory latency.
- **Load-issue serialisation.** Hoisting all `ROWS` weight and scale loads
  ahead of the arithmetic so they are genuinely in flight together is kept:
  110.2 -> 108.2 ms (9.07 -> 9.24 tok/s).
- **Activation-traffic amortisation via `ROWS=8`.** Worse: 110.2 -> 117.2 ms.
- **Interleaving small kernels between GEMVs.** Adding a 32-element
  `add_kernel` after every GEMV in `stream` (`--interleave`) leaves bandwidth
  untouched. Kernel separation is not a mechanism.
- **The tiny `in_proj_a/b` matrices.** Excluding all 96 of them
  (`store-stream --skip-small`) moves the total from 168.9 to 173.2 GB/s, i.e.
  they cost only their own 4.2 ms. Not a disproportionate pipeline drain.
- **The loader.** `Store::linear` and the benchmark's uploader both end in
  `Stream::alloc` + `memcpy_htod`; `memcpy_stod` and `clone_htod` are literally
  the same body in cudarc 0.19.9 (`driver/safe/core.rs`).
