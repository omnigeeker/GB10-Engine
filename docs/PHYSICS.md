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

## M7 lead: in-model GEMV bandwidth is 162 GB/s, not 245 GB/s

The single most important measurement so far, because it says the remaining
throughput is *not* in the GEMV inner loop.

`gb10-bench stream` runs the 401 text-decoder GEMVs back to back with nothing
in between and reports the achieved bandwidth directly:

```
per-token weight traffic : 17.555 GB
time per token           : 71.50 ms
achieved bandwidth       : 245.5 GB/s
projected decode         : 13.99 tok/s (single stream)
```

Stable across 30 sustained iterations (249.8, 242.4, 247.1, 244.7, 247.6,
247.7 GB/s at iters 5/10/15/20/25/29), so this is not thermal or power
throttling — the short-burst roofline is real and sustainable.

Inside the model, `nsys` gives the same kernels 108.8 ms per step, i.e.
**162 GB/s**:

| kernel | gridX | calls/step | ms/step | us/call | GB/s |
|---|---|---|---|---|---|
| nvfp4 (gate/up) | 544 | 128 | 39.46 | 308 | 163 |
| nvfp4 (down) | 160 | 64 | 20.69 | 323 | 155 |
| fp8 (in_proj_qkv) | 320 | 48 | 14.16 | 295 | 178 |
| fp8 (out/o_proj) | 160 | 64 | 12.27 | 192 | 164 |
| fp8 (in_proj_z) | 192 | 48 | 8.23 | 172 | 184 |
| fp8 (q_proj) | 384 | 16 | 5.21 | 326 | 193 |
| nvfp4 (lm_head) | 7760 | 1 | 4.11 | 4112 | 174 |
| **bf16 (in_proj_a/b)** | **2** | **96** | **2.80** | **29** | **17** |
| fp8 (k/v_proj) | 32 | 32 | 1.86 | 58 | 90 |

Total 108.79 ms/step. GEMV is 95.7% of GPU kernel time; every elementwise,
norm, RoPE, recurrence and attention kernel together is 4.9 ms/step.

`nsys` also shows the GPU is ~100% busy (113.0 ms wall vs 113.7 ms of kernel
time), so this is not launch-gap starvation: the same kernels simply run slower
when they are separated by dependency-chained small kernels than when they run
back to back.

Two concrete, separately-attributable defects fall out of the table:

1. **`bf16_gemv` at gridX=2.** `in_proj_a`/`in_proj_b` are `[48, 5120]`, so
   `gx = ceil(48/32) = 2` blocks — 2 of 48 SMs. 47 MB of traffic at 17 GB/s
   costs 2.8 ms/step, ~2.5% of the whole token, for 0.27% of the bytes. This
   needs a split-K or small-N kernel.
2. **Bandwidth scales with grid size** (gridX=384 -> 193 GB/s, gridX=160 ->
   155-164 GB/s), which is the signature of insufficient concurrent memory
   demand rather than a bad access pattern — the reads are already fully
   coalesced and sector-aligned.

### Ruled out, with evidence

- **`exp2f` in the E2M1 decode.** The original decode used
  `(m ? 1.5f : 1.0f) * exp2f(e - 1)`, and `exp2f` is MUFU.EX2 (SFU, ~1/4 FMA
  throughput) with ~18.4e9 NVFP4 weights per token. Replacing it with an
  integer bit-pattern assembly (`kernels/gemv_common.cuh`) is strictly better
  and is kept, but it only moved the token from 113.0 to 110.2 ms — the decode
  was being hidden by memory latency, not limiting.
- **Activation-traffic amortisation.** `ROWS=8` (halving the per-row x reload)
  made things *worse*: 110.2 -> 117.2 ms. Losing blocks costs more than the
  saved activation traffic. Reverted to `ROWS=4`.
- **Load-issue serialisation.** Hoisting all `ROWS` weight and scale loads
  ahead of the arithmetic so they are genuinely in flight together is worth
  keeping: 110.2 -> 108.2 ms (9.07 -> 9.24 tok/s).
- **Throttling.** See the 30-iteration table above.

### Narrowed: it is the NVFP4 kernel specifically, not the context

Running `nsys` on `gb10-bench stream` (30 iterations, 241.9 GB/s) gives
per-kernel times for exactly the shapes the model uses. Normalising both sides
to per-step milliseconds:

| kernel | gridX | calls/step | stream ms | model ms | model/stream |
|---|---|---|---|---|---|
| fp8  | 320 | 48 | 14.09 | 14.16 | **1.00** |
| fp8  | 192 | 48 | 8.13 | 8.23 | **1.01** |
| fp8  | 384 | 16 | 5.32 | 5.21 | **0.98** |
| fp8  | 32  | 32 | 1.97 | 1.86 | **0.94** |
| fp8  | 160 | 64 | 10.96 | 12.27 | 1.12 |
| nvfp4 | 544 | 128 | 18.87 | 39.46 | **2.09** |
| nvfp4 | 160 | 64 | 9.89 | 20.69 | **2.09** |
| nvfp4 | 7760 | 1 | 2.09 | 4.11 | **1.96** |

The FP8 GEMVs are *identical* across the two contexts; every NVFP4 GEMV is
uniformly ~2x slower in the model. Both take the same `memcpy_stod` upload path
(`crates/gb10-model/src/weights.rs`), both use the same kernel and the same
launch config, so this is not allocation, alignment, scheduling or occupancy --
it is a property of the NVFP4 kernel's own execution in this process.

That points at the two things only the NVFP4 kernel touches: the per-group
E4M3 `weight_scale` load and the E2M1 nibble decode. The decode is now
data-independent integer work, so the leading suspect is the **`wscale` byte
load**: it is a separate 32-byte-per-warp memory stream from the packed
weights, and `scalerow` is passed in by the launcher rather than derived in the
kernel. Next step is `ncu` counters (`dram__bytes_read.sum`,
`lts__t_sector_hit_rate.pct`) on both sides to see whether the model is
actually reading more bytes for the same matrix.

### Also ruled out

- **Interleaving small kernels between GEMVs.** Adding a 32-element `add_kernel`
  after every GEMV in `stream` (`--interleave`) leaves the bandwidth untouched:
  245.5 -> 243.0 GB/s. Kernel separation is not the mechanism.

### Correction: the 245 GB/s "achieved" number is cache-inflated

Splitting the same `stream` run by matrix type exposes an impossibility:

| type | traffic/iter | ms/iter | implied GB/s |
|---|---|---|---|
| NVFP4 (gate/up/down + lm_head) | 10.34 GB | 30.85 | **335** |
| FP8 (all attention + in_proj) | 7.23 GB | 40.47 | 179 |

335 GB/s is above the 228 GB/s DRAM peak measured by `bench/hw/bw4.cu` on an
incompressible 16 GiB buffer, so at least a third of `stream`'s NVFP4 reads are
being served from cache. `stream` re-reads the same 21 GB thirty times in a
row with *nothing else running*; the model does the same re-reading but shares
the machine with 150 MB/step of recurrent state, KV cache and activations.

So the honest statement of the M7 gap is narrower than the first draft of this
section claimed:

- pure DRAM streaming ceiling (`bw4.cu`, incompressible, cache-defeating): **228 GB/s**
- model, FP8 GEMV: **178 GB/s**
- model, NVFP4 GEMV: **161 GB/s**
- `stream`, NVFP4: 335 GB/s — not a valid ceiling, cache-assisted

The model is therefore at 70-78% of the real ceiling, not 66% of a 245 figure,
and the achievable target remains the `docs/TARGETS.md` T1 roofline of
12.99 tok/s. The unexplained part is still real and still specific: **FP8 GEMV
is byte-for-byte the same speed in both contexts (178 vs 179 GB/s) while NVFP4
is exactly 2x slower in the model**, which no scheduling or allocation
difference can explain. `ncu` counters would settle it but are unavailable:
`ERR_NVGPUCTRPERM`, and this account has no sudo to set
`NVreg_RestrictProfilingToAdminUsers=0`.
