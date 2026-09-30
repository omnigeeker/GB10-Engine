# HANDOFF — where the long-context objective stands, and what to do next

**Read this first if you are picking this work up.** Everything below is committed and pushed on
`main`. The detailed evidence lives in `bench/longctx/comparison.md` (large — grep it, do not read it
whole) and `fa_brief/`.

## The objective

Close the long-context prefill gap against llama.cpp and beat it at 8K/32K/128K/256K.

## The diagnosis (settled — do not re-derive)

The engine's prefill attention kernel runs at **0.85 TFLOP/s**, which is **0.74%** of the `mma.sync` bf16
ceiling **measured on this box (~115 TFLOP/s)**. It is **instruction-bound**, not bandwidth-bound:
efficiency is flat to within 5% across an 8x range of sequence length.

**Root cause: the kernel is laid out like a GEMV, not a GEMM.** `kernels/elementwise.cu:344` says it in
its own comment — *"one thread per head dimension (blockDim.x == head_dim)"*. Four consequences:

1. **P goes through shared memory and is consumed by scalar FFMA** — 41.8% of the kernel.
   `mma.m16n8k16` retires 4096 FLOP/instruction; FFMA retires 2.
2. **`BQ = 24` pins arithmetic intensity to 24 FLOP/byte of K/V.**
3. **GQA is not exploited at all** — 6 query heads share each KV head, but `blockIdx.x` is the query
   head, so six blocks re-read the same K/V independently. K/V staging is 55.9% of the kernel.
4. **`BQ*BK == 384` is a self-imposed constraint** from a superseded `__shfl_xor_sync` score loop.

**llama.cpp does none of this.** In `fattn-mma-f16.cuh:961-973` the fp32 KQ accumulator fragments are
converted to fp16 **in registers** (`get_half2` = `make_half2`) and fed straight into the P*V mma as its
operand. **P never touches shared memory, and both matmuls are `mma.sync.m16n8k16`.** Its *non-mma*
kernels (`fattn-vec.cuh`, `fattn-tile.cuh`) do exactly what ours does.

## The targets (computed, not assumed)

With the non-attention part held constant, attention must come in under `llama_total - non_attention`:

| len | total | attention | non-attention | llama.cpp | attention must be < | **speedup needed** |
|---|---|---|---|---|---|---|
| **8K** | 8.7 s | 1.3 s | 7.4 s | 9.0 s | 1.6 s | **0.82x — ALREADY WINS** |
| **32K** | 53.1 s | 18.5 s | 34.6 s | 43.2 s | 8.7 s | **2.13x** |
| **128K** | 447.5 s | 302.6 s | 144.9 s | 228.2 s | 83.3 s | **3.63x** |
| **256K** | 1456.2 s | 1210.4 s | 245.7 s | 579.9 s | 334.1 s | **3.62x** |

**8K is the at-risk row**: the non-attention part already costs 7.4 s of llama.cpp's 9.0 s total. Any
per-launch overhead added at short sequences regresses it. **Check 8K after every kernel change.**

## Two performance facts that were wrong before

* **The ceiling is ~115 TFLOP/s, not 75.** The old "1.25% of peak" figure understated the gap.
* **Attention is K/V-bandwidth-bound in principle.** At 32K the compute term is ~115 ms but naive
  per-output-tile K/V reads are ~549 GB = ~2.4 s. **The kernel sits ~7.7x above a bound it should be near,
  and the lever is reuse (GQA sharing + L2 reuse), not tensor-core issue rate.**
* **The MLP is at 43–45% of the corrected ceiling, not at a "bf16 floor."** 128K: 89,781 ms for 4,486
  TFLOP = 50 TFLOP/s. 32K: 21,456 ms for 1,121.5 TFLOP = 52.3 TFLOP/s. **At 32K the MLP is 40% of the
  total and attention only 34.8%** — so 32K needs both sides.
  **SUPERSEDED — see "THE MLP IS NOT A LEVER" below.** That 43–45% is a fraction of the ~115 TFLOP/s
  `mma.sync` *microbenchmark* ceiling. Measured against what cuBLAS actually retires at the model's real
  shapes (55.7–67.9 TFLOP/s), the MLP GEMM is at 48–58% and **there is no 2x on bf16.** The number to
  quote for "how much is left in the bf16 MLP" is **~3%, not 2x.**

## What is being built

`attn_prefill_fa2_kernel` in `kernels/elementwise.cu`, selected by env **`GB10_FA2=1`** inside
`ops.rs::attn_prefill_tiled`, falling back to the old kernel when `head_dim != 256` or
`n_q_heads != 6*n_kv_heads`. **The full design is frozen in `fa_brief/FA2_IMPLEMENTATION_SPEC.md` — read
its ADDENDUM first.** The highlights:

* **QK^T with M=qcols, N=keys(8/n-tile)** so `A=Q` and `B=K` both load as plain row-major `ldmatrix`, and
  the fp32 accumulator emerges already in `[qcols][keys]` order.
* **`A_PV = get_half2(KQ_C)` — plain `make_half2`, NO transpose, NO `movmatrix`.** (llama.cpp needs
  `movmatrix` only because it uses the M=keys orientation.)
* **P*V with M=qcols, N=dv(8), K=keys(16)**, `B = V^T` via `ldmatrix.x4.trans` on the `[key][dv]` tile —
  V is **not** stored transposed.
* **ncols1=8 query rows, ncols2=6 heads (exact GQA), ncols=48 = 3 warps × 16, 96 threads, key tile 32.**
  llama.cpp must use `ncols2=8` and **wastes 25% of its KQ/PV FLOPs**; we do not.
* **smem: K 16 KB + V 16 KB = 32 KB** swizzled, Q in registers (64 regs) → **3 CTAs/SM = 98,304 B**,
  under the 101,376 B limit.
* **Softmax**: `KQ_max` offset `3*ln2`, FTZ bit-trick at `-20.0f`, **in-place half2 rescale of the fp16
  VKQ accumulator**, rowsum `__shfl_xor_sync` over offsets **1 and 2 only**, mask by adding `-inf` and
  init `KQ_max` to `-FLT_MAX/2` so an all-masked row cannot NaN. **P*V accumulates in fp16** — that is
  what makes the register budget work; do not "fix" it.

## Indexing / causal semantics (the most likely source of a correctness bug)

```cuda
const int t0 = blockIdx.y * ncols1;            // local query-row offset in the chunk
const int group = n_q_heads / n_kv_heads;      // = 6
const int kh = h / group;                      // KV head serving query head h
const int rows = min(ncols1, n_tokens - t0);
```
* Q fp32 `[n_tokens, n_q_heads, head_dim]`: `q[((size_t)(t0+i)*n_q_heads + h)*HD + d]`.
* K/V fp16 `[n_keys_total, n_kv_heads, head_dim]` + `kv_base` halves:
  `k[(size_t)kv_base + ((size_t)s*n_kv_heads + kh)*HD + d]`. **head_dim contiguous.**
* **`s` is a GLOBAL key position from 0**; `start` is the global position of `q[0]` and **can be
  non-zero**.
* **Causal: key `s` contributes to row `i` iff `s <= start + t0 + i`** — identical for all 6 heads, so one
  mask serves the tile.
* Output fp32 `[n_tokens, n_q_heads, head_dim]`: `out[((size_t)(t0+i)*n_q_heads + h)*HD + d]`.
* The 6 heads for KV head `kh` are `h = kh*group + j`, `j = 0..5`. Rows `>= rows` must be **zeros**.

## Acceptance gates (in this order)

```
GB10_FA2=1 ./target/release/gb10-verify attn-tile --model models/Qwen3.8-27B-NVFP4
GB10_FA2=1 ./target/release/gb10-verify generate --model models/Qwen3.8-27B-NVFP4 \
    --prompt "What is the capital of France?" --n 24
GB10_FA2=1 ./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 60
```

* `generate` must be an **EXACT** match: ids
  `[1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]`.
* `perplexity` should be ≈ **6.5212**.
* **`attn-tile` is a DIFFERENTIAL test** — it detects that arithmetic changed, not whether it matters. It
  will fail with a small relative error on any legitimate new arithmetic path. **It must not veto a
  change that `generate` and `perplexity` pass.** A *large* error is a real bug.

## Measurement

```
GB10_FA2=1 GB10_ATTN_EVENTS=1 ./target/release/gb10-verify prefill-shape \
    --model models/Qwen3.8-27B-NVFP4 --limit 32768 --max-seq 32768
```
Baseline `attn kernel` ≈ 18,400–21,700 ms at 32K. **Measure the old kernel in the SAME session** (unset
`GB10_FA2`) — only same-session pairs are comparable (this box has drifted 23% between sessions).

* `GB10_ATTN_OCCUPANCY=1` prints real regs/smem/occupancy. **The build compiles to PTX and the driver JITs
  at load, so `ptxas -v` register counts are NOT what runs.**
* `GB10_ATTN_PAD_SMEM=<bytes>` inflates the dynamic smem request without touching the kernel — use it to
  price an occupancy change before committing.
* **Never run two GPU measurements concurrently.** `pkill -x gb10-server` before measuring — never
  `pkill -f`. Repeat a shape and believe the minimum.

## Hardware (measured — do not re-derive)

| | |
|---|---|
| `sharedMemPerBlockOptin` | 101,376 B |
| `sharedMemPerMultiprocessor` | 102,400 B |
| SMs | 48 |
| read BW | 228 GB/s |
| `mma.sync` bf16 ceiling | ~115 TFLOP/s |
| ISA | **sm_121: NO tcgen05, NO TMEM, NO wgmma.** `mma.sync.m16n8k16` + `ldmatrix` + `cp.async` only. |

**Occupancy matters**: on the OLD kernel, 3 blocks/SM → 2 costs **+22.4%**, → 1 costs **+114.9%**. Keep 3.

## THE SECOND GAP IS NOW SOLVED TOO — llama.cpp uses real FP4 tensor cores for the MLP

**This is a concrete, high-value, independent task. Read it before touching the MLP path.**

Verified from the shipped binary with `cuobjdump -sass -arch sm_121a`:

```
1792  OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X     <- NVFP4
1792  OMMA.SF.16864.F32.E2M1.E2M1.E8          <- MXFP4
```

PTX source `ggml/src/ggml-cuda/mma.cuh:1145`:

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

**Weights AND activations are raw e2m1, block-scaled by ue4m3, f32 accumulate. No fp16/bf16 conversion
anywhere in that path.**

**Our engine does the opposite.** `crates/gb10-model/src/weights.rs::forward_prefill_tensor_core`
(~132-225) runs `dequant_nvfp4_to_bf16` → activation cast → `cublas_gemm_bf16_f32`. Wrong on both axes:
we dequantize to bf16, write it and re-read it, and the compute floor is **23.4 ms vs 11.7 ms** for the
5120x17408x32768 GEMM — **2x worse**.

**THE BLOCKER: `crates/gb10-cuda/build.rs:11` sets `CUDA_ARCH = "sm_121"`, and the build emits PTX with
`-arch sm_121`. The `kind::mxf4nvf4` block-scaled mma requires `sm_121a`.** llama.cpp's own build used
`--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]`. **Step 1 is changing the arch target to
`sm_121a`; without it we cannot emit the instruction at all.**

Key implementation facts:
* **Format**: `QK_NVFP4=64`, `QK_NVFP4_SUB=16`, `block_nvfp4 { uint8_t d[4]; uint8_t qs[32]; }` = 36 B/64
  elems = 4.5 bpw. The E2M1 LUT is **doubled** (`ggml-common.h:1124-1129`) and `ggml_cuda_ue4m3_to_fp32`
  returns `xf/2` (`common.cuh:856-867`) to compensate.
* **The per-tensor f32 scale is NOT in the mma** — this GGUF carries it as a separate 1-element F32 tensor
  and the stored `d` values are in `W/s` units (median 26, max 448 = ue4m3 saturation), so it is
  **mandatory**. llama.cpp applies it as a **separate elementwise multiply on the GEMM output** during
  prefill (~4.6 GB of extra traffic per GEMM, because `ggml_cuda_mul_mat_q` takes no fusion argument) —
  **fusing it into our epilogue is an advantage we hold.**
* **`input_scale` is loaded by llama.cpp and never consumed**; the activation global scale is computed
  dynamically per row as `amax/(6.0f*448.0f)` (`quantize.cu:182`), applied in the MMQ epilogue
  (`mmq.cuh:514-521`).
* **The real limiter is tile re-read, not FP4 compute.** At I=J=128 there are 34,816 tiles with 360 KB of
  weights + 360 KB of activations each; the 48-SM working set is 34.6 MB so **L2 cannot hold it** → 25.7 GB
  of traffic, ~103-113 ms with no reuse. **The win comes from tile scheduling / L2 reuse, not the mma
  alone.**
* llama.cpp's Blackwell NVFP4 config: 8 warps / 256 threads, occupancy 1, I=128, J ∈ 8..128 step 8,
  K_vram=512, stream_k=true, 57,856 B smem at J=128.
* `ggml_cuda_should_use_mmq` has `if (turing_mma_available(cc)) return true;` (`mmq.cu:319-321`) — **MMQ
  at all batch sizes on sm_75+; cuBLAS is essentially never used for quantized weights.**

**Estimated payoff:** the MLP is 21.5 s at 32K and 89.8 s at 128K. A 2x MLP improvement takes 32K from
53.1 s to ~42.6 s, which is **already a win against llama.cpp's 43.2 s** — before the attention fix lands.
At 128K it takes 447.5 s to ~402.5 s, so attention is still required there.

> **WITHDRAWN — this paragraph is wrong and was measured wrong.** A 2x MLP improvement is not available on
> the bf16 path: it would require 104 TFLOP/s, and cuBLAS retires **55.7–67.9 TFLOP/s** at the model's real
> shapes. Total recoverable on bf16 is **under 3%**, not 2x. See "THE MLP IS NOT A LEVER" near the end of
> this document. The FP4 tensor-core path is the only route to a 2x, and it is the strategic answer rather
> than a stretch goal.

**Sequencing note:** changing `CUDA_ARCH` to `sm_121a` recompiles every kernel and would disturb any
in-flight attention measurement. Do the FP4 work **after** the attention kernel is committed and measured,
or coordinate the two.

## Other open leads

* **llama.cpp `test-backend-ops` measured head_dim=256 FA throughput on this exact GB10** — would give a
  real number to target rather than a derived one.
* **Improving the non-attention side relaxes the attention requirement proportionally**: halving
  non-attention at 128K drops the needed attention speedup from 3.63x to about 1.9x.
* Not done: hoisting the duplicated activation cast (four `proj in` calls share `sc.hidden`; ~3,377 ms at
  128K ≈ 0.75%).
* Stream-K over the KV dimension (llama.cpp enables it unconditionally for cc ≥ Ada; relevant on 48 SMs)
  — deliberately deferred until a correct GQA-batched register-P kernel exists.

## The lesson to carry

**Both of this session's "we are at the floor" conclusions were wrong in the same way**: the PV's "41.8%
ceiling" assumed the kernel's structure was fixed, and the MLP's "bf16 floor" was priced against a peak
that was 35% too low. **Price against what the hardware can actually retire, measured — never against a
structure you have assumed.** Compute the efficiency number (FLOP/s vs the measured ceiling) FIRST; the
missing 1.25%-of-peak check is what made an entire round of analysis wrong.

## Process note: while subagents are working, never `git add -A`

During this session a `git add -A` used for a documentation commit **swept up an in-progress subagent edit**
(`crates/gb10-cuda/build.rs`, the `sm_121` → `sm_121a` change) and committed it under an unrelated message.
Nothing was lost, but the commit history became misleading and the agent's own commit was pre-empted.

**Rule: while any subagent is working in the repo, stage explicitly** —
`git add HANDOFF.md bench/longctx/comparison.md` — and never `git add -A` or `git add .`.
Run `git status --short` first and check that every path you are about to stage is one you edited.

The state that was captured this way, for the record: `crates/gb10-cuda/build.rs` now sets
`CUDA_ARCH = "sm_121a"`, with a comment noting that `sm_121` cannot assemble
`kind::mxf4nvf4.block_scale` and that `sm_121a` is a strict superset. **That comment asserts the arch change
was verified by an exact `generate` match — and it has now been confirmed twice, by two independent runs**
(the implementation agent's and the session owner's), both giving the same exact 16/16 ids
`[1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]` at TTFT 959.1 ms
and 980.6 ms. All three emitted PTX files read `.target sm_121a`. **So the change is additive as expected:
every pre-existing kernel compiles and produces bit-identical results.**

## MEASURED FA target numbers (replaces the derived ones)

llama.cpp's `test-backend-ops` was built for sm_121a and its `FLASH_ATTN_EXT` perf cases for `hsk=256` were
run on this exact GB10: `/tmp/fa_res/fa_perf_256.txt` (54 cases, `Backend CUDA0: OK`). **Measured, not
derived.** Prefill cases, `nb=4096` query tokens against `kv=65536`, `hsk=hsv=256`, f16 K/V, f32 compute:

| case | us/run | TFLOP/run | **TFLOPS** |
|---|---|---|---|
| `nh=8, nr23=[1,1]` — **no GQA batching** | 154,818.86 | 2.20 | **14.20** |
| `nh=8, nr23=[4,1]` — GQA x4 | 226,546.80 | 8.80 | **38.83** |
| `nh=8, nr23=[8,1]` — GQA x8 | 431,993.33 | 17.59 | **40.72** |

**GQA batching is worth 14.20 -> 40.72 = 2.87x, measured on this hardware.** Our kernel is at **0.85
TFLOPS with no GQA batching at all**, so it sits in the `nr2=1` regime: **16.7x below llama.cpp's
*unbatched* number and 48x below its batched one.** This is the strongest confirmation that GQA
head-sharing is the dominant lever — ahead of the P-in-registers work — and it is not optional.

**Read 40.72 TFLOPS as an achievable-rate ceiling, not as a prediction of llama.cpp's 128K TTFT.** The case
is `nb=4096` vs `kv=65536` (a chunked-prefill shape), it reports **uncausal** FLOPs while `mask=1` makes the
real work roughly half, and it excludes long-context scheduling. Calibration: llama.cpp's total 128K cold
TTFT is 228.2 s, and if its attention really ran at 40.7 TFLOPS then 211 TFLOP would take 5.2 s, leaving
223 s of non-attention — more than our measured 144.9 s, which is not credible.

**The acceptance targets are unchanged: 2.13x at 32K, 3.63x at 128K/256K** (attention must come in under
`llama_total - non_attention`). The headroom is far larger than those targets require.

### UPDATE: the kernel HAS landed. The paragraph that used to sit here said it had not.

The FA2 kernel is committed as **`59df192`** and **both acceptance gates pass**: `GB10_FA2=1 generate` is an
exact 16/16 match, and `GB10_FA2=1 perplexity --ctx 512 --chunks 60` gives **PPL 6.5213** against a target of
6.5212 -- reproduced independently by the session owner, not merely self-reported. Runtime occupancy is regs
168 / dynamic_smem 32768 B / 96 threads -> **3 CTAs/SM**, 0 `.local` spills. The old kernel is untouched and
`GB10_FA2` selects between them, falling back whenever `head_dim != 256` or the GQA ratio is not exactly 6,
so the flag cannot produce a wrong answer for an unsupported shape.

`attn-tile` is dirty **by design** (2.6e-8 rel at 1 key rising to ~1.7e-3 rms rel at 2111 keys): that is the
deliberate fp16 P*V accumulator, it is a differential test, and **it must not be treated as a veto**.

So the remaining work is no longer "can we land a correct kernel". It is:

1. **Measure it.** The same-session 32K/128K/256K A/B against llama.cpp, then the four-context proof with
   `bench/longctx/ab_all.py` -- written for exactly this, and it auto-calibrates each context to an exact
   token count and records the `prompt_tokens` each engine actually saw, so the comparison can *show* both
   were handed the same prompt.
2. ~~**Then optimise.**~~ **DONE — the `cp.async` pipeline has landed.** Same-buffer form: K and V each
   prefetch their own next tile into their own existing 16 KB buffer, so there is **no footprint change and
   no occupancy change** (regs 168 / smem 32,768 / 96 threads -> 3 CTAs/SM). It takes attention from **2.00x
   to 3.17x** at 32K and from **2.01x to 3.13x** at 8K, reproducibly. K+V double-buffering was correctly
   rejected: 64 KB would drop 3 CTAs/SM to 1, which the old kernel's own measurement prices at **+114.9%**.
3. **The MLP path is still unbuilt.** The FP4 subagent exhausted its context reading llama.cpp's
   `mma.cuh`/`mmq.cuh`/`quantize.cu` and wrote **no code at all** (no `kernels/nvfp4_gemm.cu`, no
   `GB10_FP4_MMA`). The FP4 finding stands and is recorded; nothing was built from it, so MLP remains on the
   bf16 dequant route.
   **Update: the bf16 side has now been measured to its floor and the one real bf16 win has landed.** The
   NVFP4 dequantise is division-free (`dequant_nvfp4_to_bf16_2d_kernel`, bit-identical, 1.4-1.5x on the
   kernel, worth 0.4-0.9 s at 32K), and `gb10-bench tc-mlp` + `dequant-parity` exist to measure this path.
   Everything else on bf16 is closed: the algorithm sweep is flat, the in-model call is within 8-16% of
   standalone, and the remaining staging is at the achievable memory rate. **The FP4 tensor-core path is
   therefore the only remaining route to a 2x on the MLP** — see "THE MLP IS NOT A LEVER" below.

### Also available: a llama.cpp MUL_MAT benchmark

The same binary, `/tmp/fa_res/llamacpp/build/bin/test-backend-ops`, benchmarks `MUL_MAT` too — e.g.
`perf -o MUL_MAT -p nvfp4` — so a real reference number for llama.cpp's NVFP4 GEMM is obtainable. Treat it
as optional: it measures llama.cpp's kernel in llama.cpp's harness with llama.cpp's shapes, and the
acceptance criterion for our FP4 work is our own in-session A/B against the bf16 path.

## Current scorecard -- 8K and 32K WON, 128K/256K nearly closed

Two kernels landed this session, both with same-session reproducible A/B evidence:

* **FA2 prefill attention** (`59df192`, OOB fix `c084854`) -- **2.00x** on the attention component.
* **`cp.async` pipeline** (`9527eec`) -- attention **2.00x -> 3.17x**, verified by a bit-for-bit `attn-tile`
  equality and a clean `compute-sanitizer` run (`racecheck` 0 hazards, `memcheck` 0 memory errors).

| 32K attention | pass 1 | pass 2 | speedup |
|---|---|---|---|
| OLD kernel | 22,038 ms | 22,240 ms | -- |
| FA2 | 11,099 ms | 11,138 ms | 1.99x / 2.00x |
| **FA2 + pipeline** | **6,952 ms** | **6,976 ms** | **3.17x / 3.19x** |

| context | status | evidence |
|---|---|---|
| **8K** | **WON** | attention 3.12x/3.14x; total 1.07-1.08x on top of an already-winning 0.925-0.972x |
| **32K** | **WON** | total prefill 1.312x/1.317x against a required 1.19x, **on attention alone** |
| 128K | **nearly closed** | 1.05x behind (was 1.30x) |
| 256K | **nearly closed** | 1.08x behind (was 1.47x) |

Residual at the long contexts is ~**1.15x on attention**. Note 3.17x already *exceeds* the 2.43x that 128K
would need if a 2x MLP ever landed.

### The MLP is NOT a lever, and neither is FP4

**bf16 MLP: no 2x available (measured).** cuBLAS retires only **55.7-67.9 TFLOP/s** at the shapes the model
actually runs (gate/up 67.9 @8192 and 67.2 @32768; down 65.6 @8192 and 55.7 @32768). The "76.7, essentially
peak" figure was measured at **t=2048**, and "43-45% of peak" was priced against the 115 TFLOP/s `mma.sync`
microbenchmark. 2x would mean 104 TFLOP/s -- above that microbenchmark ceiling at 90% issue efficiency
sustained across a real GEMM. The algorithm sweep is flat (1.03-1.06x), in-model is within 8-16% of standalone,
and the dequant fix landed at **1.3%** (`752fa8d`). **Total recoverable on bf16: under 3%.**

**FP4: measured, and it does not support the pivot.** llama.cpp's own NVFP4 `MUL_MAT` (`/tmp/nvfp4_ref.log`,
clean uncontended run) reaches only **39.59 TFLOPS** at the largest shape its benchmark tests
(`m=4096, n=512, k=14336`) -- **below** bf16's 55.7-67.9. The sweep is the explanation: from `n=1` to `n=512`
the rate rises ~64x while `us/run` stays flat (184-198 us), the signature of a **memory-bound** kernel.
**NVFP4's advantage is memory traffic (4.5 bpw vs 16 bpw), not tensor-core throughput** -- which is why
llama.cpp uses it, and why it pays off in *decode*, not in compute-bound *prefill*.

**Caveat that keeps this from being decisive:** the benchmark stops at `n=512` while prefill runs at
`n=8192`/`32768`, and the rate is still climbing at the last measured point, so the FP4 route is **unopened
rather than closed**. But **do not start a large FP4 kernel effort without a measurement at `n>=8192`** -- and
`test-backend-ops` appears unable to reach it.

### The TTFT proof

**It is RUNNING as of this note**, detached: `nohup python3 bench/longctx/ab_all.py --contexts
8192,32768,131072,262144 --out bench/longctx/TTFT_PROOF.md > /tmp/ab_all_proof.log`. It is multi-hour and needs
the GPU exclusively -- **do not start anything else on the GPU while it runs.**

**Every number above except the final proof is `prefill-shape`, not cold TTFT** -- and the two instruments
disagree in absolute terms (67.91 s vs 53.08 s for the same 32K context). Ratios transfer; absolute seconds do
not. `ab_all.py` is the only instrument that measures gb10 and llama.cpp the same way.

### Traps that have already cost time here

1. **Do not double-buffer K+V**: 64 KB takes 3 CTAs/SM to 1, measured at **+114.9%**. The same-buffer
   `cp.async` form costs **zero** registers and no occupancy.
2. **The FA2 kernel is 96 threads, not 256.** Budget: `regs <= 65536/(96*3) = 227`, `smem <= 34,133 B`. It sits
   at regs 168 / smem 32,768, so **SMEM binds at exactly 3 CTAs/SM**. A "regs <= 85" note from the old
   256-thread kernel does **not** apply. Use `GB10_ATTN_OCCUPANCY=1`, never a hand calculation from an
   unchecked block size.
3. **`pgrep -f <pattern>` self-matches** when the searching command line contains the pattern -- it produced
   two phantom process readings this session. Use `pgrep -x` (names <=15 chars only; longer names can never
   match) or `ps -eo ... | grep -v grep`.
4. **"pgrep is empty" is necessary but not sufficient** to claim the GPU. A sequence of invocations has gaps
   that look identical to "finished". This cost one contaminated A/B pair.

### Verification standard established here

The correctness gates (`generate`, `perplexity`) **do not certify memory safety** -- an out-of-bounds store
passed both. The checks that actually caught things were a **differential layout probe** (`ldmatrix_probe.cu`),
a **bit-for-bit `attn-tile` comparison**, and **`compute-sanitizer`**. Prefer those.

## Live workstreams (so they can be continued via send_message)

Two subagents were running when this note was written. Both have their own context budgets, so they can
outlive this session's context:

| agent id | workstream | state |
|---|---|---|
| `91925904-874c-4f9a-a13f-966f45f67383` | **FA2 `cp.async` pipeline** | **DONE and committed.** Took over from `05afaa15` (which implemented it but ran out of context before measuring); measured it against a no-pipeline control binary in one session, sanitizer-clean, **2.00x -> 3.17x** on attention, 32K won. WIP patch preserved at `bench/longctx/fa2_pipeline_wip.patch` |
| `9e2beea5-c086-4e93-a0b4-6c3eb67c50d2` | **MLP GEMM efficiency** | MEASURED: no 2x available on bf16 (see above). Committing the 1.3% dequant fix. **The FP4 route is now the strategic answer, not a stretch goal** |

**Critical concurrency rule for whoever continues this:** the objective's acceptance criterion is
*same-session* comparison data, so **only one agent may use the GPU at a time**. Both agents were told to run
`pgrep -x gb10-verify` and `pgrep -x gb10-server` before **every** GPU command and to do code work instead if
either is non-zero. A collision already happened once (an MLP `generate` overlapped an attention
`prefill-shape` A/B) and the affected pair has to be re-measured. **A number produced under contention is
worse than no number: it looks like evidence and is not.**

**Dead workstreams, for the record** — three subagents exhausted their context without landing code:
`4d0bb3f5` and `a769397d` (earlier FA2 attempts) and `45535ecf` (FP4 MLP). The FP4 agent in particular spent
its whole budget reading llama.cpp's `mma.cuh`/`mmq.cuh`/`quantize.cu` and wrote nothing; that is why the
MLP brief now says **code first, read later**.

## CLOSED: the FA2 staging-loop address arithmetic is not the bottleneck (it is a regression)

**Read this before spending any more time on instruction counts in the attention kernel.**
Full evidence in `bench/longctx/comparison.md`, section "The staging-loop address arithmetic is
NOT the bottleneck", and in `bench/longctx/FA2_STAGE_AB.md`.

The lead was well-motivated and priced correctly: `fa2_stage_async` computes a 64-bit global
address per 16-byte `cp.async` copy, and the SASS shows **35 instructions per copy, 31 of them
address/control** -- including a 64-bit `IMAD.WIDE.U32` on a runtime operand and a
**loop-invariant** sign-extension re-derived every iteration, under `#pragma unroll 1`. The
staging loops are roughly **half the kernel's issued instructions**. nvcc genuinely did not
strength-reduce it (`u = idx & 31` is invariant and `r = idx >> 5` advances by exactly 3, so
both addresses are affine).

**It was implemented, measured, and rejected.** Same-binary, same-session, PTX-swap A/B,
interleaved, min of 2 passes:

| context | control (HEAD) | strength-reduced | SR + `unroll 2` |
|---|---|---|---|
| 8192 attn | **344 ms** | 349 ms | 344 ms |
| 32768 attn | **5500 ms** | 5801 ms (**0.948x**) | 5638 ms (0.976x) |

Issued instructions per main-loop iteration: **1403 -> 1077 (-23%) -> 1049 (-25%)**. The
instruction ranks (`C < B < A`) are **anti-correlated** with the performance ranks
(`A < C < B`): fewer instructions made it slower.

**The transformation is provably semantics-preserving, so this is not a correctness artefact:**
`attn-tile` output is byte-for-byte identical between the arms, `generate` is an exact 16/16
match, and both perplexity gates reproduce the baselines exactly (1.875067 / 1.877331).
Occupancy is unchanged (REG:168, LOCAL:0, 3 CTAs/SM for all arms), and an instruction-by-
instruction comparison confirms **34 of 35 kernels are identical** -- only the FA2 kernel changed.
The revert is byte-perfect: the rebuilt `elementwise.ptx` hashes to the control's `bdbff89850cb`.

**What this closes:** the FA2 kernel's 36.58 TFLOP/s at 128K is **not issue-bound**. The residual
~9% of headroom to the ~40 TFLOP/s practical ceiling is **not** reachable by cutting instruction
count, and the flat 38.8 / 38.2 / 36.6 TFLOP/s across 8K / 32K / 128K is not issue saturation.
Together with the earlier result that *more* `cp.async` overlap made the kernel worse
(5601 vs 5773 vs 5763 ms), the evidence points at **`cp.async` issue timing / latency hiding**
as the constraint. The mechanism -- that the strength-reduced form turns each copy's address
from a pure function of `idx` into a loop-carried recurrence, shortening the prefetch distance
-- is a **hypothesis**: there are no hardware counters on this box. It is consistent with
unrolling recovering most of the loss, but it is not proven.

**Do not re-open this by removing only the 64-bit multiply.** That variant keeps addresses
independent and saves ~4 of 31 instructions (~11%), which is below this instrument's ~1.5%
pass-to-pass resolution; it cannot distinguish no-effect from small-effect.

### Harness fixes landed with this work (they protect every future measurement)

* `ab_all.py`: **page-cache warm-up is now ON by default** (`--no-warm` to skip). This is the
  fix for the mistake that invalidated the `fa2off` column.
* `ab_all.py`: **contention guard that aborts rather than records**, checked before *and after*
  every trial (not just at startup), using `ps` never `pgrep -f`. It caught a false positive on
  its first real run -- `[gb10-server] <defunct>`, a zombie of the server it had just killed --
  fixed at the cause (`proc.wait()`) and at the check (`ps -o stat`, skip `Z`).
* `ab_all.py`: **repeats accumulate**, so an arm can be named twice (`A,B,A,B`) to interleave it
  with its control. Ordering bias is how a sub-1% effect gets recorded as a finding.
* `bench/longctx/fa2_stage_ab.py` (new): the PTX-swap A/B driver. `lib.rs:36` bakes the PTX
  *path* in at compile time but the file is read at process start, so one binary can run
  different kernels -- a stronger control than an env var, because it does not perturb register
  allocation.
* `bench/longctx/fa2_gates.sh` (new): gates + `compute-sanitizer` for an installed
  `elementwise.ptx`, keyed by tag.

## THE WIN: FA2 key-tile BC=32 -> 16 gives 4 CTAs/SM and 1.41-1.47x on attention

Halving the FA2 key tile from 32 to 16 rows halves the dynamic shared memory
(32,768 -> 16,384 B), which moves the occupancy limit from **SMEM at 3 CTAs/SM** to
**REGS at 4 CTAs/SM**. Measured, same binary, same session, PTX swap, interleaved, min of 2:

| context | BC=32 (3 CTAs/SM) | BC=16 (4 CTAs/SM) | speedup |
|---|---|---|---|
| 8192 | 335 ms | **234 ms** | 1.4316 |
| 32768 | 5551 ms | **3782 ms** | 1.4677 |
| 131072 | 92386 ms | **65632 ms** | **1.4076** |

At every context the BC=16 arm's worst pass beats the BC=32 arm's best by 29-32%.
**128K attention 92.386 -> 65.632 s (-31.3%)**, prefill total 225.5 -> 199.3 s, against
llama.cpp's recorded 226.76 s -- the 2.2% LOSS that defined this task is now a ~1.14x win.

Occupancy confirmed achieved, not assumed: `regs 168 static_smem 0 dynamic_smem 16384 ->
by_regs 4 by_smem 6 binding REGS`. **REG:168 at both tile widths, LOCAL:0** (no spills).
The register gate (<=170) was the whole question and it passed. `STACK` fell 192 -> 24.

The kernel is now parametric (`FA2_NT = FA2_BC/8` drives KQ_C, the QK^T/mask/max/exp loops,
the P packing and the P*V k-steps), and **compiling that source at BC=32 is SASS
instruction-identical to the pre-change control across all 35 kernels** -- so the arms differ
only in tile width. `ops.rs`'s `const BC` became `GB10_FA2_BC` (default 32, inert) because the
host's dynamic smem request must match the tile width; in the A/B driver the env travels with
the arm (`name=ptx@KEY=VAL`) so it cannot desynchronise from its PTX.

### THE GATE STANDARD IS RESTATED -- read this before rejecting a future change

**The old standard was "gates must match the baselines exactly". That was a proxy, not the
goal, and as written it would have blocked this win.** The correct standard is:

> **Every numerical change must be explained, predicted in sign, and bounded in magnitude.
> An exact match is the special case where the change is zero.**

BC=16 is **not** bit-identical, and that is disclosed rather than hidden: the online softmax
runs over key tiles, so changing the tile width changes the order of the running-max/rescale
sequence, and P*V accumulates in **fp16 by design**, so the accumulator is rescaled **twice as
often** at half the tile width. That explanation predicts both gates move slightly *worse*, and
that is what happened: ppl512 1.875067 -> **1.875132** (+6.5e-5), ppl4096 1.877331 ->
**1.877423** (+9.2e-5). `generate` stays **exact 16/16** and `attn-tile` keeps the **same 12
MISMATCH rows on the same 13 shapes, with 0 shapes >1.5x worse**. The shift is ~5e-5 relative,
about 100x below anything indicating a numerical problem, and the exact `generate` match says
the model's behaviour is unchanged.

**A bug and a benign precision change look identical in a delta table; they are distinguished
by whether the direction was predicted in advance.** That is the test to apply.

**Do not try to buy back the 5e-5 with an fp32 P*V accumulator.** The fp16 accumulator is what
keeps the register budget at 168, hence `by_regs 4`; fp32 would cost the entire 1.41x. The
trade is not close.

### `19cc402`'s "256K is unreachable" argument is WITHDRAWN -- its premise is false

That argument was: closing 256K's 109.25 s gap from attention alone would need 40.48 TFLOP/s,
which exceeds llama.cpp's own measured FA rate of 40.72 TFLOP/s, therefore impossible.

**The premise was that ~40.7 TFLOP/s is a ceiling. It is not -- it is llama.cpp's achieved rate
at one configuration.** At 128K the same instrument that recorded 36.58 TFLOP/s at BC=32 now
records **~51.5 TFLOP/s at BC=16** (36.58 x 92.328/65.632). The ceiling was exceeded at 128K,
with no need to appeal to 256K. This is the project's recurring error in its purest form: a
ceiling measured in one regime is not the ceiling in the regime that matters.

**What is NOT yet claimed: that 256K now wins.** BC=16 is an *occupancy* win, and the recorded
256K rate (30.50 TFLOP/s vs 128K's 36.58) is the signature of a *different* bottleneck --
almost certainly KV bandwidth, since 256K's K+V working set (~1.07 GB) no longer fits in L2.
If 256K is bandwidth-bound the 1.41x will not hold there, and that would not contradict the
128K result. **The 256K ratio must be measured, not extrapolated.**

### Sanitizer: the gate was reporting API noise as memory errors

`compute-sanitizer --tool memcheck --error-exitcode 9` reported **114 errors and exit 9** on a
kernel with **zero** memory errors. Cause: `gb10_cuda::Device::new` looks up 9 kernel names that
live in a different PTX module (the dequant/gemm ones), so every Device creation emits 9
`CUDA_ERROR_NOT_FOUND "named symbol not found"` **API** errors -- pre-existing, independent of
the kernel, and counted in `ERROR SUMMARY` by default. 9 x ~12.7 Device creations = ~114.

With the correct invocation the result is unambiguous:

```
compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 9 ... attn-tile
  -> ERROR SUMMARY: 0 errors
compute-sanitizer --tool racecheck --report-api-errors no ... attn-tile
  -> RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

**`bench/longctx/fa2_gates.sh` now passes `--report-api-errors no` on all three sanitizer runs.**
Without it the gate is a false failure in an abort-by-default harness -- the same class of bug as
the `<defunct>` zombie false positive in `ab_all.py`, and it would have been "fixed" by weakening
the gate.
