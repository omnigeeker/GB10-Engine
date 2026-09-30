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
2. **Then optimise.** The kernel has **no pipelining at all**: plain `uint4` staging with two
   `__syncthreads()` per key tile, strictly serial, so the tensor cores idle through every staging phase.
   Note that K+V double-buffering needs 64 KB and would drop 3 CTAs/SM to 1 -- which the old kernel's own
   measurement says costs more than it gains (+114.9% for 3->1) -- so the cheap step is `cp.async` into the
   *same* buffer with the sync narrowed to cover only the copy.
3. **The MLP path is still unbuilt.** The FP4 subagent exhausted its context reading llama.cpp's
   `mma.cuh`/`mmq.cuh`/`quantize.cu` and wrote **no code at all** (no `kernels/nvfp4_gemm.cu`, no
   `GB10_FP4_MMA`). The FP4 finding stands and is recorded; nothing was built from it, so MLP remains on the
   bf16 dequant route.

### Also available: a llama.cpp MUL_MAT benchmark

The same binary, `/tmp/fa_res/llamacpp/build/bin/test-backend-ops`, benchmarks `MUL_MAT` too — e.g.
`perf -o MUL_MAT -p nvfp4` — so a real reference number for llama.cpp's NVFP4 GEMM is obtainable. Treat it
as optional: it measures llama.cpp's kernel in llama.cpp's harness with llama.cpp's shapes, and the
acceptance criterion for our FP4 work is our own in-session A/B against the bf16 path.
