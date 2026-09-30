# FA2 prefill attention for GB10-Engine — implementation spec

Read `fa_brief/llamacpp_fa_prefill_brief.md` FIRST. It is the code-level analysis of llama.cpp's
`fattn-mma-f16` and it contains the fragment layouts, the smem swizzle, the softmax tricks and the exact
config to port. This document only adds the parts specific to *this* engine.

## Why this is being done

The existing `attn_prefill_tiled_kernel` runs at **0.81–0.85 TFLOP/s**, i.e. ~1.25% of this part's bf16
peak, and it is **instruction-bound, not memory-bound** (efficiency is flat to within 5% across an 8x
range of sequence length). Its two structural faults:

1. **The P matrix goes through shared memory and is consumed by scalar FFMA.** `mma.sync.m16n8k16`
   retires 4096 FLOP per instruction; FFMA retires 2. The PV is 41.8% of the kernel.
2. **Arithmetic intensity is pinned to `BQ = 24`** (24 FLOP per byte of K/V) and **GQA is not exploited**
   — 6 query heads share each KV head, but `blockIdx.x` is the query head, so six blocks re-read the same
   K/V independently.

llama.cpp does neither of these things. Target: match its design, then beat it.

## Non-negotiable correctness gates

1. `./target/release/gb10-verify generate --model models/Qwen3.8-27B-NVFP4 --prompt "What is the capital of France?" --n 24`
   must print `exact match` / `generate: OK`. The expected ids are
   `[1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]`.
2. `./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 60`
   must stay at **PPL ≈ 6.5212** (baseline). A small increase is acceptable *only* if `generate` stays
   exact; report the number either way.
3. `./target/release/gb10-verify all --model models/Qwen3.8-27B-NVFP4` must still print `all gates: OK`.

**`attn-tile` WILL FAIL on any change of arithmetic — it is a differential test against a scalar
reference, so it detects that arithmetic changed, not whether it matters. Do NOT let it veto this work.
`generate` + `perplexity` are the acceptance gates.**

## Hardware constraints (measured on this box, do not re-derive)

| constraint | value |
|---|---|
| `sharedMemPerBlockOptin` | **101,376 B** (~99 KB) |
| `sharedMemPerMultiprocessor` | **102,400 B** |
| `regsPerBlock` | 65,536 |
| `maxThreadsPerMultiProcessor` | 1536 |
| SMs | 48 |
| measured read BW | 228 GB/s |
| bf16 peak | ~75–80 TFLOP/s (**but see the caveat below**) |

**ISA: sm_121 has NO tcgen05, NO TMEM, NO wgmma.** The fastest available path is
`mma.sync.aligned.m16n8k16`. Do not use tcgen05 or wgmma. `cp.async` and `ldmatrix` are available.

**Occupancy matters a lot here — measured, not assumed.** For the *existing* kernel, dropping from 3
blocks/SM to 2 costs **+22.4%** and to 1 costs **+114.9%** on the attention kernel. That was measured by
inflating the dynamic smem request with `GB10_ATTN_PAD_SMEM=<bytes>` (a launch parameter, so the kernel
was untouched). The new kernel has a different balance, but **do not casually let smem or registers push
occupancy to 1.** llama.cpp's config needs ~33.8 KB dynamic smem and gets 3 blocks/SM at 128 threads.

## Integration interface

Existing kernel, for reference:

```cuda
extern "C" __global__ void attn_prefill_tiled_kernel(
    const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    float* __restrict__ out, int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    float scale, int start, int kv_base);
```

* **Q** is fp32, laid out `[n_tokens, n_q_heads, head_dim]`.
* **K and V** are fp16, laid out **`[n_keys_total, n_kv_heads, head_dim]`**, offset by `kv_base` halves.
  **`head_dim` is contiguous — this is already the right layout.** Index:
  `k[kv_base + ((size_t)s * n_kv_heads + kh) * head_dim + d]`.
* **out** is fp32, `[n_tokens, n_q_heads, head_dim]`.
* **`start`** is the sequence position of `q[0]`; **`kv_base`** the position of `k[0]`.
* **Causal mask**: query row at global position `start + t0 + r` attends to keys `s <= start + t0 + r`,
  where `t0 = blockIdx.y * ncols1` (in the old kernel). Keys beyond that must not contribute.
* **`scale`** is the softmax scale (already folded — apply as `q * scale` or `S * scale`, whichever the
  reference does; check `ops.rs` to see exactly what is passed).

The launch site is `crates/gb10-cuda/src/ops.rs::attn_prefill_tiled` (~line 1760–1900). It currently
launches grid `(n_q_heads, ceil(n_tokens/BQ), 1)` with block `(head_dim, 1, 1)`.

## What to build

Add a **new** kernel `attn_prefill_fa2_kernel` alongside the existing one. **Do not delete or modify the
existing kernel.** Select it with an environment variable, e.g. `GB10_FA2=1`, defaulting to the existing
kernel, so the two can be A/B'd in the same session and the change is revertible.

### Configuration to port (from the brief, `fattn-mma-f16.cuh:70-73`)

| parameter | value | note |
|---|---|---|
| query rows per block (`ncols1`) | **8** | |
| query heads per block (`ncols2`) | **6** | **exact GQA** — llama.cpp must use 8 (power of two) and wastes 25% of QK^T; we own the kernel, so use 6 |
| `ncols = ncols1 * ncols2` | **48** | 6 m16n8 n-tiles — legal |
| KV rows per tile (`nbatch_fa`) | **32** | |
| threads | **128** (4 warps) | |
| head dim | **256, in ONE tile** | no inner k-loop over the head dim |
| accumulator types | QK^T **fp32**, P*V **fp16** | the fp16 P*V accumulator is what makes the register budget work |

If `ncols2 = 6` turns out to be awkward in the mma tiling, **fall back to `ncols2 = 8` with two
zero-padded heads** (exactly what llama.cpp does) to get a correct kernel first, then optimise. Correctness
first; a correct ncols2=8 port that beats the old kernel is worth more than a broken ncols2=6 one.

### The two things that must not be compromised

1. **P must stay in registers.** After the QK^T mma the fp32 accumulator fragments are converted to fp16
   **in registers** and fed straight into the P*V mma. Use the pattern from the brief
   (`fattn-mma-f16.cuh:961-973`): `get_half2` = `make_half2` pairing (`mma.cuh:711-720`), and
   `get_transposed` = `movmatrix.sync.aligned.m8n8.trans.b16` (`mma.cuh:722-728`). **No shared memory for
   P, and no FFMA for the PV.** This is the single most important requirement.
2. **GQA batching.** One block must handle all the query heads that share a KV head (6 of them), so the
   K/V tile staged in shared memory is reused 6x instead of once. This is what raises arithmetic
   intensity from 24 to ~48–64 FLOP per byte of K/V.

### Softmax (copy llama.cpp's, it is proven)

* `KQ_max`/`KQ_rowsum` in registers, warp reduce with `__shfl_xor_sync` (offsets 2 then 1).
* **Rescale in place on the fp16 P*V accumulator fragments** with a `half2` multiply.
* `FATTN_KQ_MAX_OFFSET = 3*ln2` shifts the running max so `exp()` stays fp16-representable.
* FTZ bit-trick at `SOFTMAX_FTZ_THRESHOLD = -20.0f`.
* Final divide by `rowsum` at the very end.

These tricks exist because the P*V accumulator is fp16. **Do not omit them** — without the max offset and
FTZ the fp16 accumulator will overflow or produce denormals and `generate` will not be exact.

### Memory pipeline

Start simple, then improve — but keep the K/V staging efficient because it was 55.9% of the old kernel:

1. **First version**: vectorized plain loads into padded shared memory (the existing kernel's staging is
   already decent — `uint4` loads with a `+8` row pad for 16-byte alignment and bank behaviour).
2. **Then**: `cp.async.cg` 16-byte with an `L2::64B` hint, and/or the XOR swizzle from
   `fattn-swizzle.cuh:6-47` (no row padding when the row length is a multiple of 32).
3. **V does not need a transposed copy** — `ldmatrix.sync.aligned.m8n8.x4.trans.b16` transposes on load.
4. Optional later: stream-k over the KV dimension (llama.cpp enables it unconditionally for cc >= Ada;
   on 48 SMs it matters for load balance). **Do not attempt this in the first version.**

## Working method (important — this is where previous attempts failed)

* **Build and test after every meaningful step.** Do not write the whole kernel and then debug it.
* Get a **correct** kernel first — even if it is no faster than the old one — verify `generate` 16/16 and
  `perplexity`, commit it, and only then optimise. A correct-but-slow new kernel is real progress; a
  fast-but-wrong one is a regression and wastes the round.
* Use `GB10_ATTN_EVENTS=1` with
  `./target/release/gb10-verify prefill-shape --model models/Qwen3.8-27B-NVFP4 --limit 32768 --max-seq 32768`
  to price the kernel. **Baseline to beat: `attn kernel` ≈ 18,400–21,700 ms at 32K** (21,734 ms is the
  value from the most recent run; 18,495 ms was the best after the staging fixes — measure the old kernel
  in the same session with `GB10_FA2` unset so the comparison is valid).
* Use `GB10_ATTN_OCCUPANCY=1` to see the real register/smem/occupancy numbers. **The build compiles to PTX
  and the driver JITs at load, so `ptxas -v` register counts are NOT what runs — always use the runtime
  value.**
* Measure a shape twice and believe the minimum. Only same-session pairs are comparable (this box has
  shown 23% drift between sessions).
* `GB10_ATTN_PAD_SMEM=<bytes>` inflates the dynamic smem request without touching the kernel — useful for
  pricing an occupancy change before committing to it.
* **Never run two GPU measurements concurrently** and **stop any server** (`pkill -x gb10-server`) before
  measuring.

## Deliverables

1. `attn_prefill_fa2_kernel` in `kernels/elementwise.cu` (new; the old kernel untouched).
2. The `GB10_FA2` dispatch and launch in `crates/gb10-cuda/src/ops.rs`.
3. `generate` 16/16 exact and `perplexity` ≈ 6.5212 with `GB10_FA2=1`.
4. A same-session A/B of the 32K attention-kernel time (old vs new).
5. Committed with a message stating what was measured, and pushed.

---

# ADDENDUM: the concrete fragment-level design (decided, not yet coded)

This design was worked out against the llama.cpp brief and verified against our kernel's indexing. It is
**better than llama.cpp's in one specific way** and should be followed.

## The key trick: use the M=qcols orientation for QK^T so P needs NO movmatrix

llama.cpp uses the **M=keys** orientation for QK^T, which is why it needs
`get_transposed` = `movmatrix.sync.aligned.m8n8.trans.b16` to turn the KQ accumulator into the PV
A-operand. **We can avoid that instruction entirely:**

* **QK^T: M = qcols (16), N = keys (8 per mma n-tile).** Then `A = Q` (16x16, row-major) and
  `B = K` (8x16, row-major) -- **both load as plain row-major `ldmatrix` fragments**, and the fp32 KQ
  accumulator comes out **already in `[qcols][keys]` order**.
* That ordering is exactly the PV A-operand layout, so:
  **`A_PV = get_half2(KQ_C)` -- plain `make_half2` pairing, NO transpose, NO movmatrix.**
* **P*V: M = qcols (16), N = dv (8 per mma n-tile), K = keys (16 per mma k-step).**
  `B = V^T`, loaded with `ldmatrix.sync.aligned.m8n8.x4.trans.b16` **directly on the `[key][dv]` tile**
  -- so **V is not stored transposed**, exactly as llama.cpp does it.

## Geometry

| item | value |
|---|---|
| `ncols1` (query rows) | 8 |
| `ncols2` (query heads per KV head) | **6 -- exact GQA, no zero padding, no wasted FLOPs** |
| `ncols` (qcols) | **48** = 3 warps x 16 qcols |
| threads | **96 (3 warps)** |
| key tile (`nbatch_fa`) | 32 |
| head dim | 256, one tile |
| warps | **exactly 3 -- no idle warp, no wasted work** |

llama.cpp must use `ncols2 = 8` (powers of two) and throws away 2 of 8 GQA slots = **25% of its KQ and P*V
FLOPs.** We do not.

## Shared memory

`K` 16 KB + `V` 16 KB = **32 KB**, XOR-swizzled with `stride = 128` half2 and **no padding**
(`fattn-swizzle.cuh:6-47`), giving **3 CTAs/SM = 98,304 B**, under the 101,376 B limit.

**`Q` lives in registers** (64 regs) -- so there is **no Q shared-memory buffer and no combine buffer**,
which is what keeps the footprint at 32 KB.

## Softmax (all of llama.cpp's numerical tricks, which the fp16 PV accumulator requires)

* `KQ_max` offset `3*ln2` (`FATTN_KQ_MAX_OFFSET`); FTZ bit-trick at `-20.0f`.
* **In-place `half2` rescale of the fp16 VKQ accumulator fragments.**
* `rowsum` reduction via `__shfl_xor_sync` over offsets **1 and 2 only** -- the qcol row is held by lanes
  with equal `lane/4`, so only bits 0-1 need shuffling (2 shuffles, not 3).
* Mask by **adding `-inf` to masked scores** and initialise `KQ_max` to `-FLT_MAX/2` (llama.cpp's choice)
  **so an all-masked row cannot produce NaN.**
* Final divide by `rowsum` at the very end.

## Dispatch

`GB10_FA2=1` selects the new kernel **inside `attn_prefill_tiled`** (so the existing `attn-tile`
differential test also exercises it when the flag is set -- useful for debugging). **Fall back to the old
kernel whenever `head_dim != 256` or `n_q_heads != 6 * n_kv_heads`.**

Confirmed against our source: `kh = h/group`, the 6 heads are `h = kh*6 + j`, and the causal rule is key
`s <= start + t0 + i` **with `start` possibly non-zero**.
