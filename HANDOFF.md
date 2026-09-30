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

## Open leads not yet resolved

* The two research agents' reports: **quantized GEMM / MMQ / NVFP4 handling** (llama.cpp's MLP path), and
  **llama.cpp `test-backend-ops` measured head_dim=256 FA throughput on this exact GB10** — the latter
  would give a real number to target rather than a derived one.
* **The MLP side (43–45% of ceiling)** is untouched. At 32K it is 40% of the total. Improving it relaxes
  the attention requirement proportionally: halving non-attention at 128K would drop the needed attention
  speedup from 3.63x to about 1.9x.
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
