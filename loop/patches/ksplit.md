# Ready-to-apply: 2-way K split for the prefill GEMM

**Why this and nothing else.** `docs/NEXT.md` records every cheaper path as measured
and closed. The prefill GEMM is at **40.6 GB/s = 18% of the 228 GB/s roofline** and
**7.4 TFLOPS** -- bound by neither -- because it runs at **47% occupancy**, and both
limits bind at once: the register budget allows **5.3 blocks/SM** (97 regs x 128 thr)
while the grid only offers **5.7** (272 blocks over 48 SMs). **Raising blocks/SM
capacity cannot help; more blocks are needed.** The n-tile route is closed
(`ops.rs:57`: `GB10_NR` must match `GB10_TN`, and rounds 96-119 settled TN=64). **A K
split is the only remaining source of parallelism.**

**Predicted ceiling: ~2x.** TTFT 434 -> ~220 ms; endpoint 19.83 -> ~35-40 tok/s, which
would clear the >= 30 tok/s endpoint target.

## The current structure (verified by reading the code)

| | |
|---|---|
| `kernels/gemm.cu:41` | `#define GB10_KC 32` |
| `kernels/gemm.cu:364` | `const int nchunk = K / GB10_KC;` |
| `kernels/gemm.cu:311` | `if (t < T) y[(size_t)t * N + n] = acc[i][j];` |
| `ops.rs:1191,1206,1221` | `grid_dim: (cdiv(n,GB10_NR), cdiv(t,GB10_TILE_T), 1)` |

`nchunk = 5120/32 = 160` for this model, so a 2-way split gives 80 chunks per block.

## Edits

1. **`kernels/gemm.cu`** -- add `#define GB10_KSPLIT 2` next to `GB10_KC`, and make the
   chunk loop cover this block's share:
   ```cuda
   const int kc_half = nchunk / GB10_KSPLIT;
   const int kc0 = blockIdx.z * kc_half;
   const int kc1 = (blockIdx.z + 1 == GB10_KSPLIT) ? nchunk : kc0 + kc_half;
   ```
   then iterate `kc0 .. kc1` instead of `0 .. nchunk`.

2. **`kernels/gemm.cu:311`** -- the store must accumulate across the split:
   `atomicAdd(&y[(size_t)t * N + n], acc[i][j]);`

3. **`ops.rs`** -- set `grid_dim.2 = GB10_KSPLIT` and **zero `y` before the launch**,
   because `atomicAdd` requires it. The output is `t * n` floats = 59 x 17408 x 4 B =
   **4.1 MB**, negligible against the 17.6 GB of weights the same launch moves. Use
   `cudaMemsetAsync` on the same stream.

4. **Guard the divisibility.** `nchunk % GB10_KSPLIT` must be 0, and the kernel needs
   the correct `z` extent -- **if it is not divisible, launch with `grid_dim.2 = 1` and
   take the non-split path.** Do not silently drop chunks: `kc1` above already handles
   the tail, but a ragged split changes the occupancy arithmetic and should be measured
   separately rather than assumed.

## The risk to measure first

**The atomics may cost more than the occupancy gains.** Each block stores
`TM x TNREG = 32` accumulators per thread x 128 threads = 4096 atomics, and with 544
blocks that is ~2.2M atomics per matrix against 1.03M output elements -- roughly one
atomic per output element, sustained across 192 matrices per prefill. **If the atomic
throughput is the new bottleneck the split will lose**, and the fallback is a second
reduction kernel (two full-tile writes plus one add) rather than atomics.

## Acceptance

* **`generate --n 16` must be 16/16.** The gate is the only thing that makes a K split
  trustworthy -- a split with a wrong `kc1` still runs and still reports a fast number.
* `chunked-prefill --n 6` must stay OK (it guards the `start > 0` prefill path).
* TTFT from `forward-cost` or `chunked-prefill`, **>= 3 runs, mean against 434 ms**, and
  the ranges should not overlap before this is called a win.
* If TTFT does not move, the occupancy hypothesis is wrong for this kernel and the
  result should be recorded as such -- **it would be the tenth mechanism eliminated on
  the prefill side, and `docs/NEXT.md` wants that recorded, not quietly dropped.**
