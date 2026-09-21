# Ready-to-apply: TM=8 / TNREG=4 outer product

**Why.** Rounds 102-111 tested six resource mechanisms on `nvfp4_gemm_kernel` and
all six failed to move it: x re-reads, local memory, the weight stream, occupancy
(17% *worse*), shared bandwidth (bf16 `xt`: gate-failing), and inner-loop ALU
issue pressure (neutral). Six failures to find a resource bottleneck is the
information: the 36% FMA efficiency at T=58 comes from the **tile shape**. So
change the shape, do not relieve a resource.

**What it does.** At TM=4/TNREG=8 the outer product issues per k: `wv` 8 B
(LDS.64) + `xv` 16 B + `xw` 16 B = **40 B of shared per 32 FMAs**. At TM=8/TNREG=4
it issues `wv` 16 B (one LDS.128) + `xv` 16 B = **32 B per 32 FMAs**, and drops the
third load. Same accumulator count (`acc[8][4]` = 32 registers, as `acc[4][8]` is
now), same tile (TT=64, TN=64), same block (128 threads).

**The thread mapping must change with it, and that is the whole trap.** Round 96
failed because `TNREG`=4 was combined with a mapping that assumed 8:
`tx = threadIdx.x & 7` covers only 8x4 = 32 of the 64 columns. Going to TM=8 fixes
it by freeing the half of the index that was covering columns:

| | TM=4 / TNREG=8 (now) | TM=8 / TNREG=4 (target) |
|---|---|---|
| `ty` | `threadIdx.x >> 3` -> 16 groups x 4 rows = 64 | `threadIdx.x >> 4` -> 8 groups x 8 rows = 64 |
| `tx` | `threadIdx.x & 7` -> 8 groups x 8 cols = 64 | `threadIdx.x & 15` -> 16 groups x 4 cols = 64 |

**Edits, in order:**

1. `#define GB10_TM 4` -> `8`; `#define GB10_TNREG 8` -> `4`.
2. Both outer products: `ty`/`tx` derivation as in the table above.
3. `gemm2d_outer` (`const float (*wt)`): `wv` becomes two `float4` loads
   (`&wt[k][ty * GB10_TM]` and `+4`); `xv` becomes one `float4` at
   `&xt[k][tx * GB10_TNREG]`; **delete `xw`**; the unrolled body becomes
   `acc[r][0..3] = fmaf(wv_lo/hi, xv, ...)` for `r` in 0..8.
4. `gemm2d_outer_bf16` (`const uint16_t (*wt)`): same, except `wv` is four
   `__bfloat1622float2` conversions producing `w0..w7`.
5. `static_assert(GB10_TM == 4 && GB10_TNREG == 8, ...)` -> `== 8 && == 4`, and
   update its message.

**Expected signals** (record them whether or not the change survives):

| signal | expectation |
|---|---|
| `ptxas -v` smem | **unchanged at 24,576 B** (`wt` and `xt` shapes do not move) |
| `ptxas -v` registers | at or below the current 96; 0 spills |
| `generate --n 16` | **must be 16/16** -- if not, the mapping is wrong again, revert |
| TTFT | must fall well below 462 ms, or the shape was not the cause either |
| `nvdisasm` LDS.128 per k | **1+1 instead of 2+1** |

**If correctness fails:** do not debug it in place. Restore `kernels/gemm.cu` and
recompute the mapping from the table -- the failure will be that `ty`/`tx` no
longer tile 64x64 with 128 threads, which is exactly what round 96 hit.

**Current baseline for comparison** (do not lose this): `kernels/gemm.cu` at
commit `b79fdad`, TTFT 462.4 ms, `generate` 16/16, 96 registers, 24,576 B smem.
