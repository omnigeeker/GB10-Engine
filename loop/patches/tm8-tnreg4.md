# Ready-to-apply: TM=8 / TNREG=4 outer product

## Next shape: 64-thread blocks, 0.750 B/FMA (added round 116)

**Do not apply "TM=16/TNREG=2", which round 115 suggested.** The arithmetic:

| TM | TNREG | threads | per k | B/FMA |
|---|---|---|---|---|
| 4 | 8 | 128 | wv 8 + xv 32 = 40 for 32 FMA | 1.250 |
| 8 | 4 | 128 | wv 16 + xv 16 = 32 for 32 FMA | **1.000 (current)** |
| 16 | 2 | 128 | wv 32 + xv 8 = 40 for 32 FMA | **1.250 (regression)** |
| **8** | **8** | **64** | wv 16 + xv 32 = 48 for 64 FMA | **0.750** |
| **16** | **4** | **64** | wv 32 + xv 16 = 48 for 64 FMA | **0.750** |

Raising `TM` grows the weight load linearly while `TNREG` shrinks the activation
load; at `TNREG`=2 the activation saving is gone and the shape is back where it
started. **Only raising both helps.**

**The closed form:** `B/FMA = 2/TNREG + 4/TM`, subject to `acc = TM*TNREG` (registers)
and `threads = 4096/(TM*TNREG)`. This is a ladder in `acc`, and it does **not** stop at 0.750 -- an earlier version of
this note claimed that and it was wrong:

| acc | threads | best B/FMA |
|---|---|---|
| 32 | 128 | 1.000 (what round 115 replaced) |
| 64 | 64 | **0.750** |
| 128 | 32 | **0.500** |

**The real limit is register pressure, not the structure.** `acc`=64 costs 64
accumulator registers on top of ~97, and `acc`=128 costs 128 -- so the ladder ends
where ptxas starts spilling, which is what the next round should measure rather
than assume. `acc`=64 is the safe next step; `acc`=128 is worth testing only if 64
comes back with 0 spills and room to spare.

**What (8,8) or (16,4) require, which is why neither is a one-liner:**

1. `acc` goes from 32 to **64 registers** -- watch for spills at 97 now.
2. **`GB10_GEMM_BLOCK` 128 -> 64**, *and* the three `LaunchConfig` block dims in
   `crates/gb10-cuda/src/ops.rs` -- `stage_xtile`/`stage_wtile` stage cooperatively
   over `blockDim`, so leaving this at 128 stages only half the tile.
3. The mapping: (8,8) -> `ty = threadIdx.x >> 3`, `tx = threadIdx.x & 7`;
   (16,4) -> `ty = threadIdx.x >> 4`, `tx = threadIdx.x & 15`. **Verify with the
   rule below before writing the unrolled body.**
4. The same `static_assert` update and regenerated bodies as the (8,4) change.

## (8,8) at acc=64 is rejected on correctness (round 119)

The generator was fixed and `TM`=8/`TNREG`=8 applied with `GB10_GEMM_BLOCK`=64 and the
matching `LaunchConfig` block dims. It compiles, and it is **wrong**:

| | (8,4) current | (8,8) |
|---|---|---|
| registers | 97 | **162** |
| smem | 24,576 B | 24,576 B |
| `generate` | 16/16 | **0/1 (0.0%)** |
| TTFT | 434.6 ms | 544.4 ms (meaningless -- gate failed) |

Reverted; baseline re-verified at 97 registers, 16/16, TTFT 434.6 ms.

**Do not quote the 544.4 ms** -- the same gate-failure trap as rounds 88 and 110.

**Two things this establishes:**

1. **The mapping rule was followed and still failed**, so unlike round 96 this is
   *not* a mapping error. `acc[8][8]` = 64 accumulators pushed registers from 97 to
   **162**, and at that pressure the shape does not hold together. The 0.750 B/FMA
   the ladder promised does not materialise because the shape does not work at all.
2. **The ladder in `acc` is not climbable past 64-with-128-threads.** (8,4) is the
   sweet spot of this structure: it raised `acc` from 32 to 64 while *keeping* the
   128-thread block, and it is the only step that has ever passed the gate. The
   `acc`=64 row in the ladder was reached at **64 threads**, and 64 threads with a
   64-wide tile is what breaks.

**So the prefill GEMM's tile shape is settled at TM=8/TNREG=4.** Further prefill
gains have to come from somewhere other than this ratio -- and the earlier
candidates are all already ruled out by measurement, so the honest position is that
the GEMM's 434 ms prefill is where this session's understanding ends.

## The body generator, and the operand asymmetry that bit twice (round 118)

`acc`=64 (`TM`=8/`TNREG`=8, 64 threads) was attempted twice and **both attempts failed
inside the body generator before writing anything**, so the tree is untouched at
round 115's (8,4) and re-confirms it: `generate` 16/16, TTFT 432.0/434.3 ms.

**The bug was the same both times and it is worth writing down, because it is an
asymmetry between the two operands:**

```python
Q = 'xyzw'
# WEIGHT: the float4 component is selected by the ROW, not the column.
# wv0 holds the weights for rows 0..3, wv1 for rows 4..7 -- each thread's weight
# value for row r is a SCALAR, so index by r % 4.
W = lambda r, c: ('wv0.%s' % Q[r]) if r < 4 else ('wv1.%s' % Q[r-4])
# ACTIVATION: this one IS selected by the COLUMN -- xv covers columns 0..3.
X = lambda c:    ('xv.%s'  % Q[c]) if c < 4 else ('xw.%s'  % Q[c-4])
```

Indexing the weight by `c` (which runs to 7 when `TNREG`=8) raises `IndexError` on
`Q[4..7]`. `TM`=8/`TNREG`=4 did not catch it because there `c` only ever reached 3.

**Everything else in this file is verified:** the mapping rule below, the
`GB10_GEMM_BLOCK` 128 -> 64 change *and* the three `LaunchConfig` block dims in
`crates/gb10-cuda/src/ops.rs`, and the `static_assert` update. With the two lambdas
above, generating the (8,8) bodies is mechanical.

**Mapping rule** (this is what round 96 got wrong and round 115 verified):
`ty_groups = 64/TM`, `tx_groups = 64/TNREG`, and `ty_groups * tx_groups` must equal
`GB10_GEMM_BLOCK`, with `ty = threadIdx.x / tx_groups` and `tx = threadIdx.x % tx_groups`.

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
