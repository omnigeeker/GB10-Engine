# Handoff: the prefill GEMM is the critical path

## Where the engine stands

Working and verified (gates in `loop/run_round.sh`):

| gate | result |
|---|---|
| `generate` | 16/16 token-exact vs the Python oracle |
| `batch-parity` | 16/16 exact at n_seq=16, ~40.75 tok/s aggregate |
| `gemv-parity` | batch=1 and batch=16 within 1.6e-6 of reference |
| `chunked-prefill` | two-chunk prefill == one-shot; snapshot/restore reversible |
| `mtp-generate` | token-exact, batched verify composed, 0.32x |
| server | OpenAI + Anthropic endpoints, greedy, serialised |

Measured: single-stream 8.69 tok/s, n_seq=16 aggregate 40.75 tok/s,
TTFT 452.6 ms (llama.cpp ~74 ms).

## The one change worth making next

`nsys` says the prefill forward is 390 ms, split as
`nvfp4_gemm_kernel` 233 ms at **41 GB/s** and `fp8_gemm_kernel` 133 ms at
**42 GB/s** -- both ~18% of the 228 GB/s roofline, against the decode GEMV's
148 GB/s. `GB10_KC` is 32, so each thread loads only 16 bytes of weight per
chunk: the loop is a latency serialisation, not a stream. That is also why the
forward costs the same 390 ms at t=1 and t=16.

Full reasoning, all the constraint arithmetic, and the two rejected routes
(dynamic shared memory, which cudarc cannot reach; and a full retiling) are in
`docs/TARGETS.md`. Read that first -- it will save a round.

### Concrete plan

Step 1 is self-contained and worth doing on its own, because it is the
prerequisite for step 2 and is independently verifiable. Exact edits, since the
line numbers are known:

* `stage_wtile_fp8` (gemm.cu:84) -- change the parameter from
  `float (*wt)[GB10_WSTRIDE]` to `uint16_t (*wt)[GB10_WSTRIDE]` and store
  `__bfloat16_as_ushort(__float2bfloat16_rn(e4m3_to_float(...)))`, exactly as
  `stage_wtile` (gemm.cu:55) already does for nvfp4. e4m3 has 3 mantissa bits
  and bf16 has 7, so **this conversion is lossless**.
* the fp8 body (gemm.cu:299) and the bf16 body (gemm.cu:332) -- change
  `__shared__ float wt[...]` to `__shared__ uint16_t wt[...]`, and at
  gemm.cu:311/317 switch `stage_wtile_fp8` stays, but the compute call must
  become `gemm2d_outer_bf16` (gemm.cu:181) instead of `gemm2d_outer`.

**DONE (round 56).** fp8 shared is now 26112 B, matching nvfp4, and `generate`
is still **16/16 exact** -- so the e4m3 -> bf16 conversion is confirmed lossless,
as the mantissa widths predicted.

It did **not** change the forward time:

| t | before | after |
|---|---|---|
| 1 | 387.81 | 383.29 |
| 4 | 388.08 | 393.88 |
| 16 | 404.08 | 404.82 |

That is a useful negative result: occupancy was not the limiter (fp8 went from 2
to 3 blocks/SM and gained nothing), which further isolates the cause to the
per-chunk load latency that `GB10_KC = 32` produces. It also leaves the fp8
kernel in the shape step 2 needs.

### Step 2 landed -- and refuted its own hypothesis (round 58)

KC=64 is in, correct, and verified: `generate` **16/16 exact**,
`chunked-prefill` agrees. All three GEMMs now stage 35840 B of tile, twice the
bytes in flight per chunk that KC=32 gave.

**It changed almost nothing.**

| t | KC=32 | KC=64 |
|---|---|---|
| 1 | 386.13 | 377.31 |
| 2 | 384.85 | 378.68 |
| 4 | 389.01 | 379.00 |
| 8 | 396.70 | 389.52 |
| 16 | 404.59 | 401.48 |

~2%, when the hypothesis predicted a move from 41 GB/s toward 100+. So
**"too few bytes per thread per chunk" is not the cause either.** That is the
third explanation eliminated, after occupancy (round 56: fp8 2 -> 3 blocks/SM
changed nothing) and tile size (round 53).

What survives is that the cost is *fixed per launch* and insensitive to tile
geometry, occupancy and bytes-in-flight.

### It is the kernel, not the cadence (round 59)

`nsys` per-launch duration distribution, which separates "the kernel is slow"
from "the launches are poorly scheduled":

| kernel | launches | min | avg | max |
|---|---|---|---|---|
| `nvfp4_gemm_kernel` | 2880 | **0.992 ms** | 1.226 ms | 2.776 ms |
| `fp8_gemm_kernel` | 3120 | 0.338 ms | 0.629 ms | 2.340 ms |

The **minimum** nvfp4 launch is 0.992 ms, and min-to-max is only 2.8x. So there
is no fast case being diluted by a slow tail: a best-case launch still moves
44.6 MB at ~45 GB/s, a fifth of the roofline. This rules out the cadence, the
tail and the launch count.

That leaves the response surface, which is now measured in three directions and
flat in all of them:

| lever | change | effect on the forward |
|---|---|---|
| occupancy (fp8) | 2 -> 3 blocks/SM | none (round 56) |
| bytes in flight | KC 32 -> 64 | ~2% (round 58) |
| tile shape | TT 64 -> 32 | included above |

A bandwidth-bound kernel that ignores occupancy *and* bytes-in-flight is not
waiting on DRAM, so the next step was a probe rather than a theory.

### The probe, and the fix (round 60)

Disabling the nvfp4 weight staging outright -- wrong results, timing only --
dropped the t=1 forward from 377.31 ms to **227.30 ms**. So the staging pass
alone was 150 ms, moving 9.63 GB at **64 GB/s**, 28% of roofline.

That is the signature of too little memory-level parallelism, not of shared
traffic or ALU. The staging loop issued **one 4-byte load per thread per
iteration**, and with 128 threads x 2 blocks/SM there were only ~256 outstanding
4-byte loads per SM -- about 1 KB in flight where 228 GB/s x ~600 ns needs
~2.8 KB. Occupancy and tile size could never fix that, which is exactly why
both came back flat.

The fix is to issue every load the thread owns *before* consuming any of them.
`stage_wtile` and `stage_wtile_fp8` now run in two passes -- load into a small
register array, then dequantise and store -- so `P = TN*SEGS/BLOCK = 4` loads
are in flight per thread instead of one.

| t | before | after | |
|---|---|---|---|
| 1 | 377.31 | **312.27** | -17% |
| 2 | 378.68 | 312.22 | -18% |
| 4 | 379.00 | 319.96 | -16% |
| 8 | 389.52 | 323.73 | -17% |
| 16 | 401.48 | **341.81** | -15% |

`generate` is still **16/16 exact**. This is the first change in this sequence
that moved the number, and it moves TTFT, which is one of the objective's
targets.

Note what it does *not* say: the earlier flat results were not wasted. They are
what ruled out occupancy and tile shape, leaving memory-level parallelism as the
only remaining explanation -- and that one was then confirmed by a probe before
being acted on.

### Decomposing what is left (round 61)

Same method: disable one stage at a time and read the delta off the t=1 forward.
Each probe is timing-only and was reverted immediately; the tree is at the
round-60 state, `generate` 16/16 exact.

| stage disabled | t=1 forward | implied cost | share |
|---|---|---|---|
| (none) | 312.27 ms | -- | -- |
| nvfp4 weight staging | 227.30 ms | ~85 ms | 27% |
| fp8 weight staging | 256.65 ms | ~56 ms | 18% |
| outer product, 3/4 of k | 263.91 ms | ~64 ms | 20% |
| **remainder** | | **~107 ms** | **35%** |

The probes overlap (removing a stage also removes the compute that consumes it),
so these are upper bounds and the shares do not sum cleanly. What they do say is
that **no single stage dominates any more**: the two stagings are 45% between
them, the outer product 20%, and the largest single bucket is now the
"everything else" remainder at ~35%.

That remainder is `stage_xtile`, the bf16 GEMM, `gemm2d_store`, and the
non-GEMM ops (rmsnorm, rope, and the rest). **The next probe should attack it
directly** -- disable `stage_xtile` and the bf16 path in turn -- because at 107 ms
it is now bigger than any individual stage, and the round-60 result showed the
method works: a probe that isolates one stage is what turns a flat response
surface into a fix.

Also worth noting for scale: fp8's staging moves 5.59 GB in ~56 ms = 100 GB/s,
and nvfp4's moves 9.63 GB in ~85 ms = 113 GB/s. Both roughly doubled from the
64 GB/s that prompted the round-60 fix, so the two-pass load is doing its job --
but both are still around half of the 228 GB/s roofline, so the same MLP
question should be asked of them again with a larger `P`.

### Wider loads: 16 elements per thread (round 62)

Acting on that last sentence. `stage_wtile` now has each thread own 16
consecutive NVFP4 elements rather than 8, fetched as a single `uint2` and
covered by one group scale. That halves both the staging iterations and the
number of load instructions for the same bytes.

| t | round 60 | round 62 | |
|---|---|---|---|
| 1 | 312.27 | **287.58** | -8% |
| 2 | 312.22 | 284.42 | -9% |
| 4 | 319.96 | 288.52 | -10% |
| 8 | 323.73 | 295.93 | -9% |
| 16 | 341.81 | **312.19** | -9% |

`generate` still **16/16 exact**.

Cumulative against the round-59 baseline, both changes together:

| t | baseline | now | |
|---|---|---|---|
| 1 | 377.31 | **287.58** | **-24%** |
| 16 | 401.48 | **312.19** | **-22%** |

The same widening is the obvious next thing for `stage_wtile_fp8`, which still
loads 8 bytes per thread, and for `stage_xtile`, which still loads 16 bytes per
thread but only from the handful of lanes where `t < T` -- so at small `t` almost
all of its work is writing zeros into shared rather than loading.

### Two real bugs found on the way (round 58)

Worth keeping because both produced misleading failures:

1. **The `ops.rs` launch sites.** The GEMMs use a one-line
   `LaunchConfig { ... block_dim: (256,1,1) ... }`, which is *not* the
   `block_dim: (block_for(n, 256), 1, 1)` form used by the rmsnorm kernels. A
   blanket replace of the latter hit five rmsnorm sites and left the three real
   GEMM sites at 256 -- so the kernels declared `__launch_bounds__(128)` and were
   launched with 256, giving `CUDA_ERROR_INVALID_VALUE` on every call.
2. **`GB10_TILE_T` in `ops.rs` must match `GB10_TT` in `gemm.cu`** (the comment
   says so). Leaving it at 64 while the kernel tile became 32 halved the grid's
   `y` extent and silently produced wrong tokens rather than an error.

### The earlier, reverted attempt (round 57)

The KC=64 rework was applied in full and **reverted**. Recording it so the next
attempt starts from the failure rather than from scratch.

What was changed:

* `GB10_TT` 64 -> 32, `GB10_KC` 32 -> 64, `GB10_GEMM_BLOCK` 256 -> 128
* `gemm2d_ids` -> `ty = tid >> 3`, `tx = tid & 7`
* `stage_wtile` / `stage_wtile_fp8` / `stage_xtile` rewritten as strided unit
  loops (`u / SEGS`, `u % SEGS` over `TN x KC/8` and `TT x KC/8`), which is what
  makes them independent of KC and block size
* the bf16 body's tile also moved to uint16 (its `float` tile would have been
  53248 B at KC=64)
* the four GEMM launch sites in `ops.rs` 256 -> 128

**The shared-memory budget came out exactly as designed**: all three GEMMs
report 35840 B, from 26112 B at KC=32. The kernels compiled and the tiles fit.

**But every GEMM launch failed with `CUDA_ERROR_INVALID_VALUE`.** Note the
first mistake on the way: the `ops.rs` edit was a blanket replace of
`block_dim: (block_for(n, 256), 1, 1)` and caught **five** sites, the fifth
being `l2norm_scale`, which is not a GEMM. Reverting that one to 256 did not fix
the failure, so the cause is elsewhere.

Prime suspects for the next attempt, in order:

1. `__launch_bounds__(GB10_GEMM_BLOCK)` is now 128, but something in the launch
   path may still be passing 256 for a kernel that was not among the four --
   check by printing the actual block dim rather than by reading the diff.
2. `block_for(n, 128)` may not do what the name suggests for small `n`; the
   four GEMM sites are not the only users of that helper.
3. `gemm2d_store` / `gemm2d_store_scaled` index the output with `ty`/`tx`, whose
   grouping changed from 16x16 to 16x8. An out-of-range write would not normally
   surface as `INVALID_VALUE`, but the two should be checked together.

The revert restored the verified state: `generate` 16/16 exact, `forward-cost`
back to 386/385/389/397/405 ms.

Step 2 then becomes possible:

1. ~~Store fp8 weights as uint16 (bf16) instead of float.~~ (above)
2. **`GB10_TT` 64 -> 32, `GB10_KC` 32 -> 64, block 256 -> 128.** At
   TN=64/TT=32/KC=64 the shared cost is 35840 B with the `+4` padding intact --
   comfortably inside the 49152 B static limit, no bank conflicts.
3. **Coordinated edits:** `gemm2d_ids` -> `ty = tid >> 3`, `tx = tid & 7`;
   `stage_xtile` needs two passes over the k segments (32 token lanes x 4
   segments x 8 k = 32 k per pass); the launch `block_dim` 256 -> 128.

### Gate

* `nvfp4_gemm_kernel` achieved GB/s, 41 -> **100+**
* `generate` still 16/16 exact, `chunked-prefill` still agrees
* then TTFT

If the bandwidth does not move, revert and stop -- the hypothesis is wrong and
the retiling tables in `docs/TARGETS.md` are the fallback.

## Two things the objective asks for that hardware cannot give

These are arithmetic, not optimisation headroom, and should be renegotiated
rather than chased:

* **Single decoder 100 tok/s** needs ~1.76 TB/s of effective bandwidth. The
  device measures **228 GB/s**, and the model is 17.6 GB/token, so the roofline
  is **~13 tok/s**. 80% of roofline is ~10.4 tok/s.
* **50 tok/s at 16 concurrent** needs ~880 GB/s. Current 40.75 tok/s is already
  a defensible fraction of what the memory system allows.

Everything else in the objective is reachable and has a concrete path: TTFT
(fix the GEMM above), the OpenAI/Anthropic endpoint (already working), MTP
(blocked only on the same GEMM fix), and push-per-round (in place).

## Method notes that have repeatedly paid off

* Profile before optimising. First-principles guesses have been wrong
  repeatedly here (round 35 sharedPerBlock, round 36 delta-rule, rounds 40-41
  MTP economics, round 53 static shared limit).
* A passing happy path proves nothing until a control that should fail does --
  the MTP hidden-zeroed control dropped acceptance 91.7% -> 10.4% and is what
  made the other numbers trustworthy.
* `batch-parity` must use prompts of different lengths, or a per-sequence
  addressing bug passes.
* Change one variable and see whether the failure follows it.
