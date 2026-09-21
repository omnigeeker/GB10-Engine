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

### The same widening on fp8 and xtile (round 63)

`stage_wtile_fp8` now takes 16 elements per thread as one `uint4` (fp8 carries a
single per-tensor scale, so there is nothing to index), and `stage_xtile` takes
16 k-values as four `float4`s. The latter matters most at small `t`: it used to
be one iteration of almost entirely zero-writes whenever `t < T`, which at t=1
is every thread but one.

| t | round 62 | round 63 | |
|---|---|---|---|
| 1 | 287.58 | **271.46** | -6% |
| 2 | 284.42 | 271.85 | -4% |
| 4 | 288.52 | 273.77 | -5% |
| 8 | 295.93 | 280.94 | -5% |
| 16 | 312.19 | **292.85** | -6% |

`generate` still **16/16 exact**.

### Cumulative

| t | round-59 baseline | now | |
|---|---|---|---|
| 1 | 377.31 | **271.46** | **-28%** |
| 4 | 379.00 | 273.77 | -28% |
| 16 | 401.48 | **292.85** | **-27%** |

Three changes -- two-pass loads (round 60), 16 elements per thread for nvfp4
(round 62), and the same for fp8 and xtile (round 63) -- have taken the prefill
forward down by ~28%. All three came from the same method: isolate one stage with
a timing probe, then fix the specific thing the probe exposed.

### Re-decomposition against the current numbers (round 64)

The round-61 shares had moved, so the same probes were re-run against the
271.46 ms baseline:

| stage disabled | t=1 forward | implied cost | share |
|---|---|---|---|
| (none) | 271.46 ms | -- | -- |
| nvfp4 weight staging | 200.52 ms | ~71 ms | 26% |
| fp8 weight staging | 230.03 ms | ~41 ms | 15% |
| outer product, 3/4 of k | 229.77 ms | ~56 ms | 21% |
| xtile loads | 289.61 ms | none measurable | -- |

Two things stand out.

**The two stagings together are still 41%** -- 112 ms to move 15.2 GB, i.e.
~136 GB/s, up from 64 GB/s before round 60 but still only 60% of the 228 GB/s
roofline. The widening helped; it did not finish the job.

**The remainder is now the largest single bucket at ~103 ms (38%)**, and the
`xtile` probe came back with *no* measurable effect -- removing its global loads
changed nothing. Since `stage_xtile` still writes every one of its shared slots
(zeros for the `t >= T` lanes, which at t=1 is all but one lane per group), that
points at the **shared-memory stores** rather than the loads, or at something
outside `stage_xtile` entirely: `gemm2d_store`, the bf16 GEMM, or the non-GEMM
ops.

Note the probe for `xtile` was the weakest of the four -- it disabled the load
branch only, so a null result there is suggestive rather than conclusive, and it
should be redone by skipping the whole function.

### The xtile probe redone, and a fix that did not work (round 65)

The round-64 `xtile` probe was the weakest of the four -- it disabled only the
load branch. Redone by skipping the whole function:

| | t=1 forward |
|---|---|
| baseline | 271.46 ms |
| all `stage_xtile` calls disabled | **200.52 ms** |

So `stage_xtile` really does cost ~71 ms (26%), and since disabling the loads
alone had shown nothing, the cost is in the **shared stores**.

At `t=1` the block is partial (`T=1 < TT=32`), so 31 of every 32 columns take
the `else` branch and write zeros -- and `gemm2d_store` guards every token with
`if (t < T)`, so those columns are never read back. That looked like 71 ms of
dead work, and removing it was implemented and measured:

| t | with zero-fill | without | |
|---|---|---|---|
| 1 | 271.46 | 278.66 | +2.6% |
| 4 | 273.77 | 280.75 | +2.6% |
| 16 | 292.85 | 307.80 | **+5.1%** |

`generate` and `chunked-prefill` both stayed correct, but it is **slower**, so it
was reverted. The likely reason is divergence: with the zero-fill every thread
runs the same store sequence, while removing it makes the `t < T` lanes skip out
of the inner loop and splits the warp. The stores were dead, but they were also
free -- they filled a branch that would otherwise have diverged.

That is a useful negative result. It means `stage_xtile`'s 71 ms is not
recoverable by deleting work; it needs the stores restructured so the tile is
filled without per-lane branching, or the tile layout changed so that partial
blocks do not need 32 columns at all.

### Where the widening stops paying (round 66)

The widening from 8 to 16 elements per thread paid (round 62, -8%). Carrying it to
32 does not:

| elements/thread | loads in flight | t=1 | t=16 |
|---|---|---|---|
| 8 | P=4 x 4 B = 16 B | 312.27 | 341.81 |
| **16** | **P=2 x 8 B = 16 B** | **271.46** | **292.85** |
| 32 | P=1 x 16 B = 16 B | 276.11 | 301.65 |

All three configurations put the same 16 bytes in flight per thread, so the
bytes-in-flight is not what separates them. What changes is the *number of
independent loads*: 4, then 2, then 1. Going 8 -> 16 helped because it halved the
instruction count without reducing the load count below 2; going 16 -> 32 crosses
that line and buys instruction count at the cost of memory-level parallelism.

So **16 elements per thread with P=2 is the optimum for this geometry**. Note
that `P` itself is not the lever: the total loads in flight is `UNITS`, not
`P * BLOCK`, so redistributing the same units over fewer threads changes nothing.
The lever is `UNITS = TN * KC/16`.

That was tested directly by doubling `TN` (64 -> 128) and halving `TT` (32 -> 16),
which doubles `UNITS` to 512 and still fits the shared budget at 44032 B:

| | t=1 | t=16 | registers | smem |
|---|---|---|---|---|
| TN=64, TT=32 | **271.46** | **292.85** | 52 | 35840 |
| TN=128, TT=16 | 303.62 | 317.10 | **122** | 44032 |

Correct (`generate` 16/16) but **12% slower**. Doubling `TN` doubles the live
accumulator and index state per thread -- 122 registers against 52 -- and pushes
the tile to one block per SM. The extra loads in flight do not compensate.

So the geometry is boxed in from both sides: `UNITS` cannot grow without `TN`,
and `TN` cannot grow without register pressure.

### Occupancy, tested a second time (round 68)

The remaining hypothesis was the shared-memory budget capping occupancy at 2
blocks/SM. Dropping the `+4` padding (`WSTRIDE = TN`, `XSTRIDE = TT`, both still
multiples of 4 so the `float4`/`ushort4` loads stay aligned) fits a third block:

| | t=1 | t=16 | registers | smem | blocks/SM |
|---|---|---|---|---|---|
| padded | **271.46** | **292.85** | 52 | 35840 | 2 |
| unpadded | 271.99 | 294.19 | 104 | 32768 | 3 |

Correct (`generate` 16/16) and **completely neutral** -- and it cost registers,
52 -> 104, presumably from the different unrolling. Reverted.

This is the second independent test of occupancy on this kernel (the first was
fp8 going 2 -> 3 blocks/SM in round 56) and both came back flat. Combined with
the flat response to tile shape and to bytes in flight, the picture is now
consistent: **the GEMM is not waiting on DRAM, and it is not short of warps.**

That leaves the ~271 ms as internal work: the dequantisation and shared stores
in the staging passes, the outer product's shared reads, and the non-GEMM ops.
The round-65 result -- that `stage_xtile`'s 71 ms is stores, and that removing
the dead zero-fill made things *slower* through divergence -- is the clearest
clue about what kind of change would help: not deleting work, but removing
per-lane branching from the shared-memory fill.

Worth noting for perspective: 271 ms for 17.6 GB is 65 GB/s, while the decode
GEMV reaches 148 GB/s on the same weights. So the ceiling is not the memory
system; the GEMM path is roughly 2.3x less efficient than the GEMV path at
identical traffic.

### Three ways to fill the xtile, and the boring one wins (rounds 65-69)

`stage_xtile` costs ~71 ms (26%) of the forward and the cost is in its shared
stores. Three variants have now been measured:

| variant | t=1 | t=16 | |
|---|---|---|---|
| zero-fill the `t >= T` lanes (original) | **271.46** | **292.85** | best |
| skip those lanes entirely | 278.66 | 307.80 | +2.6% / +5.1% |
| branchless: redirect them to a spare column | 289.92 | 310.48 | +6.8% / +6.0% |

All three are correct (`generate` 16/16, `chunked-prefill` agrees). The original
wins, and the reason is now clear from having tried both alternatives:

* Skipping the lanes splits the warp, and the divergence costs more than the
  dead stores saved.
* Redirecting them keeps the store sequence uniform but makes those lanes load
  token `t0` as well, so it *adds* real global traffic to remove a branch.

So the 71 ms is not recoverable by touching the fill logic at all: the dead
stores are cheaper than either removing them or paying to keep them uniform.
Any further gain there has to come from a different tile layout that does not
need `TT` columns when the prompt is shorter than `TT` -- not from the fill.

That closes out the `stage_xtile` line of attack, and with it the shared-store
hypothesis from round 65. The remaining measured shares are the two weight
stagings (41% together, ~136 GB/s) and the outer product (21%).

### Where the two weight stagings stand (round 70)

Arithmetic on the staging, to work out what its 136 GB/s actually is:

* Coalescing is fine. Thread `u` reads `w + n*(K/2) + (c*KC + gr*16)/2` with
  `gr` fastest, so threads 0-3 cover 32 consecutive bytes of one row and a warp
  covers 8 rows x 32 B = 256 B in 8 fully-used sectors.
* Memory-level parallelism is fine. The two-pass loop issues 4 independent loads
  (2 units x 1 weight + 1 scale) per thread, 32 B in flight, ~8 KB per SM against
  the ~2.8 KB needed to cover 600 ns at 228 GB/s.
* Latency per chunk cannot explain it either: 80 chunks x 600 ns is 48 us per
  block, and ~2.8 waves of blocks gives ~0.14 ms -- nowhere near the 71 ms
  measured for the nvfp4 staging.

So 136 GB/s is a *bandwidth* figure, not a latency one, and neither coalescing
nor outstanding loads explain the shortfall from 228 GB/s. That leaves two
candidates, and they are cheap to tell apart:

1. **The dequantisation ALU.** Per 8-byte load the staging runs 16 `e2m1_to_float`
   calls plus 16 bf16 converts and 16 multiplies. If `e2m1_to_float` is a branch
   or a lookup rather than pure bit arithmetic, that is a lot of work per byte.
2. **The shared-memory stores**, which are bank-conflicted: `wt` has stride
   `TN+4 = 68` uint16 = 136 B = 34 banks, so consecutive `nl` land 2 banks apart
   and 32 threads hit 16 banks twice.

The discriminator is to replace `e2m1_to_float(nib) * s` with a plain constant
inside the staging (wrong results, timing only, as in round 60). If the staging
time collapses, it is the ALU and the fix is a cheaper dequant -- a 16-entry
`__constant__` table indexed by the nibble would do it, or hoisting the scale
multiply into the outer product where it is already done for fp8. If it does not
move, it is the shared stores and the fix is the tile layout.

This is the same method that produced every gain so far: isolate one thing,
measure, then act.

### Both candidates tested, both ruled out (rounds 71-72)

**Dequantisation ALU.** The 16 `e2m1_to_float` calls per 8-byte load were
replaced with a cheap value derived from the same register (keeping the loads
alive, timing only):

| | t=1 |
|---|---|
| baseline | 271.46 ms |
| dequant ALU removed | 280.85 ms |

No gain. **The ALU is not the limit.**

**Bank conflicts in the `wt` stores.** `WSTRIDE = TN+4 = 68` uint16 = 136 B =
34 banks, so rows land 2 banks apart and a warp's four rows overlap; padding to
`TN+16 = 80` puts rows 8 banks apart and makes the pattern conflict-free:

| | t=1 | t=16 |
|---|---|---|
| WSTRIDE = 68 | **271.46** | 292.85 |
| WSTRIDE = 80 | 272.26 | 290.42 |

Neutral. **Bank conflicts are not the limit either.**

### What that leaves

Every explanation on the list has now been tested and come back flat: DRAM
bandwidth, occupancy (twice), bytes in flight, load count, tile shape, load
width, dequant ALU, shared-store bank conflicts, and per-chunk latency (which
arithmetic rules out by three orders of magnitude -- 0.14 ms predicted against
71 ms measured).

The staging moves 9.63 GB in ~71 ms, which is 136 GB/s against a 228 GB/s
roofline. Since none of the mechanisms above explain the gap, the next thing to
question is **the roofline number itself**. 228 GB/s came from `bench/hw/bw4.cu`,
an incompressible random-access probe. The staging's pattern is different in two
ways that probe does not capture: a warp reads 8 rows that are 2560 B apart
rather than one contiguous run, and the working set is 44.6 MB of weights
streamed through a 25 MB L2. Either could hold the achievable rate below 228.

**The cheap next step is therefore to measure the achievable rate for this
exact pattern** -- a standalone probe that reads the real weight layout with the
real warp access pattern and nothing else.

### The roofline was wrong (round 73)

`bench/hw/stage_bw.cu` does exactly that: two kernels, same bytes, one reading
contiguously and one reproducing `stage_wtile`'s addresses with no dequant and
no shared store. On a 44.6 MB matrix (one layer's NVFP4 weights), 64 reps:

| pattern | GB/s |
|---|---|
| contiguous 16 B per thread | 206.2 |
| **staging (the real addresses)** | **89.7** |

**The staging pattern's own ceiling is ~90 GB/s, not 228.** The real kernel
reaches 136 GB/s on this pattern -- *above* what the isolated probe manages, so
the staging is not leaving performance on the table at all. It is at the limit of
the addresses it issues.

The reason is cache-line granularity. `KC = 64` NVFP4 elements is 32 bytes, and a
warp's four `gr` lanes cover exactly those 32 bytes of a row; the next row starts
2560 bytes away. So every row contributes a **32-byte sector to a 64-byte line**
-- half of each fetched line is never used. That predicts ~114 GB/s against the
228 GB/s contiguous figure, which is the right order for both the 90 measured in
isolation and the 136 measured in the kernel.

This reframes the whole line of attack. The two weight stagings were 41% of the
forward, but they are not underperforming; they are reading 32 bytes per row when
the hardware wants 64. **The fix is a layout change, not a tuning change**: get a
warp to cover 64 consecutive bytes of one row. The direct route is `KC = 128`,
which makes a row's chunk exactly one cache line -- but that needs
`2*128*(TN+4)*2 + 2*128*(TT+4)*4 = 71680 B` of shared against a 49152 B budget, so
it requires shrinking `TT` or storing `xt` in bf16 (both already scoped above).

That is the next experiment, and it is the first one in several rounds with a
clear mechanism behind it rather than an elimination.

### The hypothesis tested before acting on it (round 74)

Rather than restructure the kernel on the strength of a mechanism, the probe was
extended with a third pattern: same layout, same bytes, but 8 lanes x 8 B so a
warp covers **64 consecutive bytes of a row** instead of 32.

| pattern | bytes per row per warp | GB/s |
|---|---|---|
| contiguous | -- | 221.5 |
| staging32 (current) | 32 | 93.7 |
| **staging64** | **64** | **176.9** |

**Confirmed, and the effect is large: 93.7 -> 176.9 GB/s, +89%.** Cache-line
granularity is the mechanism; half of every fetched line was being discarded.

Projected on the real forward: the two stagings cost ~112 ms (41%) at ~136 GB/s.
At 177 GB/s the same bytes take ~86 ms, so **t=1 should fall from 271 to roughly
245 ms (-10%)**, and further if the kernel tracks the contiguous figure rather
than the isolated one.

### The change this implies, and its cost

`KC` has to become 128, because a row's chunk is `KC/2` bytes and only `KC = 128`
makes it a full 64-byte line. `KC = 128` divides every K in the model (5120/128 =
40, 17408/128 = 136, 6144/128 = 48, 10240/128 = 80), so the constraint is met.

The obstacle is shared memory. With `KC = 128`, `TN = 64`, `TT = 32`:

```
wt  2*128*(64+4)*2 = 34816
xt  2*128*(32+4)*4 = 36864
                    ------
                    71680   vs a 49152 budget
```

The options, in order of intrusiveness:

| wt | xt | total | fits |
|---|---|---|---|
| `TT=32`, xt float, pad 4 | 34816 + 36864 | 71680 | no |
| `TT=16`, xt float, pad 4 | 34816 + 20480 | 55296 | no |
| `TT=32`, xt bf16, pad 4 | 34816 + 18432 | 53248 | no |
| `TT=16`, xt bf16, pad 4 | 34816 + 10240 | **45056** | **yes** |
| `TT=32`, xt bf16, no pad | 32768 + 16384 | **49152** | exactly at the limit |

So it needs `TT = 16` **and** a bf16 `xt` tile, or a bf16 `xt` with no padding.
`TT = 16` also changes `gemm2d_ids` (32 row-groups x 4 token-groups) and
`GB10_TILE_T` in `ops.rs`, and a bf16 `xt` changes every `gemm2d_outer` reader.
That is a coupled change across three places, so it should be done in one step
with the `generate` 16/16 gate as the check, and reverted as a whole if the
timing does not improve.

**The cheapest version to try first is `TT=32` + bf16 `xt` + no padding**, since
it touches only the `xt` type and the two stride macros and leaves the tiling
alone -- at the cost of sitting exactly on the 49152 B limit, which may need a
little slack.

### A structural point that changes the cost (round 75)

Working out the `KC = 128` variants turned up something worth writing down before
any code changes, because it moves the problem.

The current tile is already **128 k-rows deep in total**: `wt[2][KC][WSTRIDE]`
holds two `KC = 64` chunks, 17408 B. A single `wt[128][WSTRIDE]` tile is *also*
17408 B. So `KC = 128` costs nothing in shared memory **if the double buffer is
given up** -- the same space simply holds one bigger k-tile instead of two
alternating ones.

That matters because it means the obstacle is not really shared memory. It is
that dropping the double buffer **serialises staging against the outer product**.
With staging projected at ~86 ms (at 177 GB/s) and the outer product at ~56 ms,
serialised they cost ~142 ms against the ~112 ms they cost overlapped today. The
cache-line win would be spent, and then some.

So `KC = 128` is only worth doing if the pipeline survives it. The shapes that
fit with double buffering all force `TT = 16` (and therefore a reworked
`gemm2d_ids`, plus a second accumulator group per thread at 128 threads), and the
ones that keep `TT = 32` sit exactly on the 49152 B limit.

The clean resolution is a **three-deep k-pipeline**: keep 128 k-rows of weights
resident, but stage the *next* 64 k-rows into the half that has just been
consumed, so the staging read still covers 64 contiguous bytes per row while
staying overlapped with compute. That needs the outer product to consume the two
halves in order, which is exactly what the current `cur`/`nxt` structure already
does -- the change is to stage both halves in one pass at the top and then
compute both, rather than staging one and computing one.

That is the version to build. It keeps `TN = 64`, `TT = 32`, the float `xt`
tile, and the 35840 B budget, and it changes only the staging/consumption order
in the three GEMM bodies.

### Built it, and the cache-line story is wrong (round 76)

The paired version was implemented for nvfp4 -- `stage_wtile_pair` stages two
adjacent k-chunks in one pass, so a warp's four `pr` lanes cover 64 contiguous
bytes of a row, and the body computes both halves before restaging. Correct
(`generate` 16/16) and **completely neutral**:

| | t=1 | t=16 |
|---|---|---|
| baseline | 271.46 | 292.85 |
| paired staging (64 B/row) | 271.18 | 295.43 |

Reverted.

**This falsifies the cache-line explanation.** Doubling the bytes a warp touches
per row, with the load count unchanged, does nothing. So the probe's 93.7 ->
176.9 GB/s was **not** caused by cache-line utilisation.

The probe was confounded. `read_staging64` changed two things at once:

* 8 lanes x 8 B per row instead of 4 lanes x 8 B -- the bytes-per-row difference
* `P = 4` instead of `P = 2` -- because `UNITS` went from 256 to 512

so it also doubled the number of independent loads in flight. The kernel
experiment separates the two and shows the bytes-per-row half is worth nothing.
The effect the probe measured must belong to the load count.

That is consistent with round 66, which found 16 elements per thread optimal and
32 worse: at 32 the loads in flight halve. So the through-line across rounds 60,
62, 66 and now 76 is **loads in flight**, and the honest state is that the
staging's 136 GB/s is explained by neither cache lines (now tested) nor any of
the other mechanisms eliminated above.

**The lesson is about the probe, not the kernel.** A probe that changes two
variables at once is as misleading as a guess, and it is more dangerous because
it looks like evidence.

### The clean 2x2, and why the probe still does not transfer (round 77)

The probe was rebuilt to vary one thing at a time -- three patterns, with the
load count and the bytes-per-row separated:

| pattern | bytes/row | loads per chunk | GB/s |
|---|---|---|---|
| 4 lane x 8 B | 32 | 256 | 90.4 |
| 8 lane x 8 B | 64 | 512 | 184.9 |
| 8 lane x 4 B | 32 | 512 | 102.7 |

* 8lane x 4B vs 8lane x 8B (load count fixed, bytes/row varies): 102.7 -> 184.9,
  **+80%**. Bytes-per-row is the dominant variable.
* 4lane x 8B vs 8lane x 4B (bytes/row fixed, load count varies): 90.4 -> 102.7,
  +14%. Load count is minor.

So the cache-line mechanism is real, and round 76's failure is explained: my
paired staging used **two 8-byte loads** that together covered 64 bytes of a row,
but the coalescer works per *instruction*, so each load still presented 8 rows x
32 B. Two requests covering the same line are not one request covering the line.

That was fixed -- a single **16-byte load** per thread, 4 lanes per row, so one
instruction's warp footprint is 8 rows x 64 contiguous bytes. The result:

| t | baseline | single 16 B load, 64 B/row |
|---|---|---|
| 1 | 271.46 | 270.21 |
| 2 | 284.42 | 271.11 |
| 4 | 273.77 | 269.78 |
| 16 | 292.85 | 290.75 |

Correct (`generate` 16/16), better at every point, and **every one of those
differences is inside the +-2% run-to-run noise measured on this box**. It is not
a defensible win, so it was reverted rather than kept on the strength of a
consistent sign.

**The conclusion is that per-warp request shape does not bind in the real
kernel**, even though it clearly binds in isolation. The likely reason is
concurrency: the isolated probe runs one warp's pattern against an otherwise idle
memory system, while the kernel has many warps from two blocks per SM issuing
interleaved requests, so the memory controller sees a much denser stream than any
single warp's shape suggests. A pattern that is 2x worse in isolation can be
indistinguishable when 20 other warps are filling the gaps.

That is worth stating plainly because it bounds what the probe can be used for:
`bench/hw/stage_bw.cu` is good for **falsifying** a mechanism (if a pattern is
slow in isolation it will not be fast in the kernel) but not for **predicting a
gain** from a pattern change.

### The kernel-level breakdown, which should have come first (round 78)

Every share quoted above came from probe deltas, which the last three rounds have
shown are unreliable in both directions. `nsys profile --stats=true` on
`forward-cost` gives the actual distribution instead:

| kernel | % GPU time | launches | avg | min |
|---|---|---|---|---|
| `nvfp4_gemm_kernel` | 32.7 | 2880 | 825.6 us | 635.9 us |
| `nvfp4_gemv_kernel` | 24.9 | 5998 | 301.2 us | 213.9 us |
| `fp8_gemm_kernel` | 20.1 | 3120 | 468.3 us | 238.3 us |
| `fp8_gemv_kernel` | 17.1 | 6448 | 192.9 us | 26.0 us |
| `rmsnorm_zero_centered` | 1.1 | 7406 | 11.0 us | 1.3 us |
| `bf16_gemv_batch` | 0.9 | 1344 | 46.8 us | 12.1 us |
| `gated_delta_rule_chunk` | 0.9 | 720 | 86.9 us | 34.8 us |
| `gated_delta_rule_step` | 0.7 | 1488 | 36.2 us | 26.5 us |
| everything else | <1% each | | | |

**The four GEMM/GEMV kernels are 94.8% of GPU time. Every non-GEMM op together
is ~5%.** The round-61/64 probes said the "remainder" was 35-38% and that
non-GEMM work was a plausible target; the profiler says it is not. Optimising
`rmsnorm`, `swiglu`, `rope` or the delta rule cannot pay more than ~5%.

**The number that matters is this.** `nvfp4_gemm_kernel` moves 9.2 GB per forward
across 15 forwards in 2.38 s, so the GEMM path runs at **~58 GB/s** (70 GB/s at
its minimum). `fp8_gemm_kernel` independently lands at the same ~58 GB/s. The
GEMV path, on the same weights, reaches ~148 GB/s.

So the finding is not about staging, cache lines, or shared stores: **the GEMM
path as a whole is 2.5x less efficient than the GEMV path at identical traffic,
and it is 94.8% of the time.** That is the thing to explain, and the profiler
gives a per-launch number to optimise rather than a probe delta.

A first estimate of where a launch's 825 us goes, for the MLP gate (N=17408,
K=5120, 272 blocks, 5.7 blocks/SM, ~2.8 waves at 2 blocks/SM):

* 164 KB of weights per block, 80 chunks of 2 KB each
* ~2.8 us per chunk measured
* the outer product is 1024 FMA per thread per chunk -- ~1.2 us per chunk at
  128 FP32 lanes/SM shared by two blocks
* the loads themselves should be ~0.6 us

which leaves roughly 1 us per chunk unaccounted for.

### Splitting the GEMM: staging 80%, outer product 20% (round 79)

Quartering `gemm2d_outer_bf16`'s k loop and reading the t=1 forward:

| | t=1 |
|---|---|
| baseline | 271.46 ms |
| outer product k-loop at 1/4 | 231.38 ms |

so the full outer product costs (271.46 - 231.38) / 0.75 = **~53 ms, 20% of the
forward**. The remaining ~218 ms is the staging, which the profiler puts at
~58 GB/s.

Putting the two measurements together, the t=1 forward decomposes as:

| part | ms | share | rate |
|---|---|---|---|
| weight staging (nvfp4 + fp8 + bf16) | ~218 | 80% | ~58 GB/s |
| outer product | ~53 | 20% | -- |
| non-GEMM ops | ~13 | 5% | -- |

**And the GEMV path moves the same weights at ~148 GB/s.**

So the isolated probe was right about the *relative* ordering after all: 32 B/row
measured 90 GB/s against 206 GB/s contiguous, and the kernel shows 58 GB/s
against 148. The probe predicted the comparison correctly; what it could not
predict is that my particular fix would not move it, because the fix changed the
bytes per row without changing the shape that actually matters.

The shape that matters is how many *different rows* a single load instruction
touches. The staging's load presents **8 rows x 32 B** to the coalescer; the
probe's fast pattern presented **4 rows x 64 B**; the GEMV streams one row
contiguously. My round-77 attempt produced **8 rows x 64 B** -- more bytes, same
eight-way row scatter, which is why it was neutral.

### Rows per instruction is not the variable either (round 80)

The 4-row hypothesis was tested by adding a fourth pattern that reproduces
**exactly** what round 77 built into the kernel -- a single 16-byte load per
thread, 4 lanes per row, so one instruction's footprint is 8 rows x 64 B:

| pattern | rows per instruction | bytes/row | GB/s |
|---|---|---|---|
| 4 lane x 8 B | 8 | 32 | 89.8 |
| 8 lane x 8 B | 4 | 64 | 176.9 |
| 8 lane x 4 B | 4 | 32 | 101.8 |
| **4 lane x 16 B (round 77's fix)** | **8** | **64** | **181.0** |

**8 rows x 64 B is just as fast as 4 rows x 64 B.** Rows per instruction is not
the variable; bytes per row is (32 B -> ~90-102, 64 B -> ~177-181).

**Which makes round 77's null result conclusive rather than puzzling.** The
kernel change did produce the 8-row x 64 B pattern, and that pattern measures
181 GB/s in isolation -- yet it was neutral in the kernel. So the staging is
**not limited by its DRAM access pattern at all**. Changing a 90 GB/s pattern
into a 181 GB/s pattern bought nothing.

This closes the DRAM line for good. Three independent attempts have now been made
to exploit the pattern (rounds 60, 76, 77) and the profiler's 58 GB/s has
survived all of them.

**What is left for the staging is the work that is not the load**: the 16
`e2m1_to_float` dequantisations and 32 shared stores per thread per unit, and the
shared traffic the outer product then reads back. Round 71 removed the
dequantisation ALU from the *old* staging and saw nothing, but that was before the
pattern was known to be irrelevant; it should be repeated, and the shared stores
tested the same way, now that the load side is ruled out.

Note also what this says about the probe: it correctly ranked the patterns, and
it correctly showed the round-77 pattern was fast. It simply cannot tell you
whether a fast pattern is what the kernel is waiting on. **A probe measures what
a pattern can reach, never what the kernel is blocked by.**

### The dequant and the stores are ruled out too (round 81)

With the load side excluded, the remaining staging work was cut 4x -- the
inner `j` loop in `stage_wtile` from 8 to 2, so one quarter of the
`e2m1_to_float` calls and one quarter of the shared stores, with the loads kept
live through `lo`/`hi`:

| | t=1 |
|---|---|
| baseline | 271.46 ms |
| staging dequant + stores at 1/4 | 266.84 ms |

**1.7%.** So the dequantisation and the shared stores are not the cost either.

That is now three separate exclusions inside the staging:

| mechanism | test | result |
|---|---|---|
| load pattern | probe + kernel change (76, 77, 79) | not the cost |
| dequant ALU | 4x fewer calls (71, 81) | not the cost |
| shared stores | 4x fewer stores (81) | not the cost |

### What that actually means

The staging was attributed **218 ms of 271 (80%)** by disabling it (round 64/71:
`if (false) stage_wtile(...)` -> 200.52 ms). But every component of the staging
has now been individually tested and none of them is expensive. Those two facts
cannot both be true of a well-behaved probe.

**The disable-probe is the thing that is wrong.** Setting `stage_wtile` to
`if (false)` does not merely remove the staging work: it removes the global
loads entirely, changes the kernel's shared-memory footprint and its scheduling,
and leaves the outer product reading uninitialised shared memory. It measures
"the kernel without this stage", which is not the same as "the cost of this
stage", and the difference is large.

So the decomposition quoted for the last several rounds -- staging 80%, outer
product 20% -- should not be trusted, and neither should the round-61/64 numbers
built on it. The outer-product figure (53 ms) came from shortening its k loop,
which is a genuine component test and does survive; the staging figure came from
a disable-probe and does not.

**The rule from here: change a component's size and measure, never disable a
component and subtract.** And prefer the profiler's per-launch number, which is
absolute.

The profiler numbers stand on their own and remain the target:
`nvfp4_gemm_kernel` 825.6 us average, 635.9 us minimum, moving 44.6 MB per
launch -- 54 GB/s average, 70 GB/s at best, against 148 GB/s for the GEMV path on
the same weights.

### A valid component test, and the gap located (round 82)

Following the new rule -- change a component's *size*, read the profiler's
absolute per-launch number -- the outer product's k loop was halved:

| `nvfp4_gemm_kernel` | avg | min |
|---|---|---|
| baseline | 825.6 us | 635.9 us |
| outer product k at 1/2 | 740.6 us | 532.2 us |
| **outer product, doubled back** | **~170 us** | **~207 us** |

That is 21-25% of the launch, and it agrees with the t=1 estimate (53 ms of 271,
20%) that came from the same kind of test. **The two independent methods agree,
which is what a valid measurement looks like.**

Combining with round 81 (dequant + shared stores: a 4x reduction moved the whole
forward by 4.6 ms, i.e. ~2 us per launch -- negligible), the launch decomposes as:

| part | per launch | share |
|---|---|---|
| weight loads | **~630 us** | ~76% |
| outer product | ~170 us | ~21% |
| dequant + shared stores | ~2 us | ~0.2% |

**So the loads move 44.6 MB in 630 us = 71 GB/s average, 104 GB/s at the
minimum -- while the pattern probe says this exact access pattern can reach
181 GB/s.**

That is the gap, now located precisely and by a sound method: **the kernel's
weight loads run at 39% of what their own access pattern is capable of.** It is
not the pattern (181 GB/s is achievable), not the dequant, not the stores, and
not the outer product.

What differs between the probe and the kernel is **concurrency**. The probe runs
272 blocks of pure loads against an idle memory system. The kernel's blocks stage
one chunk, `__syncthreads()`, run the outer product, `__syncthreads()`, stage the
next -- so loads are issued in bursts separated by a barrier, and at any instant
far fewer loads are in flight than the probe sustains. The outer product is only
~27% as long as the loads, so it cannot hide them.

**That points at the pipeline shape, which is exactly what round 74 flagged and
then set aside.** The next experiment is to overlap more: either issue the next
chunk's loads before the barrier rather than after it, or deepen the k pipeline
so that two chunks are always in flight. Both are size changes to the existing
structure, and both can be judged against the 825.6 us launch time rather than
against a probe.

### A caution learned in round 64

The probes were scripted with a `cp` restore from a scratch copy that predated
round 63, which silently reverted that change. Always restore probes with
`git checkout`, and check `git status` after.

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
