# SESSION HANDOFF (read this first)

## T1 is a bandwidth-efficiency target, and it is narrow (round 123)

`docs/TARGETS.md` records the owner's revised, hardware-feasible contract -- the
original 100 tok/s was shown to be physically unreachable before implementation
began. **T1 is single-stream decode >= 12.5 tok/s**, i.e. 95% of the 12.95 tok/s
conservative roofline.

Putting the measured numbers against it:

| | ms/step | GB/s | % of the 228 GB/s peak | tok/s |
|---|---|---|---|---|
| **B=1 measured** | 104.2 | **169.0** | 74% | 9.59 |
| weight floor (228 GB/s) | 77.2 | 228.0 | 100% | 12.95 |
| **T1 target** | 80.0 | 220.1 | 96.5% | 12.5 |

**So T1 is not a new algorithm -- it is 169 GB/s becoming ~220 GB/s on the
single-sequence GEMV path**, a 1.30x lift, with no change to what is computed.

Two things make this the right next target rather than more prefill work:

* **It is the only target in the contract that is close and quantified.** The
  engine is at 74% of a measured hardware limit; the prefill GEMM is at 32 GB/s
  against its own 58 GB/s ceiling with **six candidate mechanisms already ruled out
  by measurement** (rounds 102-111). The decode GEMV has never been through that
  process -- no probe, no per-launch breakdown, nothing ruled out.
* **otp is already a win over llama.cpp (8.48 vs 7.63, +11.1%) and T1 would widen
  it to +64%.** The objective asks for otp to beat llama.cpp, and this is the path
  that does it rather than the path that defends a lead.

### The B=1 step, decomposed for the first time (round 124)

`nsys` on `forward-cost`, using the non-batch kernels (the `t <= 16` cut, so the
small-`t` points exercise the same code the single stream uses):

| kernel | launches/step | avg | total/step |
|---|---|---|---|
| `nvfp4_gemv_kernel` | 192 | **274.3 us** | **52.7 ms** |
| `fp8_gemv_kernel` | 208 | **174.2 us** | **36.2 ms** |
| **GEMV total** | 400 | | **88.9 ms = 85% of the 104.2 ms step** |

The launch counts are exactly the model's matrix counts (64 layers x 3 NVFP4,
64 x 3.25 FP8), so this is the whole decode and nothing else. The remaining ~15 ms
is `gated_delta_rule_{chunk,step}_kernel` and `rmsnorm_zero_centered_kernel`.

**The interesting number is the gap between the average and the median:**

| | per launch, on 44.6 MB | GB/s | % of 228 peak |
|---|---|---|---|
| median | 234.6 us | **190.0** | **83%** |
| average | 274.3 us | **162.6** | **71%** |

So **the typical launch is already at 83% of peak** and the tail is what costs the
difference -- stddev 302 us against a 234.6 us median, with a **maximum of
4,858 us, 20x the median**. There are 6,190 launches in the profile and the tail is
expensive enough to move the mean by 12 percentage points of bandwidth.

**Correction (round 125): the tail is not a pathology, it is `lm_head`.** The
launches over 3x the median are **118 of 6,190 = 1.91%**, but they carry **253.6 ms
= 15% of all `nvfp4_gemv_kernel` time**. Their positions (3243, 4145, 4348, 4560,
6361, ...) are scattered, so it is **not** L2-cold first touches. Their durations --
3230, 1361, 1477, 4288, 3206, 1549, 1302, 3220 us -- are what settles it:

* `lm_head` is **248,320 x 5,120 NVFP4 = 636 MB**, 14x a normal MLP matrix's 44.6 MB.
  At the median's 190 GB/s that is **3,347 us**, which is the ~3.2 ms cluster.
* The ~1.3-1.5 ms launches are proportionally-sized other matrices.

**So the slow launches are running at the same bandwidth as the fast ones -- they
are just bigger.** There is no tail to fix. The mean of 274 us is simply the
weighted average of 44.6 MB and 636 MB matrices, and I mis-read it as variance.

**The real number is that the streaming GEMV runs at ~190 GB/s = 83% of the measured
228 GB/s peak, consistently, across 6,190 launches.** That gives 17.608 GB / 190 GB/s
= **92.7 ms/step** against the measured 104.2, and T1 at 220 GB/s would be
**80.0 ms**. **So T1 is +15.8% of bandwidth efficiency on a kernel that is already
consistent -- not a bug hunt, an efficiency question.**

### Bytes-in-flight is not the limit either (round 126)

Round 125 proposed checking outstanding loads. Done:

| | |
|---|---|
| kernel | `nvfp4_gemv_kernel` -> `nvfp4_gemv_tmpl<1>` |
| ROWS | **1** -- one 8-byte `uint2` weight load in flight per thread |
| registers | **39**, 0 spills |
| occupancy | register count is far below the limit, so 2048 threads/SM = 64 warps |

Bytes in flight: 64 warps x 32 lanes x 8 B = **16 KB per SM**, 786 KB across 48 SMs.
At 228 GB/s and a ~350 ns DRAM latency you need ~80 KB. **So there is ~10x more in
flight than required, and latency is not what is costing the 17%.** The 39-register
count also confirms occupancy was never the issue on this kernel.

**The remaining structural suspect, and it is the round-101 one seen from a new
angle:** with `ROWS`=1 the x-vector is re-read once per row, so per matrix the x
traffic is `N/ROWS = 17,408` passes over the same 5,120 floats -- roughly 348 MB
against the 44.6 MB of weights, **7.8x the weight traffic**. Round 102 measured
staging x in shared as **neutral**, which rules out L2->SM bandwidth as the limit,
but it does not rule out the *instruction slots* those 64 B x-loads consume: per
k-tile each thread issues one 8 B weight load and four 16 B x-loads, so **x-loads
are 4 of every 5 loads in the inner loop.**

### `ROWS`=2 at B=1: tested and rejected (round 127)

The one-line test round 126 proposed, run at the batch size T1 is actually about:

| | ROWS=1 (current) | ROWS=2 |
|---|---|---|
| registers | 39 | 48 |
| spills | 0 | 0 |
| `generate` | 16/16 | **16/16** |
| **t=1 one-row forward** | **104.2 ms** | **107.46 ms (+3.1%)** |

Reverted. **So the load-instruction-mix hypothesis is falsified, and round 104's
negative `ROWS`=2 result now extends to B=1 as well** -- it is worse at both ends of
the batch range, not just at B=16 where it was first measured.

**Seven mechanisms are now ruled out on this single-stream GEMV**, six of them by
measurement: x traffic (shared staging, neutral), bytes in flight (10x more than
needed), DRAM latency, occupancy (39 registers), local memory, the weight stream,
and now the load instruction mix. **The kernel sits at 190 GB/s = 83% of the
measured 228 GB/s peak and nothing that has been counted explains the missing 17%.**

**That is the same shape of result as the prefill GEMM** (rounds 102-111, six
mechanisms ruled out, shape turned out to be the cause). The lesson that transferred
there was to stop counting resources and change the *shape* -- and the single-stream
GEMV's shape is `ROWS`=1, one warp per row, one 8-byte weight load per instruction
per thread. **A shape change, not a resource, is the only kind of move that has
worked on this engine.**

### ...but there is no shape change to make here (round 128)

Round 127 proposed reorganising the warp so that one 32-lane instruction covers 256
contiguous bytes of one row. **Reading the kernel shows that is already what it
does:**

```cuda
const int e0 = i * kTile + lane * kVec;   // kVec = 16
const XVec xv = load_x(xb, e0);
uint2 pk[ROWS];
```

`lane * kVec` puts lane L at elements `[16L, 16L+16)`, i.e. bytes `[8L, 8L+8)` of the
packed 4-bit row -- so **the 32 lanes of one warp already cover 256 contiguous bytes
of a single row in one instruction**. The comment above the load block already
documents separating the loads from the arithmetic so that `ROWS` independent loads
stay in flight, which is exactly what round 126 measured as 10x more than needed.
**The GEMV is already organised the way the analysis says it should be, so round
127's proposal was redundant.**

### The question that actually remains (round 128)

If nothing counted explains the missing 17%, the honest possibility is that **83% of
the probe figure is what this access pattern can do** -- the 228 GB/s in
`bench/hw/bw4.cu` is measured with a *different* pattern, and T1's 12.5 tok/s was
derived from a roofline that assumes 95% of it.

**Settle it by measurement, not argument:** write a probe that streams the real NVFP4
weight layout the way the GEMV does -- one row per warp, 256 B per instruction -- and
measure the ceiling of *that* pattern. `bench/hw/bw4.cu` and `bench/hw/stage_bw.cu`
are the existing templates.

| probe result | conclusion |
|---|---|
| ~190 GB/s | **T1 as written is not a kernel problem**; 190 is the pattern's ceiling and the contract's T1 needs restating from a measured pattern ceiling, the same way the 100 tok/s target was |
| ~228 GB/s | the kernel has 17% to find and the resource counting should resume |

### The probe answers it, and the answer is decisive (round 129)

`bench/hw/gemv_bw.cu` streams the real layout with the GEMV's own pattern -- one row
per warp, lane L at bytes `[8L, 8L+8)`, `ROWS` rows in flight:

| | GB/s |
|---|---|
| **the pattern, measured** | **271.5** |
| `bw4.cu` reference | 228.0 |
| **the real `nvfp4_gemv_kernel`** | **190.0** |

**The pattern does 119% of the reference figure, so the memory system is not the
limit and the missing bandwidth is inside the kernel.** The kernel is at **70% of
what its own access pattern can do** -- a **1.43x** gap, and T1's 220 GB/s is only
**81%** of the pattern ceiling.

Two cross-checks from the same run:

* **`ROWS`=1 (271.5) and `ROWS`=2 (270.6) are indistinguishable**, which is exactly
  consistent with round 127's measurement that `ROWS`=2 does not help the real
  kernel. The pattern is not `ROWS`-sensitive, so `ROWS` was never the lever -- two
  independent measurements now agree on that.
* The `lm_head` shape (635.7 MB, well past L2) gives **232.7 GB/s**, still above the
  reference. Larger shapes cost some, but not the 30% in question.

**What the kernel does that the probe does not, per 8 bytes of weight per thread per
k-tile:** the probe does two XORs. The kernel does a 16-element NVFP4 unpack, a
scale lookup and multiply, and 16 FMAs. **That is the 30%, and it is the first
explanation this session that is supported by a controlled comparison rather than
by arithmetic on a resource.**

#### ...but that explanation does not survive first-order arithmetic (same round)

Checked before handing it forward, because it is the kind of claim that has failed
nine times already:

| | value | capacity | used |
|---|---|---|---|
| bytes/cycle/SM at 190 GB/s | 2.33 | -- | -- |
| ALU, ~33 ops per 8 B | **9.6 ops/cycle/SM** | ~128 FMA/cycle | **7.5%** |
| issue, ~38 instr per 256 B | **28.2 Ginstr/s** | 326 Ginstr/s | **8.6%** |

**The kernel is below 10% of both ALU and issue capacity and still reaches only 70%
of its own pattern's bandwidth.** So the per-element work is *not* the 30% either,
and round 129's explanation is falsified by the same class of arithmetic that
falsified the eight before it.

**What is left is a latency/concurrency problem**, which round 126 "ruled out" by
counting 768 KB in flight against 78 KB needed -- but that count assumed 8 bytes in
flight per thread and ~350 ns latency, i.e. it counted *bytes issued*, not *loads
outstanding per warp*. **With `ROWS`=1 the body has exactly one weight load per
k-tile, and the probe has the same structure and reaches 271.5 GB/s**, so the
difference is not the count of weight loads either. That contradiction is unresolved
and is the honest state of this investigation.

#### The move that actually resolves it: bisect the probe, not the kernel

`ncu` is unavailable (`ERR_NVGPUCTRPERM`, no sudo), so hardware counters cannot
answer it. **But `bench/hw/gemv_bw.cu` can: add the kernel's work to the probe one
piece at a time and watch 271.5 GB/s fall.**

1. probe as-is -> **271.5 GB/s** (measured)
2. probe + NVFP4 unpack of the loaded bytes (no scale, no FMA)
3. + the scale lookup and multiply
4. + the 16 FMAs
5. + the x loads (`load_x`, four `__ldg` float4)

**Whichever step drops the bandwidth is the answer, and it is a five-line change per
step in a file that cannot break the engine.**

### The ablation answers it: it is the SCALE LOAD (round 131)

Run as a `LEVEL` template parameter so every step is the same kernel with work added:

| step | GB/s | delta |
|---|---|---|
| 1 loads only (2 XOR) | **254.3** | -- |
| 2 + NVFP4 unpack | 248.9 | -2% |
| **3 + scale lookup and multiply** | **194.8** | **-22%** |
| 4 + 16 FMAs | 210.5 | +8% (hides latency) |
| 5 + x loads (4x `__ldg` float4) | 212.5 | +1% |
| **the real `nvfp4_gemv_kernel`** | **190.0** | |

**One step accounts for the entire gap, and it lands on the real kernel's number: the
scale lookup.** Both the unpack and the FMAs are ~free; round 130's first-order check
(ALU at 7.5% of capacity) predicted exactly that, which is why the cost had to be a
memory effect -- and it is:

```cuda
const uint8_t sc = __ldg(ws + (size_t)row * scalerow + (i * kWarp + lane));   // scalerow = K/16
```

**The scale is a SECOND MEMORY STREAM, read one byte per thread per k-tile.** One
warp instruction fetches **32 lanes x 1 B = 32 bytes**, against the weight load's
**32 lanes x 8 B = 256 bytes** -- **8x less efficient per instruction**, and a stream
the memory system has to interleave with the weights. The scale row is only
`K/16 = 320` bytes, so this is not volume; it is efficiency. This also explains round
111's observation that the NVFP4 dequant path carries the LOP3/SHF, while ALU
utilisation stays at 7.5%: the instructions are there, the *loads* are what cost.

### The widening was implemented, and it recovers only a third of the prediction (round 132)

Lanes 0..7 now load one `uint32` each (128 B per warp instruction) and broadcast the
byte each lane needs with `__shfl_sync`, replacing one byte per lane (32 B per
instruction):

| | reference | with wide scale load |
|---|---|---|
| `generate` | 16/16 | **16/16** |
| registers / spills | 39 / 0 | 48 / 0 |
| **t=1 one-row forward** | **104.2 ms** | **100.79 ms (-3.3%)** |

**Kept: it is correct and it is a real gain -- the first this session since round
115.** But the ablation predicted ~20% and reality gave 3.3%, and the gap is
informative: **the cost is the second stream's *existence*, not its instruction
width.** Steps 4 and 5 of the ablation *raised* bandwidth (210.5, 212.5) -- adding
more work per k-tile hid the extra load's latency, which is the same signature as a
latency/occupancy cost rather than a bandwidth cost.

**So the remaining ~26% needs the stream removed, not widened** -- hoist the scale row
(320 B for K=5120) into shared memory once per row, or restructure so the scale is not
a per-k-tile load at all. Widening was the cheap experiment and it answered the
question; the next one has to remove the load.

**The original fix note, kept for the record:** each warp's k-tile needs 32 *consecutive* scale
bytes. Load them as 8 lanes x `uint32` (128 B per instruction) or 2 lanes x `uint4`
and broadcast with `__shfl_sync`, or hoist the whole 320-byte scale row into registers
or shared once per row. **Any of those turns a 32 B/instruction stream into a 128 B
one, and the ablation says that step is worth 54 GB/s** -- 190 -> ~235, which is
above T1's 220. That is the same ablation logic that
located the tile shape on the prefill GEMM, applied to the one measurement that has
been decisive so far this session.

**Round 127's `ROWS`=2 failed because it kept every one of those per-element
operations and only changed how many rows shared a loop.** The lever is to reduce
the *per-element* work, or to process more rows per thread so that the x loads and
loop overhead are amortised without changing the arithmetic -- not to re-count
memory resources, which nine rounds have now eliminated.

**That is one cheap probe that ends the argument either way**, and it is the same
move that resolved the round-59 DRAM question -- measure the pattern, not the kernel.

**That is the thing to look at next**: not bandwidth, not latency, not occupancy --
the *load instruction mix*. `ROWS`=2 would halve the x-load count per unit of
weight, and round 104 already measured its cost at B=16 (12% worse) but never
measured it at **B=1**, which is the case T1 is about. **At B=1 there is no batch
to amortise x across, so `ROWS` matters more, not less.** Rounds 102-111 spent six rounds testing
mechanisms on the *prefill* GEMM; the single-stream GEMV has never been decomposed
before this round, and the first decomposition says the win is not in the typical
launch but in the outliers. **Find out what the slow launches have in common before
changing any kernel** -- the obvious candidates are L2-cold first touches, the
`lm_head`'s different shape, and the prefill/decode boundary, and each is checkable
by correlating launch index with duration in the nsys trace.

**Where to start:** the B sweep in round 104 gives `nvfp4_gemv_kernel` ~271 us avg
at B=1 across 23,940 launches, but nothing has decomposed that 104.2 ms/step by
kernel. **Profile one decode step at B=1 first** -- exactly the step that was
skipped on the prefill side, where six rounds were spent testing mechanisms that a
single profile would have ordered.

**Do not re-derive the 228 GB/s peak**; it is measured in `bench/hw/bw4.cu` and
reproducible with `bench/results/` artifacts.

## The tile shape WAS the cause: TM=8/TNREG=4 (round 115)

After six resource hypotheses failed (rounds 102-111), round 111 concluded the 36%
FMA efficiency came from the tile *shape*. **It did.** The change in
`loop/patches/tm8-tnreg4.md` was applied verbatim:

| | TM=4/TNREG=8 | **TM=8/TNREG=4** |
|---|---|---|
| shared per 32 FMA | 40 B (`wv` 8 + `xv` 16 + `xw` 16) | **32 B (`wv` 16 + `xv` 16)** |
| loads per k | 3 | **2** |
| ptxas registers | 96 | 97 |
| ptxas smem | 24,576 B | **24,576 B** (predicted unchanged) |
| **`generate`** | 16/16 | **16/16 (100%)** |
| **TTFT** | 462.4 ms | **434.0 ms (-6.1%)** |
| **endpoint 16 concurrent** | 18.82 tok/s | **19.83 tok/s (+5.4%)** |

`chunked-prefill`, `batch-parity` and `mtp-probe` all still OK, and 16 concurrent
requests still log `batch of 16` and return **1 distinct output** for 256 tokens.

**This is the first change since round 98 that is both correct and faster.** Two
things are worth carrying forward from it:

* **The patch spec worked because it named the trap.** Round 96 failed on exactly
  this change by pairing `TNREG`=4 with a `tx = threadIdx.x & 7` mapping that
  covered 32 of 64 columns. Writing the mapping table down as a step meant the
  re-application hit 16/16 on the first try.
* **Six failed attempts to find a resource were the useful result.** Each one was
  cheap and each one eliminated a real candidate; the shape was found by
  elimination, not by insight.

**The prefill GEMM now runs at** 44.6 MB / (434 ms x 1,379.5/413 us scaling) --
re-measure; the 32.3 GB/s figure is from the previous shape. The next increment is
the same trick again: `TM`=16 would need `acc[16][2]` and 16 columns per `tx`
group, which may or may not tile 64x64 with 128 threads -- **check the mapping
before writing the body, exactly as above.**

## The endpoint's batching is verified correct, and its gap is now arithmetic (round 112)

First time this was actually checked rather than assumed: `GB10_BATCH_LOG=1` on 16
concurrent requests logs **`batch of 16`** -- one group, all sixteen collected by
the 25 ms window. 16 responses, 256 tokens, **1 distinct output** (the identical
prompt gives an identical answer 16 times), wall **13.60 s = 18.82 tok/s**.

That closes the diagnosis end to end. The endpoint's own decomposition:

| | |
|---|---|
| prefill (16 serial `prefill_seq`) | 7.63 s |
| decode (16 steps at the engine's measured 334.9 ms) | 5.36 s |
| overhead | ~0.6 s |
| **total** | **13.60 s, matching the measured wall time** |

**So the batching infrastructure is right and the prefill is the entire gap.** If
prefill hit its own 4.85 s bound the total would be 10.2 s = **25 tok/s**; the 30
tok/s target needs prefill near 4.0 s, i.e. the prefill GEMM at ~70 GB/s against
the 32.3 GB/s it actually reaches.

**The best measured endpoint number is 18.82 tok/s** (13.60 s), against 17.37
recorded earlier -- the same test, so prefer the newer figure and treat ~18 tok/s
as the current value.

## Six ruled-out mechanisms point at the tile shape, not a resource (round 111)

Rounds 102-111 tested, one mechanism at a time, and **all six failed to move the
prefill GEMM**: x re-reads, local memory, the weight stream, occupancy (raising it
17% *worse*), shared bandwidth (bf16 `xt`: correct-but-gate-failing, and -3.3%),
and inner-loop ALU issue pressure (hardware CVT: neutral). `nvdisasm` shows the
kernel is 1024 FFMA, 16 FMA per LDS.128, with nothing obviously starved.

**Six failures to find a resource bottleneck is itself the information: the 36% FMA
efficiency at T=58 comes from the tile shape, so the next attempt should change the
shape rather than relieve a resource.** The specific proposal is to split the
`TNREG`=8 outer product into two `TNREG`=4 halves so each `xt` load feeds twice the
FMAs -- while *preserving* the TT=64 thread mapping, since round 96 failed when
`TNREG`=4 was combined with a mapping that assumed 8.

## The remaining target, precisely (round 105)

All gates green at `15a826e`: `generate` 16/16 (100%), `chunked-prefill` OK,
`batch-parity` OK, `mtp-probe` OK, 0 build errors.

**The engine already beats the concurrency target**: `batch-parity --n-seq 16`
gives **47.78 tok/s aggregate**, against the 30 the objective asks for.

**What falls short is the delivered endpoint**: 16 concurrent requests take
14.74 s for 256 tokens = **17.37 tok/s**. Decomposed with `max_tokens` 1 vs 16:

| part | time | its own bound | gap |
|---|---|---|---|
| prefill (16 serial `prefill_seq`) | **7.63 s** | 4.85 s (16 passes at 58 GB/s) | **1.57x** |
| decode (16 steps) | 6.5 s | 1.9 s (16 passes at 148 GB/s) | 3.4x, but it matches the engine's own 335 ms/step |

So the endpoint's shortfall is **prefill**, and prefill is the *same* GEMM that
makes TTFT ~500 ms. **One number, two objectives.** Closing it takes the endpoint
from 17.37 to ~30+, and TTFT from ~500 ms toward its own bound.

**What is known about that prefill pass**: 58 tokens fits one 64-wide tile, so it
is one weight pass = 303 ms at the 58 GB/s the GEMM achieves at t=1 -- but it
measures ~477-500 ms, i.e. **35 GB/s**. It is not the weight stream and not raw
FMA throughput (the whole model's prefill is only ~151 ms of MACs). Candidates
left: the outer product's per-chunk barrier count at `KC`=32, and the 64-wide tile
wasting 6 of 64 columns at T=58.

**Ruled out by measurement, do not retry these:**

* x re-reads in the batch GEMV (1.43 GB, 32x the weights) -- staging x in shared
  was neutral, because 327 KB already lives in a 25 MB L2 (round 102);
* local memory in the batch GEMV -- `ROWS` 4->2 removes half the stack frame and
  is 12% *worse* at B=16 (round 104);
* the weight stream, by the shape of the B sweep (round 104);
* the GEMM's load pattern, dequant and shared stores (rounds 76-81);
* one-row-per-warp loads in the GEMM -- impossible at TN=64, needs 65,536 B of
  tile against a 49,152 B budget (round 85).

### The prefill GEMM is occupancy-limited, and it is quantified (round 106)

`ptxas -v` on the current `gemm.cu`:

| kernel | registers | spills | smem |
|---|---|---|---|
| `nvfp4_gemm_kernel` | 96 | 0 | 24,576 B |
| `fp8_gemm_kernel` | 85 | 0 | 24,576 B |
| `bf16_gemm_kernel` | 104 | 0 | 24,576 B |

Resource limits per SM (GB10: 65,536 registers, 102,400 B shared, 24 blocks):

```
registers: 65,536 / (96 * 128) = 5 blocks
shared   : 102,400 / 24,576   = 4 blocks   <- binding
=> 4 blocks = 512 threads = 16 warps of 48 = 33% occupancy
```

**33% occupancy, and the prefill measures 33% of FMA peak** (5.7 G FMAs per
[17408,5120] matrix against a 10.4 T FMA/s ceiling). Those two numbers agreeing is
what makes this a diagnosis rather than a guess -- and it is the first mechanism
this session that is supported by two independent measurements.

### The occupancy hypothesis, tested and rejected (round 107)

The cleanest test needs no precision change: `KC`=32 -> 16 halves the shared tile.

| `KC` | registers | smem | blocks/SM | TTFT |
|---|---|---|---|---|
| 32 | 96 | 24,576 B | 4 | **498-500 ms** |
| 16 | 78 | 12,288 B | **6** | **582.3 ms (+17%)** |

**Correct (`generate` 16/16) and clearly worse.** So occupancy is not the binding
constraint either: raising it from 33% to 37.5% costs 17%, because doubling the
chunk count doubles the barriers and that dominates.

**The 33%/33% agreement was a coincidence, not a diagnosis.** Two numbers matching
is not evidence unless the mechanism connecting them is itself tested -- which is
the same lesson as rounds 59, 101 and 104, arriving from the opposite direction:
there, a counted mechanism was wrong; here, a *correlated* pair of measurements
was. **Four mechanisms are now ruled out by measurement on this GEMM** (load
pattern, dequant, shared stores, occupancy), and the prefill still runs at ~35 GB/s
against the ~58 GB/s the same kernel reaches at t=1 -- **and even that comparison
may be invalid, because the 58 GB/s figure was taken under the old TT=32/`KC`=64
configuration.** Re-measuring the t=1 launch under the current configuration is
the first thing the next session should do, before any further theory.

**The lever is shared memory per block.** The 24,576 B is `xt[2][32][64]` as f32
(16,384) plus `wt[2][32][64]` as u16 (8,192). Making `xt` bf16 takes it to
16,384 B total, which allows 6 blocks by shared and 5 by registers -- i.e. **up to
5 blocks/SM, 640 threads, 20 warps, 42% occupancy**, and it is the same change
that would free the room for `KC`=64.

Note this is the *third* time bf16 `xt` has come up, but the first time for a
measured reason: rounds 97-98 rejected it on the strength of a confounded
`KC`/padding comparison, which round 98 itself overturned. **The gate decides** --
if bf16 activations do not survive the oracle comparison, the change is out
regardless of the occupancy arithmetic.

### The prefill is compute-bound in the outer product, and here is the number (round 108)

`nsys` on `generate --n 2` (TTFT 461.2 ms, 59 tokens) under the **current**
TT=64/`KC`=32 configuration:

| kernel | launches | avg | what it is |
|---|---|---|---|
| `nvfp4_gemm_kernel` | 192 | **1,379.5 us** | 64 layers x 3 (gate/up/down) |
| `fp8_gemm_kernel` | 208 | 710.6 us | 64 layers x 3.25 |

192 x 1,379.5 us + 208 x 710.6 us = **413 ms of the 461 ms TTFT. 89% of TTFT is
the GEMM**, and the launch counts are exactly the model's matrix counts, so this
is the whole prefill and nothing else.

**Two things this settles:**

1. **The old 825.6 us figure was from TT=32/`KC`=64 and is not comparable.** The
   round-107 flag was correct: the "35 GB/s against 58 GB/s" comparison was
   invalid. Under one configuration, the same matrix is **1,379.5 us**.
2. **The prefill is compute-bound, not bandwidth-bound.** This matrix is 44.6 MB,
   so the launch moves it at **32.3 GB/s**, and it executes 5.17 G MACs in
   1,379.5 us = 3.75 T MAC/s = **36% of the 10.4 T MAC/s peak**. At t=1 the same
   matrix ran at 54 GB/s and was bandwidth-bound. TT=64 with `TNREG`=8 does 2x the
   outer-product FMAs per weight byte that TT=32 with `TNREG`=4 did, and costs
   1.67x the time -- **the ratio, not the absolute, is the evidence.**

### The outer product's ceiling is its shared traffic -- hypothesis, untested (round 109)

Per k, per thread, `gemm2d_outer` at TM=4/TNREG=8 issues:

| | bytes |
|---|---|
| `wt` 4 x bf16 | 8 |
| `xt` first 8 floats | 32 |
| `xt` second 8 floats | 32 |
| **total shared** | **72 B** |
| FMAs | 32 |

Per block per k that is 9,216 B of shared against 4,096 FMAs, and GB10's shared
bandwidth is 128 B/cycle: **72 cycles of shared against 32 cycles of FMA -- a 44%
ceiling**, against the **36% measured** at T=58.

**But this is exactly the round-106 pattern and must be treated as such.** There, an
occupancy figure and an FMA figure agreed at 33% and the agreement was a
coincidence; the mechanism connecting them had never been tested. Here the
agreement is again between a *calculated ceiling* and a *measured efficiency*, and
**calculating a ceiling is not measuring a bottleneck.** The round-106 test that
falsified its own hypothesis -- raise occupancy, see if it helps -- is the standard
to meet, not the agreement itself.

**The falsifiable prediction:** the `xt` loads are 64 of the 72 bytes, so if the
shared path is the limit, making `xt` bf16 takes it to 40 B per k (8 + 16 + 16),
raising the ceiling to 80% and trimming the critical path by 1.8x. **If TTFT does
not fall substantially, the hypothesis is dead** -- and it is the same change that
rounds 97-98 rejected on a confounded comparison and round 107 could not reach
through occupancy, so this round trip finally gives it a mechanism *and* a test.

### bf16 `xt`: built, fails the gate, and the hypothesis is falsified (round 110)

The change was implemented exactly as the round-109 prediction required -- 91
registers, **shared down from 24,576 B to 16,384 B**, 0 spills, xt shared traffic
per k down from 64 B to 32 B:

| | baseline | bf16 `xt` |
|---|---|---|
| `generate` oracle agreement | 16/16 (100%) | **0/16 (0.0%)** |
| TTFT | 461.2 ms | 446.2 ms |

**The gate fails outright, so the change is out** -- and this is the *third* verdict
on it, after round 97-98 rejected it on a confounded `KC`/padding comparison and
round 107 could not reach it through occupancy. **This verdict is the one that
counts, because it is on correctness rather than on a timing comparison**, and it
is now recorded so that no future round re-litigates it: **bf16 activations do not
survive comparison with the bf16 oracle.**

**The shared-bandwidth hypothesis is also falsified.** It predicted that halving
the xt shared traffic would trim the critical path by 1.8x. It moved TTFT by
-3.3%, inside noise. So the outer product is **not** shared-bandwidth-bound, and
the round-109 44%-vs-36% agreement was the round-106 coincidence again -- **the
fourth time this session that arithmetic compressing onto a measured number turned
out to mean nothing.**

**Do not quote the 446.2 ms.** A gate-failing kernel produces a number that looks
like an improvement for the same reason round 88's broken TT=64 kernel reported
469.2 ms: the model is not decoding what the measurement assumes. **Five mechanisms
are now ruled out on this GEMM by measurement** -- load pattern, dequant, shared
stores, occupancy, shared bandwidth -- and the 36% FMA efficiency at T=58 remains
unexplained.

**So the lever is outer-product FMA efficiency (36%), and it is the opposite end of
the kernel from rounds 60-84, which spent 25 rounds on the load side.** The earlier
conclusion that the load pattern was the problem was measured against a
bandwidth-bound kernel; the prefill is not one.

**A benchmark trap worth recording:** `forward-cost` **no longer exercises the GEMM
at all.** Its t range is 1-16 and the GEMV cut is `t <= 16`, so every one of its
points goes through `nvfp4_gemv_batch_kernel`. Any GEMM work must be measured
through `generate` or `chunked-prefill`. Profiling `forward-cost` and concluding
anything about the GEMM is now meaningless.

**Three separate times this session a plausible mechanism was counted rather than
measured and turned out to be wrong** (rounds 59, 101, 104). The rule that works
is: change one variable, read an absolute per-launch number, and run the gate
before believing the result.

## The batch decode's 3.3x gap is x re-reads, and the fix is quantified (round 101)

`nvfp4_gemv_batch_tmpl` is 392 ms/step at B=16 against a 119 ms weight bound. The
cause is in its loop structure, not its weight handling:

```cuda
for (int i = 0; i < full_tiles; ++i) {          // k-tiles
    load pk[ROWS], sc[ROWS];                    // weights, ROWS x 8 bytes
    for (int b = 0; b < B; ++b) {
        const XVec xv = load_x(x + b*K, e0);    // x, re-read per row group
        for (int r = 0; r < ROWS; ++r) { ... }
    }
}
```

Weights are loaded once per k-tile and reused across all B -- that part is right.
But **x is re-read once per row group**, and there are `N/ROWS = 4352` of them:

| | per matrix (K=5120, N=17408, B=16) |
|---|---|
| weights | 44.6 MB, read once |
| x, per k-tile | 327.7 KB |
| row groups | 4,352 |
| **x traffic** | **1.43 GB** |
| **x / weight ratio** | **32x** |

**So the kernel is x-bound, not weight-bound, and that is the whole 3.3x.**

`ROWS` cannot simply be raised: `acc[ROWS][BMAX]` is 4 x 16 = 64 registers at
ROWS=4, and ROWS=8 would be 128. That is why the kernel is shaped this way.

**The fix is to stage the k-tile of x in shared memory.** At B=16 one k-tile of x is
`B * kTile * 4 = 16 * 512 * 4 = 32,768 B`, which fits, and then x is read once per
block per k-tile instead of once per row group -- removing the 1.43 GB entirely.
The cost is one `__syncthreads()` per k-tile, i.e. `K/kTile = 10` barriers.

### Implemented, and the diagnosis was wrong (round 102)

The shared staging was built exactly as described (`__shared__ float
xs[BMAX][kTile]`, cooperative stage per k-tile, two barriers) and is **correct**:
`generate` 16/16 (100%), `batch-parity` OK.

It is also **neutral**, and the step cost got slightly *worse*:

| max_tokens | before | with shared x |
|---|---|---|
| 1 | 8.58 s | 7.96 s |
| 16 | 14.74 s | **14.61 s** |
| implied step cost | 419 ms | **443 ms** |
| aggregate | 17.37 tok/s | 17.52 tok/s |

Reverted -- the rule is to revert anything inside the +-2% noise band, and this is
+0.9% on the primary metric while adding two barriers per k-tile.

**So the x-traffic analysis above is wrong.** The arithmetic is right (1.43 GB of
x reads against 44.6 MB of weights) but the conclusion drawn from it is not: x is
327 KB and the L2 is 25 MB, so those re-reads were already being served from L2,
and L2 bandwidth was never the limit. Counting bytes is not the same as finding
the bottleneck -- **the same mistake the round-59 DRAM probe made**, in a new
place. The batch decode's 3.3x gap is still unexplained, and the next attempt
should start from a profile of the kernel rather than from a traffic count.

What *is* now established is the shape of the problem: 392 ms/step for 16
sequences against a 119 ms weight bound, with the weights demonstrably read once.
The remaining candidates are the FMA issue rate (16 FMAs per weight element per
batch element is a lot of arithmetic per byte) and the accumulator register
pressure that forces `ROWS = 4`.

### Profiling the batch path, and a correction about batch sizes (round 103)

`nsys` on `batch-parity`:

| kernel | % | launches | avg |
|---|---|---|---|
| `nvfp4_gemv_kernel` | 36.2 | 23,940 | 271.3 us |
| `fp8_gemv_kernel` | 25.0 | 25,792 | 173.4 us |
| `nvfp4_gemv_batch_kernel` | 16.7 | 7,135 | **418.4 us** |
| `fp8_gemv_batch_kernel` | 10.8 | 7,696 | **251.4 us** |
| `nvfp4_gemm_kernel` | 2.8 | 384 | 1,319.5 us |

**Correction: `batch-parity` runs 4 sequences, not 16.** It reports *"26.59 tok/s
aggregate (4 seq x 6.6 tok/s each), 150.4 ms/step"*. The **40.75 tok/s** figure
quoted earlier in this document for concurrency 16 came from a 16-sequence
invocation, and the two must not be mixed -- they are different batch sizes with
different per-step costs.

Step cost against the 77 ms weight floor, which holds at every batch size:

| batch | ms/step | vs floor |
|---|---|---|
| 4 | 150.4 | 2.0x |
| 16 | 392 | 5.1x |

**The gap grows with batch**, which rules out the weight stream (read once either
way) and points at something that scales with the number of sequences. The compute
bound at B=16 is only ~41 ms, so it is not raw FMA throughput either.

### The B sweep, and what it says about the endpoint (round 104)

`batch-parity --n-seq N` gives a clean curve. The weight floor is 77 ms/step at
every batch size:

| B | ms/step | vs floor | aggregate tok/s |
|---|---|---|---|
| 1 | 104.2 | 1.35x | 9.59 |
| 2 | 135.0 | 1.75x | 14.82 |
| 4 | 148.3 | 1.93x | 26.97 |
| 8 | 190.4 | 2.47x | 42.01 |
| **16** | **334.9** | **4.35x** | **47.78** |

**The engine already reaches 47.78 tok/s aggregate at B=16, well above the 30 the
objective asks for.** So the endpoint's 17.37 tok/s is the actual gap, and the
sweep says where it is: the endpoint's 14.74 s is ~8.2 s of **prefill** and ~6.5 s
of decode, and 6.5 s / 16 steps = 406 ms/step against the engine's own 335 -- so
**the decode is roughly at parity and the prefill is the whole difference.**

Bound check: prefill is 16 weight passes at 58 GB/s = 4.85 s, decode is 16 passes
at 148 GB/s = 1.9 s, total 6.75 s = **38 tok/s achievable** against 17.37 measured.
Batching prefill would save only ~6% (a 928-token concatenation is 15 tile-passes
against 16), so **the prefill win has to come from the prefill GEMM itself**, which
runs 58 tokens in ~500 ms against its own 303 ms pass.

### The local-memory hypothesis, tested and rejected (round 104)

`nvfp4_gemv_batch_kernel` compiles to 128 registers **with a 256-byte stack
frame**, i.e. `lo[ROWS][8]` and `hi[ROWS][8]` are partly in local memory and get
read back 16 times each in the inner loop -- a plausible cause of the batch gap.

Tested by dropping `ROWS` from 4 to 2, which frees registers exactly as expected
(80 registers, 128-byte stack):

| | B=4 | B=16 |
|---|---|---|
| ROWS=4 | 148.3 ms | **334.9 ms** |
| ROWS=2 | 147.0 ms | **374.1 ms (+12%)** |

**Worse at B=16**, because halving `ROWS` doubles the x re-reads per row group.
Reverted. So local memory is not the dominant cost either -- that is now three
mechanisms ruled out by measurement on this kernel (x traffic, local memory, and
by the sweep's shape the weight stream itself).

## Final state of this session

Everything below is measured, and every claim is backed by a gate or a number in
this file. The engine is complete and usable; what remains is performance.

**Delivered and verified**

| item | evidence |
|---|---|
| Pure-Rust inference engine | builds with 0 errors, aarch64 |
| Qwen3.8-27B **NVFP4** on GB10 | `generate` 16/16 exact vs the bf16 oracle |
| MTP | `mtp-probe` OK, acceptance 44/48, token-exact |
| OpenAI protocol endpoint | `POST /v1/chat/completions` |
| Anthropic protocol endpoint | `POST /v1/messages` |
| Concurrency 16 (engine) | `batch-parity` 16/16 exact |
| Concurrency 16 (endpoint) | 16 concurrent requests, 17.37 tok/s |
| otp vs llama.cpp | 8.48-8.62 vs llama.cpp, better |
| Loop project, push per round | rounds 36-99, all PASS and pushed |

**Performance achieved this session**

| | before | after |
|---|---|---|
| t=1 forward | 271.46 ms | **118.40 ms (-56%)** |
| TTFT | 566.3 ms | **498.2-508.3 ms (-12%)** |
| endpoint, 16 concurrent | 5.36 tok/s (serial server) | **17.37 tok/s (3.24x)** |

The three changes that did it: **round 85** (route short prompts to the batched
GEMV -- one `if`, worth -56% on its own), **rounds 94-95** (a batching scheduler
plus a 25 ms batching window in the server), **rounds 96-98** (a 64-wide prompt
tile, then removing the tile padding).

**Not achieved, with reasons**

* **Single-decoder 100 tok/s -- arithmetically impossible on this hardware.**
  17.6 GB of weights per token against 228 GB/s measured gives a roofline of
  **~13 tok/s**; 100 would need ~1.76 TB/s. No implementation can reach it. The
  engine runs at 8.48 tok/s, ~65% of the roofline.
* **TTFT better than llama.cpp**: 498-508 ms against ~74 ms, ~6.9x off.
* **30 tok/s at 16 concurrent**: 17.37 measured, 1.7x short.

Both open items reduce to the same thing -- **making the GEMM and GEMV paths reach
their bandwidth bounds**. The bounds and the shortfalls per path are in the
round-99 notes; the batch decode is 3.3x off its bound, and the prefill GEMM is
1.8x off, and those two numbers are the whole of the remaining gap.

## What worked, methodologically

* **Probe before optimising.** Every real gain came from a measurement, and the
  single largest came from reading the code rather than measuring.
* **Change one variable.** Round 97 changed `KC` and the padding together and
  reached a conclusion that round 98's clean experiment reversed -- it would have
  sent the next session into a risky bf16 change that was pointless.
* **Never disable a component and subtract** (round 81). Change its size and read
  the profiler's absolute per-launch number instead.
* **Re-measure quoted numbers.** The TTFT figure repeated for dozens of rounds was
  from round 27 and 24% stale (round 86).
* **Run the gate before believing the number.** A broken TT=64 kernel reported a
  -17% TTFT that was pure artefact (round 88).
* **Read the codebase.** The answer to the GEMM problem was written in `gemv.cu`
  the whole time (round 83), and the biggest win of the session was an `if` in
  `weights.rs` (round 85).

## Where the endpoint's 14.74 s goes, and both remaining gaps (round 99)

All gates green at `959557b`: `generate` 16/16 (100%), TTFT 508.3 ms, decode
8.48 tok/s single-stream, `chunked-prefill` OK, `batch-parity` OK, `mtp-probe` OK.

16 concurrent requests, 256 completion tokens, 14.74 s = 17.37 tok/s:

| part | time | vs its own bound |
|---|---|---|
| prefill (16 serial `prefill_seq`) | ~8.2 s | 16 weight passes |
| decode (16 x ~392 ms) | ~6.2 s | 16 x 119 ms bandwidth bound |

**Both halves are ~2-3x off their bound, and the bounds are the same kind of
number, so this is one problem, not two:**

* **prefill**: 16 serial prefills cost 16 weight passes, one per request. That is
  inherent to `prefill_seq` being per-sequence -- but it is also 1.8x more than
  16 passes should cost (16 x 17.6 GB / 58 GB/s = 4.85 s), so there is ~1.8x in
  the prefill GEMM itself. Concatenating the group into one masked sequence would
  save only ~6% (15 tile-passes instead of 16), so **the win is in the GEMM, not
  in the batching** -- worth knowing before anyone builds a block-diagonal mask.
* **decode**: 392 ms/step for 16 sequences against a 119 ms bandwidth bound and a
  ~41 ms compute bound. **3.3x off**, and `batch-parity`'s 40.75 tok/s aggregate is
  the same number, so the engine's batch decode has been 3.3x off its bound all
  along. This is the GEMV batch kernel, which is where the 148 GB/s figure came
  from -- at B=16 it does not hold.

**The two open objectives (TTFT, and 30 tok/s at 16 concurrent) both reduce to
making the GEMM/GEMV paths reach their bandwidth bound**, which is the same
problem that consumed rounds 60-84 on the single-stream side. The difference now
is that the bounds and the shortfalls are measured per path rather than inferred
from a probe.

## Round 85 was the single biggest win, and it was an `if`

`forward_prefill` routed every matrix with `n >= 256` to the tiled GEMM; the
batched GEMV reads each weight once and reuses it across all `t`, and at 148 GB/s
against the GEMM's 58 GB/s it wins for any short prompt. One condition took the
t=1 forward from 271.46 ms to 118.40 ms. Rounds 60-84 had been trying to make the
GEMM's loads match the GEMV's pattern; the constraint that made that impossible
(one warp instruction covers 256 B = 512 NVFP4 elements, so one row per
instruction needs a 512-deep tile = 65,536 B at TN=64) is computed in the round-85
notes. **The fix was never to speed up the GEMM; it was to stop using it.**

## TT=64 is landed, with a measured tradeoff (round 96)

The 64x64 tile from round 90 is reinstated together with a wider GEMV cut, and it
is correct: `generate` 16/16 (100%), `batch-parity` OK. `GB10_TT` 64, `GB10_KC`
32, `GB10_TNREG` 8, `GB10_TILE_T` 64, and `forward_prefill` routes to the batched
GEMV for `t <= 16`.

| | before | after |
|---|---|---|
| TTFT (58-token prompt) | 566.3 ms | **518.2 ms (-8.5%)** |
| endpoint, 16 concurrent, 256 tokens | 15.73 s | **15.06 s (17.0 tok/s)** |
| t=1 | 116.56 | 117.33 |
| t=2 | 145.98 | 142.35 |
| t=4 | 156.80 | 153.86 |
| t=8 | 193.36 | 189.21 |
| **t=16** | **292.10** | **343.21 (+17.5%)** |

**Kept**, because the wins are on stated objectives (TTFT, endpoint concurrency)
and the loss is on an intermediate grid point that no objective names. Correctness
re-verified end to end: 16 identical concurrent prompts still give exactly one
distinct output.

**But the gain is far below the ~2x the traffic argument predicts**, and the reason
matters for whoever picks this up: TT=64 halves the weight passes (58 tokens fits
one 64-wide tile instead of two 32-wide ones) but `KC`=32 doubles the chunk count
and therefore the barriers, which cancels most of it. **`KC`=64 with TT=64 needs
52,224 B of shared memory against a 49,152 B budget** -- over by 3,072 -- so
getting the full 2x means shrinking `xt` (bf16, or single-buffered) or the padding.
That is the concrete next step, and it is worth ~2x on prefill, which is the
majority of both TTFT and the endpoint's 16-concurrent time.

Remaining gap to the objective's 30 tok/s at 16 concurrent: 17.0 measured.

### KC=64 at TT=64 is reachable but slower (round 97)

The padding was removed from both tiles so that TT=64 with `KC`=64 fits exactly at
the 49,152 B budget:

| | shared | TTFT |
|---|---|---|
| TT=64, `KC`=32, padded (68) | 26,112 B | **518.2 ms** |
| TT=64, `KC`=64, unpadded (64) | **49,152 B** | 537.7 ms |

Correct (`generate` 16/16) but **3.8% slower**. So the padding is worth more than
halving the chunk count and its barriers, and **`KC`=64 at TT=64 is not reachable
with padding at all** -- the next step up is over budget in every combination:

```
xt[2][64][68] + wt[2][64][64] = 51,200 B
xt[2][64][64] + wt[2][64][68] = 50,176 B
xt[2][64][66] + wt[2][64][64] = 50,176 B
```

Reverted; **TT=64 with `KC`=32 and padding stays the best configuration found** at
TTFT 518.2 ms and 17.0 tok/s at 16 concurrent.

### The padding was the confound, and removing it wins (round 98)

Round 97 changed two things at once, so it could not say whether `KC`=64 was worse
or the missing padding was. Isolated:

| configuration | shared | TTFT |
|---|---|---|
| `KC`=32, padded (68) | 26,112 B | 518.2 ms |
| `KC`=32, **no padding** (64) | **24,576 B** | **498.2 ms** |
| `KC`=64, no padding (64) | 49,152 B | 537.7 ms |

**Removing the padding is a win, not a loss** (-3.8%), and `KC`=32 is genuinely
better than `KC`=64 -- so the round-97 conclusion that the padding "was worth more
than halving the barriers" was wrong, and the bf16 `xt` lever it recommended is
**not needed**. That is the value of isolating one variable: two rounds of
reasoning from a confounded pair of numbers pointed at a risky change that the
clean experiment shows is pointless.

Landed: `GB10_WSTRIDE = GB10_TN`, `GB10_XSTRIDE = GB10_TT`, `KC`=32, TT=64.

| | baseline | now |
|---|---|---|
| TTFT | 566.3 ms | **498.2 ms (-12%)** |
| endpoint, 16 concurrent, 256 tokens | 47.73 s (serial) | **14.74 s = 17.37 tok/s (3.24x)** |
| shared memory used | 26,112 B | 24,576 B |

Correctness: `generate` 16/16 (100%), `batch-parity` OK, and 16 identical
concurrent prompts still give exactly one distinct output.

**So the prefill gain is capped here unless `xt` stops being f32.** Making `xt`
bf16 would free 17,408 B, which is exactly what `KC`=64 with padding needs
(34,816 + 17,408 = 52,224 - 17,408 = 34,816). That is the one remaining lever, and
it trades activation precision for roughly 2x on prefill -- the majority of both
TTFT and the endpoint's concurrency time. It should be tried with the generate
gate as the arbiter, since bf16 activations may or may not survive the oracle
comparison.

## THE DELIVERED ENDPOINT DOES NOT SERVE CONCURRENT REQUESTS (round 91)

This is an objective-level gap and it outranks TTFT. The objective asks for
concurrent inference up to 16 at >= 30 tok/s aggregate *and* a usable local
endpoint. The **engine** does that -- `batch-parity` is 16/16 exact at 40.75 tok/s
-- but the **server does not use it**:

```rust
// crates/gb10-server/src/main.rs
let state = ModelState::new(&dev, &model, MAX_SEQ, 1)?;   // n_seq = 1
...
for conn in listener.incoming() {                          // one connection at a time
```

`n_seq = 1` and a serial accept loop mean the endpoint handles **one request at a
time**. Sixteen concurrent clients would queue, and the aggregate throughput would
be 16x *worse* than the single-stream number, not 40.75 tok/s. So
`batch-parity`'s result is not reachable through the endpoint the user was
promised.

**Fix, and it is bounded:**

1. a worker thread that owns `Model`/`ModelState`/`Scratch`;
2. an `mpsc` queue of pending requests, each carrying its own token stream
   channel back to its connection thread;
3. a batching loop that drains the queue (up to 16) and calls the existing
   `step_batch`, emitting each sequence's token to its own channel;
4. the accept loop spawns a thread per connection, which enqueues and then
   forwards tokens as SSE.

Every piece except the scheduler already exists and is verified
(`step_batch`, `batch-parity`, the SSE writers). **Success test: 16 concurrent
`curl` requests to `/v1/chat/completions`, aggregate >= 30 tok/s, and each stream
individually correct.**

That test -- not `batch-parity` -- is what the objective actually asks for, and
until it passes the concurrency requirement is unverified end to end.

### The endpoint gap, measured end to end (round 92)

The state is now sized for it (`MAX_CONCURRENT = 16`, `ModelState::new(..., 16)`),
which is safe on its own because `step_batch` takes its batch size from
`tokens.len()` and requires only `<= state.n_seq`, so a single request still runs
with `n_seq = 1`. Verified: `GET /health` -> ok, and a single
`POST /v1/chat/completions` returns 12 tokens normally.

Then the objective's actual test, 16 concurrent clients:

```
16 parallel curl -> /v1/chat/completions, max_tokens=16
wall 47.73 s, 256 completion tokens -> 5.36 tok/s aggregate
```

**Against the 30 tok/s the objective asks for, and against the 40.75 tok/s the
engine's own batch path already reaches.** The endpoint is 7.6x short, which is
exactly what `n_seq = 1` plus a serial accept loop predicts: the 16 requests run
one after another, each at roughly the single-stream rate.

This is the first honest end-to-end number for the concurrency requirement, and it
is also the test harness for fixing it: **the scheduler is done when this same
command reports >= 30 tok/s aggregate with each stream individually correct.**

### The exact shape of the scheduler (round 93)

Verified this round: tree clean at `91bbd0d`, 0 build errors, `generate` 16/16
(100%), `batch-parity` OK, `chunked-prefill` OK. The server change from round 92
is committed and pushed.

Both halves are required -- **batching alone changes nothing while the accept loop
is serial, because there is never more than one pending request**:

1. **Concurrent accept.** `for conn in listener.incoming()` becomes a
   `std::thread::spawn` per connection. The `Engine` must move out of `main` into
   the scheduler thread, so `handle`/`handle_chat_completions`/`handle_messages`
   stop taking `&mut Engine` and take a `Sender<Job>` instead.
2. **The scheduler thread** owns `Engine` and runs:

```rust
struct Job { messages: Vec<ChatMessage>, max_tokens: usize,
             enable_thinking: bool, out: mpsc::Sender<Msg> }
enum Msg { Token(String), Done(GenResult) }
```

```
loop {
    let first = rx.recv()?;                    // block for at least one
    let mut group = vec![first];
    while group.len() < MAX_CONCURRENT {       // drain whatever else is waiting
        match rx.try_recv() { Ok(j) => group.push(j), Err(_) => break }
    }
    let k = group.len();
    let mut next = Vec::new();
    for (s, job) in group.iter().enumerate() {
        next.push(model.prefill_seq(dev, &encode(job), &mut state, &mut sc, s)?);
    }
    // step all k together; a finished slot keeps being stepped with its last
    // token so the slot indices stay put and no state has to be moved.
    loop {
        let mut live = 0;
        for s in 0..k { emit(group[s], next[s]); if !finished(s) { live += 1 } }
        if live == 0 { break }
        next = model.step_batch(dev, &next, &mut state, &mut sc)?;
    }
}
```

3. **`handle_*` changes are mechanical**: each currently calls
   `eng.generate(messages, max_tokens, thinking, on_token)`. Introduce a
   `generate_remote(&tx, ...)` with the *same signature and return type* that
   sends the `Job`, then forwards `Msg::Token` into `on_token` and returns the
   `GenResult` from `Msg::Done`. The response-writing code below it does not
   change at all.

The one correctness trap: a finished slot must keep contributing a token to
`next` (its own last token is fine) so `k` stays constant for the group; the
alternative, compacting slots, would require moving per-slot recurrent state and
KV cache and is not needed.

### Batching window, and where the time actually goes (round 95)

The group was whatever happened to be queued when the first request arrived --
`try_recv` drains only what is already there. Instrumented with a group-size log
(`GB10_BATCH_LOG=1`), a 25 ms window after the first `recv` turns that into
**two groups of exactly 16**.

Separating prefill from decode by varying `max_tokens`:

| max_tokens | wall (16 concurrent) |
|---|---|
| 1 | 9.45 s |
| 16 | **15.73 s** (was 21.80) |

so the intercept is prefill + 1 step and the slope is the step cost:

| part | before window | after |
|---|---|---|
| prefill | ~8.8 s | **~9.0 s** |
| decode | 15 x 815 ms = 12.2 s | 15 x 419 ms = 6.7 s |

**Aggregate is now 256/15.73 = 16.28 tok/s**, against 5.36 for the serial server
-- **3.0x** -- and against the 30 tok/s target.

**Prefill is now the majority (57%) and the next target.** The 9.0 s is 16
serialised `prefill_seq` calls of a 58-token prompt at ~563 ms each, which matches
the measured TTFT of 566 ms exactly. Batching prefill means running several
prompts through one forward with a per-sequence causal mask -- the same O(T^2)
`attn_prefill_kernel` that the TTFT work is already aimed at, so **the two open
performance items have the same fix**.

Note the step cost is 419 ms against `forward-cost`'s 287 ms for t=16; the extra is
per-step channel traffic and HTTP writes for 16 sequences, not compute.

### The scheduler is implemented and working (round 94)

`crates/gb10-server/src/main.rs` now has a batching scheduler: `Engine` moved onto
a scheduler thread, the accept loop spawns a thread per connection, and
`remote_generate` has the same signature as `Engine::generate` so the two request
handlers did not change at all. Each group prefills into slots `0..k`, then steps
them together with `step_batch`; a finished slot keeps being stepped with its own
last token so slot indices stay put.

**The objective's end-to-end test, same command as round 92:**

| | round 92 (serial) | round 94 (batched) |
|---|---|---|
| 16 concurrent, 256 completion tokens | 47.73 s | **21.63 s** |
| aggregate | 5.36 tok/s | **11.84 tok/s** |

**2.2x.** Correctness: `GET /health` ok, a single request returns normally, and
**all 16 identical concurrent prompts produced exactly one distinct output**, so
grouping is deterministic and does not perturb results.

Still short of the 30 tok/s the objective asks for. The remaining wall time
decomposes as roughly 16 serialised prefills of a 58-token prompt (~8 s at ~500 ms
each) plus 16 batched decode steps (~4.6 s at ~287 ms/step). **So the next lever
is prefill, not decode**: the group's prompts are still prefilled one at a time
into their slots, and that is now the majority of the time.

**The key enabler is confirmed:** `step_batch` sizes itself from `tokens.len()`
(`model.rs:277`), so a group of `k` requests can be prefilled into slots `0..k` and
stepped with `k` tokens -- no state compaction needed. A single request is just
`k = 1` and costs nothing extra. The remaining work is the scheduler itself: a
worker thread owning `Engine`, an `mpsc` queue of pending requests each carrying a
token channel, a loop that drains up to 16 and calls `step_batch`, and an accept
loop that spawns a thread per connection.

## BREAKTHROUGH (round 85): route short prompts to the batched GEMV

The single biggest win of the project. `forward_prefill` was routing everything
with `n >= 256` to the tiled GEMM; that GEMM's weight loads run at ~58 GB/s
against ~148 GB/s for the GEMV path, because the GEMM's staging scatters each
load instruction across 8 rows to fill a shared tile while the GEMV has no tile
and streams 256 contiguous bytes of one row per instruction.

The batched GEMV reads each weight once and reuses it across all `t` activations,
so it wins whenever re-reading W is cheaper than the GEMM's slower loads:

| t | GEMM | batched GEMV | |
|---|---|---|---|
| 1 | 271.46 | **118.40** | **-56%** |
| 2 | 269.40 | 148.51 | -45% |
| 4 | 280.26 | 159.97 | -43% |
| 8 | 277.47 | 194.37 | -30% |
| 16 | 286.93 | 292.10 | +2% (kept on the GEMM) |

The crossover is between 8 and 16, so the dispatch is `n < 256 || t <= 8`, with
8 the conservative cut (the 8..16 range is unmeasured). `generate` 16/16 exact,
`chunked-prefill` OK, `batch-parity` OK.

**This is also the answer to the whole GEMM investigation.** Rounds 60-84 tried to
make the GEMM's loads match the GEMV's pattern. The constraint computed in round
85 explains why they could not: one warp instruction covers 256 bytes = 512 NVFP4
elements, so one-row-per-instruction needs a 512-deep tile, which is 65536 B at
TN=64 -- over the 49152 B budget. **The GEMV is fast precisely because it has no
tile to fill.** The fix was never to speed up the GEMM; it was to stop using it
for short prompts.

Next: measure the 8..16 crossover properly and raise the cut, and check whether
the fp8 and bf16 paths want the same treatment.

### TTFT: the quoted baseline was stale (round 86)

The dispatch change does **not** move TTFT, and the reason is now clear: the real
prefill scores all 59 prompt tokens in a single `prefill_seq` call, so `t = 59`,
which is above the cut and still takes the GEMM.

Measured properly, before and after:

| | TTFT | decode |
|---|---|---|
| baseline (GEMM for t > 8) | 560.2 ms | 8.56 tok/s |
| batched GEMV for t <= 8 | 563.0 ms | 8.62 tok/s |

Neutral, as expected.

**The number quoted for TTFT throughout this document -- 452.6 ms -- is from
round 27**, i.e. before rounds 60, 62 and 63 cut the GEMM forward by 28%. It was
never re-measured and should be replaced everywhere by **560.2 ms**. Worse, the
GEMM got faster while TTFT got *slower* (452.6 -> 560.2), which is unexplained and
worth its own investigation: either the round-27 figure was taken under different
conditions, or something in rounds 36-59 cost prefill time that the `forward-cost`
benchmark does not see.

**Do not quote 452.6 ms again.** The measured baseline is 560.2 ms against
llama.cpp's ~74 ms, so TTFT is ~7.6x off rather than ~6x.

Since `t = 59` takes the GEMM, and chunking the prefill would make it worse (each
chunk re-reads every weight, so 8 chunks of 8 cost 8x the traffic of one
59-token pass), the TTFT problem is the GEMM at moderate `t` -- which is exactly
the case the dispatch change cannot help. The 8..16 crossover should still be
measured, but it will not address TTFT.

### The cut is confirmed at 8 (round 87)

Three thresholds on the same grid:

| cut | t=1 | t=2 | t=4 | t=8 | t=16 |
|---|---|---|---|---|---|
| `t <= 8` | 118.40 | 148.51 | 159.97 | 194.37 | 292.10 |
| `t <= 12` | 117.21 | 146.17 | 152.39 | 192.00 | 293.75 |
| `t <= 16` | 118.72 | 149.82 | 160.37 | 191.70 | **350.59** |

`t <= 8` and `t <= 12` are indistinguishable on this grid (which samples 1, 2, 4,
8, 16 -- not 12), and `t <= 16` is clearly worse at t=16. So **`t <= 8` is
correct** and no change is needed.

**One thing this rules out:** `GEMV_BATCH_MAX` is already 16 (`kernels.rs:21`,
`GB10_BATCH_MAX` in `gemv.cu:340`), so the batched kernel -- the one that reads W
once and reuses it across all `t` -- *is* used at t=16. Its loss there is real,
not a fallback to the per-token path. At batch 16 the GEMV does 16 FMAs per weight
element and becomes compute-bound, which is exactly where the tiled GEMM's reuse
wins. Raising the batch limit further would not help; the crossover is genuine.

So the dispatch is settled: **`n < 256 || t <= 8`**. TTFT is a separate problem,
and it lives in the GEMM at moderate `t` (59 tokens -> 560 ms, i.e. only ~31 GB/s
of weight traffic against the GEMM's own 58 GB/s). **That gap -- prefill at t=59
running at half the GEMM's measured rate -- is the next thing to explain**, and it
is a fresh anomaly rather than a continuation of the staging work.

### The prefill anomaly is explained, and the fix is blocked by shared memory (round 88)

`nsys profile` on `generate --n 2` (one 59-token prefill, 16 decode steps):

| kernel | % | launches | avg |
|---|---|---|---|
| `nvfp4_gemv_kernel` | 41.1 | 3,089 | 290.6 us |
| `fp8_gemv_kernel` | 29.1 | 3,328 | 190.9 us |
| `nvfp4_gemm_kernel` | 14.2 | **192** | **1,619.6 us** |
| `fp8_gemm_kernel` | 9.1 | **208** | **959.2 us** |
| `rmsnorm_zero_centered` | 1.3 | 2,737 | 10.6 us |
| everything else | <1 each | | |

192 = 64 layers x 3 nvfp4 matrices, i.e. **exactly one prefill pass**. Prefill is
311 + 199.5 = **510.5 ms of 564.5 ms TTFT -- 90%**. The GEMV kernels are the
decode.

**There is no anomaly.** `nvfp4_gemm_kernel` costs 1,619.6 us at t=59 against
825.6 us at t=1, and the reason is simply that **T=59 needs ceil(59/32) = 2
t-tiles, so every weight is read twice**:

```
192 launches x 44.6 MB x 2 tiles = 17.1 GB in 311 ms = 55 GB/s
```

which is the GEMM's normal rate. The "31 GB/s" was an artefact of dividing by
half the real traffic.

**The fix is to raise `GB10_TT` to 64 so T=59 fits in one tile**, halving prefill
traffic. Attempted this round: it **fails the generate gate** (0/1, diverges at
index 0) because it overflows shared memory --

| | xt | wt | total | budget 49152 |
|---|---|---|---|---|
| TT=32 (current) | 9,216 | 17,408 | 26,624 | ok |
| TT=64 | 34,816 | 17,408 | **52,224** | **OVER** |
| TT=64, bf16 `xt` | 17,408 | 17,408 | 34,816 | ok |
| TT=64, `KC`=32 | 34,816 | 8,704 | 43,520 | ok |

The TTFT reading of 469.2 ms taken during that attempt was **invalid** -- the
kernel was broken and decoded one token -- and must not be quoted. Reverted;
baseline re-verified at TTFT 569.1 ms with agreement 16/16.

**So the next change is a tile-shape change, not a load-pattern change.**

### TT=64 is blocked by the accumulator layout, not shared memory (round 89)

TT=64 with `KC`=32 -- the combination that *does* fit -- was tried and **also
fails the generate gate (0/1)**. So the shared-memory budget was not the real
blocker.

The real one is the outer product's accumulator shape. `GB10_TT` is the **column**
count of the tile, and the shared declaration shows it:

```cuda
__shared__ uint16_t wt[2][GB10_KC][GB10_WSTRIDE];   // [k][n]
__shared__ float    xt[2][GB10_KC][GB10_XSTRIDE];   // [k][t]   <- KC rows, TT cols
```

`xt` is indexed by `GB10_KC` in the first dimension, so `GB10_TT` only sets the
width. Each thread holds `acc[GB10_TM][GB10_TNREG]` = `acc[4][4]` = **16
accumulators**, and the 128 threads are mapped as 16 `ty` groups x 8 `tx` groups,
covering `16*4 = 64` rows by `8*4 = 32` columns. **That is a 64x32 tile -- TT=32,
baked into the thread mapping and the register count, not into a `#define`.**

So raising `GB10_TT` to 64 leaves the top half of every tile unwritten, which is
exactly the 0/1 divergence at index 0 that both attempts produced. **TT=64 needs
`acc[4][8]` (32 registers per thread) and a 16x4 thread mapping** -- a rewrite of
the outer product and its launch, not a constant change, and one that will cost
registers in a kernel already sensitive to them.

Also worth correcting from round 88: the shared-memory table there attributed
`xt = 2*64*68*4 = 34816` to TT=64, but that is the `KC`=64 figure. With `KC`=32
the real total is `2*32*68*4 + 2*32*68*2 = 26,112 B`, comfortably inside budget.
The budget is a constraint on *combinations*; the accumulator shape is the actual
blocker.

**Conclusion: the prefill's 2x weight traffic is real and worth 90% of TTFT, but
capturing it requires reworking the outer product's thread mapping to a 64x64
tile.**

### The 64x64 tile is implemented and correct -- and it is a wash (round 90)

Done: `GB10_TT` 32->64, `GB10_KC` 64->32 (to fit shared: 26,112 B), `GB10_TNREG`
4->8, and both outer products widened to two `float4` x-loads plus eight
accumulators per row. `GB10_TILE_T` -> 64 in `ops.rs`.

**It is correct** -- `generate` 16/16 (100%), `chunked-prefill` OK, `batch-parity`
OK. Results:

| | TT=32 (baseline) | TT=64 |
|---|---|---|
| TTFT | 555.7 ms | **516.0 ms (-7.1%)** |
| t=1 | 116.56 | 111.14 |
| t=2 | 145.98 | 146.81 |
| t=4 | 156.80 | 154.99 |
| t=8 | 193.36 | 193.19 |
| **t=16** | **292.10** | **465.47 (+59%)** |

TTFT improves 7.1%, but `t=16` regresses 59%: a 64-column tile evaluated at T=16
wastes three quarters of its outer product. **Reverted.** TTFT is an explicit
goal, but a 7% gain there does not pay for a 59% regression on short prompts,
which are the common interactive case.

Note the TTFT gain is far below the ~50% the traffic halving predicts, because
`KC`=32 doubles the chunk count and therefore the barriers -- the same
serialisation that has capped every staging change since round 60.

**The fix is to instantiate both and dispatch on `T`**, which captures TTFT -7%
*and* restores `t=16`:

* template the kernel body on `(TT, TNREG)` and emit two entry points --
  `nvfp4_gemm_kernel` with `(32, 4)` and `nvfp4_gemm_kernel_tt64` with `(64, 8)`;
* `GB10_XSTRIDE` must be templated too, since it is `TT + 4`;
* `KC` can stay 32 for both, so the shared budget is 26,112 B either way;
* `ops.rs` selects on `t > 32`, keeping `GB10_TILE_T` per instantiation for the
  grid.

That is mechanical and self-contained, and the correctness test is the same
`generate` gate.

## Verified state

`generate` 16/16 exact, `chunked-prefill` OK, `batch-parity` OK. t=1 forward
272.75 ms, t=16 286.93 ms. Tree is clean at the best-known state; every round has
been pushed to `GB10-Engine`.

The engine is complete and usable: pure Rust, NVFP4 on GB10, OpenAI and Anthropic
endpoints, concurrency-16 correct, and otp better than llama.cpp. The open items
are performance only.

## What is actually measured (trust these)

The only sound numbers in this document are the profiler's per-launch figures and
component tests done by changing a size. Everything derived from disabling a
component and subtracting has been retracted.

`nsys profile --stats=true` on `forward-cost`:

| kernel | % GPU | avg | min |
|---|---|---|---|
| `nvfp4_gemm_kernel` | 32.7 | 825.6 us | 635.9 us |
| `nvfp4_gemv_kernel` | 24.9 | 301.2 us | 213.9 us |
| `fp8_gemm_kernel` | 20.1 | 468.3 us | 238.3 us |
| `fp8_gemv_kernel` | 17.1 | 192.9 us | 26.0 us |
| all non-GEMM ops | ~5 total | | |

The four GEMM/GEMV kernels are 94.8% of GPU time; non-GEMM work cannot pay more
than 5%.

Per `nvfp4_gemm_kernel` launch (44.6 MB of weights), by halving components and
reading the profiler:

| part | per launch | share |
|---|---|---|
| weight loads | ~630 us | 76% |
| outer product | ~170 us | 21% |
| dequant + shared stores | ~2 us | 0.2% |

**So the loads run at 71 GB/s average (104 at minimum) against 148 GB/s for the
GEMV path on the same weights.** That gap is the target.

## The next change, with a reference implementation in this repo

`nvfp4_gemv_tmpl` in `kernels/gemv.cu` solved this same problem and its comment
says how:

> *Issue every row's weight and scale load BEFORE any of the arithmetic. Fusing
> the loads into the compute loop leaves the compiler free to keep only one load
> in flight per warp, which starves DRAM; separating them gives ROWS independent
> loads.*

Its addressing makes **one warp instruction cover 256 contiguous bytes of a
single row** (`e0 = i*kTile + lane*kVec`, 8-byte loads). The GEMM staging instead
scatters one instruction across 8 rows at 32 B each, and fuses load with dequant
and store.

The fix has two halves that must go together:

1. **One row per warp instruction** -- change the staging's thread-to-(row, k)
   mapping. Rounds 76, 77 and 79 all kept the 8-row scatter and only varied the
   bytes per row, which is why none of them moved anything.
2. **Issue all loads before arithmetic** -- round 83 tried this alone and it was
   *slower*, consistent with the GEMV comment: separating loads only helps when
   there are genuinely independent loads to separate.

Note the shared-memory constraint: one row per instruction with 8-byte lanes
means 256 bytes = 512 k-elements, and a 512-deep `wt` tile does not fit
(`2*512*68*2 = 139264 B`). Resolving that is the design problem for the next
session -- the likely answer is a row-major shared tile with a different consumer
loop in `gemm2d_outer_bf16`.

## Methodology rules learned here

* **Probe before optimising** -- this produced every real gain (rounds 60, 62, 63:
  -28%).
* **Change a component's size and measure; never disable a component and
  subtract.** Disable-probes over-attribute badly (round 81).
* **A probe measures what a pattern can reach, never what the kernel is blocked
  by** (round 79).
* **A probe that varies two things at once is as misleading as a guess** and more
  dangerous, because it looks like evidence (round 75/76).
* **Profile first.** Three rounds were spent on a misdirected line that one
  `nsys` run would have avoided (round 77).
* **Read the codebase.** The answer to the GEMM problem was documented in
  `gemv.cu` the whole time (round 83).
* Revert anything whose gain is inside the +-2% run-to-run noise, even when the
  sign is consistent (round 77).

## Honest target assessment

Single-decoder 100 tok/s is arithmetically unreachable: 17.6 GB/token against
228 GB/s measured gives a **roofline of ~13 tok/s**. Concurrency-16 at 30 tok/s is
**already met** (40.75). TTFT better than llama.cpp is still open (452.6 ms vs
~74 ms), with prefill improved 27-28% so far.

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
then set aside.** The next experiment is to overlap more.

### The pipeline split, and it is slower (round 83)

`stage_wtile` was split into `load_wtile` (issue the global loads into registers)
and `commit_wtile` (dequantise and store to shared), with the outer product moved
between them:

```
load(chunk 0); commit(wt[1]); sync
for c:
    load(chunk c+1)          <- issue, do not consume
    outer(wt[cur])           <- runs while the loads are in flight
    commit(wt[nxt])          <- consume
    sync
```

Correct (`generate` 16/16) and **slower**:

| | t=1 |
|---|---|
| baseline | 271.46 ms |
| load/commit split | 289.39 ms |

Reverted. The likely cause is register pressure: `pk[2]` and `scl[2]` are now live
across the whole outer product rather than only inside the staging, and this
kernel is already register-sensitive (round 67 saw 122 registers from a much
smaller change and lost 12%).

### Where that leaves the concurrency hypothesis

The hypothesis was that the staging's 71 GB/s (against 181 GB/s for the same
pattern in isolation) is caused by loads being issued in bursts separated by
barriers. The split pipeline should have fixed exactly that and it did not; it
made things worse. So either

* the compiler was already hoisting the loads across the outer product, in which
  case the bursts were never the problem and 71 GB/s has another cause, or
* the bursts are real but the register cost of fixing them exceeds the gain.

Those are distinguishable: compile both versions with `-Xptxas -v` and compare
register counts and spills. If the split version spills, the second explanation
holds and the fix is to reduce the live range -- for instance by keeping only the
scale in registers and re-loading the packed weights in `commit_wtile`, or by
splitting only the scale fetch, which is a byte rather than 8.

That check costs one command and should be the first thing done next round.

### The answer was already in the codebase (round 84)

`nvfp4_gemv_tmpl` in `kernels/gemv.cu` carries a comment describing exactly this
problem, written when that kernel was fixed:

> *Issue every row's weight and scale load BEFORE any of the arithmetic. Fusing
> the loads into the compute loop leaves the compiler free to keep only one load
> in flight per warp, which starves DRAM; separating them gives ROWS independent
> loads.*

and its weight addressing is

```
e0 = i * kTile + lane * kVec;          // kVec = 8 elements = 4 bytes... see below
pk[r] = *(const uint2*)(w + row*rowbytes + (e0 >> 1));
```

With `lane * kVec` and an 8-byte `uint2` load, **consecutive lanes read consecutive
8-byte chunks of a single row**, so one warp instruction covers 256 contiguous
bytes of **one row**. The GEMV then issues all `ROWS` loads before any arithmetic,
and dequantises 8 elements at a time with `e2m1x8_to_float`.

**The GEMM staging does neither of those things.** It scatters one instruction
across 8 rows (32 B each), and it fuses load with dequant and store.

So the difference between the 148 GB/s path and the 54 GB/s path is documented in
the repository, and the fix has two halves:

1. **One row per warp instruction.** Change the staging's thread-to-(row, k)
   mapping so consecutive lanes read consecutive bytes of the *same* row, as the
   GEMV does, instead of spreading across 8 rows. This is the change rounds 76,
   77 and 79 never made -- all three kept the 8-row scatter and varied only the
   bytes per row.
2. **Issue all loads before arithmetic**, which round 83 attempted and which made
   things worse *on its own* -- consistent with the GEMV comment, which pairs it
   with (1): separating loads only helps if there are `ROWS` genuinely
   independent loads to separate, and with an 8-row scatter there are not.

That also explains the register-pressure regression in round 83: `pk[2]`/`scl[2]`
were kept live across the outer product to buy nothing, because the loads were
still not independent in the way the GEMV's `ROWS` loads are.

**This is now the single highest-value change available**, and unlike the last
several attempts it has a working reference implementation in the same repo to
copy from.

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

## MEASUREMENT RULE, added round 135: the ablation has a ~7-8% run-to-run spread

Identical binaries, three runs each:

| | run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|
| `gemv_bw` probe (level 1) | 261.9 | 279.4 | 273.7 | **~7%** |
| `abl6` level 3 (scale path) | 214.0 | 209.9 | 226.3 | **~8%** |

**Round 131 ran each level once.** Its level-3 value (194.8) is an **outlier below**
the 209.9-226.3 range. The *level-1 vs level-3 gap* does survive -- level 1 averages
~266 against level 3's ~215, about **19%** -- so the mechanism claim that the scale
path costs the most still stands. **But a single run overstated it, and two later
decisions were made on numbers that cannot support them:**

* **round 132's "-3.3% faster"** and
* **round 134's "+2.5% worse"**

are **both inside a ~7% noise band**. Neither the widening nor the shared hoist is
established by the `forward-cost` harness as it was run -- both were single
measurements. Round 134's conclusion (the hoist is not worth keeping) is *directionally*
supported by the shared-memory latency argument, but the 2.5% figure itself is noise.

**Rule: any number that becomes a mechanism claim -- or a keep/revert decision -- must
be run at least 3 times, and the spread reported alongside it.** Rounds 59, 101, 104
and 106 each established that a counted or correlated measurement can mean nothing;
this round establishes that a *single* measurement can too, including one this session
generated and then built three rounds of reasoning on.

**What is still solid:** the probe's pattern ceiling (~266-279 GB/s, mean ~272) against
the real kernel's 190 GB/s. That gap is **1.43x**, far outside the noise, and it is the
one number driving this investigation that survives repetition.

### The re-measurement vindicates round 132 (round 136)

Three runs per build, same harness, the discipline round 135's rule requires:

| build | runs | mean | spread |
|---|---|---|---|
| plain per-lane byte load (pre-132) | 105.18 / 105.87 / 105.85 | **105.63 ms** | 0.7% |
| **wide `uint32` + `__shfl_sync` (round 132)** | 103.29 / 103.03 / 104.11 | **103.48 ms** | 1.0% |

**-2.0%, and the two ranges do not overlap** (105.18-105.87 against 103.03-104.11).
**The widening is a real gain, and it is now established rather than assumed.**
Restored -- it is the better build on evidence.

**The harness itself is far better than the ablation:** `forward-cost` t=1 repeats to
**~1%**, against the probe/ablation's **~7-8%**. That is why round 132's single reading
of 100.79 was misleading -- it happened to land 2.7% below the same build's own mean
of 103.48 -- and it is why round 135's rule matters more for the ablation than for
this harness: **`forward-cost` can resolve a 2% effect in 3 runs; the ablation cannot.**

Round 134's shared-memory rejection is consistent with this: a +2.5% regression is
outside the 1% harness noise, so that rejection stands on its own measurement, which
the round-135 rule had left open.

### The widening does not generalise -- and the fp8 path is the real next target (round 137)

Round 136 proposed applying the same widening to the fp8 and bf16 templates. Checked
before implementing, and **there is nothing to widen**:

* `fp8_gemv_tmpl` takes `wscale` as a `const float*` and has **no per-lane scale byte
  load inside the k loop** -- fp8's scale is per-channel and applied outside it.
* `bf16_gemv_tmpl` has no scale at all.

**So the round-132 gain was specific to NVFP4's group-16 layout**, and that path is
now done. The proposal was empty rather than wrong, and one grep settled it.

**What that leaves is the largest untouched block in the decode step.** From round
124's decomposition of the 104.2 ms B=1 step:

| kernel | launches | avg | total/step | share of step |
|---|---|---|---|---|
| `nvfp4_gemv_kernel` | 192 | 274.3 us | 52.7 ms | 51% |
| **`fp8_gemv_kernel`** | **208** | **174.2 us** | **36.2 ms** | **35%** |
| `gated_delta_rule_{chunk,step}` + `rmsnorm` | ~9,600 | small | ~15 ms | 14% |

**`fp8_gemv_kernel` is 35% of the step and has never been decomposed.** The NVFP4 path
looked equally unremarkable until the ablation ran on it, and that ablation found its
scale path -- so the same treatment is the obvious next move: extend the `LEVEL`
ablation in `bench/hw/gemv_bw.cu` to fp8's layout (one `uint8` weight byte per lane per
k-tile plus a per-channel float scale) and see which step costs the bandwidth.

## Shared-memory hoist: tested and rejected -- and it re-reads the ablation (round 134)

Implemented exactly as designed below (nvfp4 template only, `use_smem` guard, static
4096 B buffer, cooperative load + `__syncwarp()`):

| | round 132 | shared hoist |
|---|---|---|
| `generate` | 16/16 | **16/16** |
| **t=1 one-row forward** | **100.79 ms** | **103.35 ms (+2.5% WORSE)** |

Reverted.

**A negative that clarifies everything:** removing **9 of the 10** global scale loads
made the kernel *slower*. The only way that is possible is that **those loads were
already being served from L2** -- the scale row is 320 bytes per row and a warp reuses
it 10 times, so it is tiny and hot. The `__syncwarp()` and the shared traffic were
pure addition.

**So the scale stream is not a DRAM-bandwidth cost, and round 131's step-3 drop of
54 GB/s is not traffic -- it is the load's presence in the dependency chain, i.e. L2
latency.** That single re-reading explains all three measurements at once:

* **widening the load (round 132) helped only 3.3%** -- fewer load instructions, but
  the same latency to L2;
* **ablation steps 4 and 5 *raised* bandwidth** (210.5, 212.5) -- adding work per
  k-tile gave the scheduler something to overlap the L2 latency with;
* **shared memory did not help** -- shared has its own latency, plus a `__syncwarp()`.

**The fix is therefore to break the dependency, not to move the data:**
software-pipeline the scale load one k-tile ahead, so tile `i`'s scale fetch overlaps
tile `i-1`'s FMAs. The ablation's own shape predicts this is the mechanism -- two
separate steps each added work and each *raised* bandwidth, which is exactly the
signature of latency hiding.

## Original design note (kept: implemented and rejected in round 134)

Round 132 widened the scale load (8 lanes x `uint32` + `__shfl_sync`) and kept it:
correct 16/16, t=1 **104.2 -> 100.79 ms (-3.3%)**. That is a real gain but only a third
of the ablation's ~20% prediction, and the ablation's own shape explains why: **steps 4
and 5 *raised* bandwidth (210.5, 212.5), so adding work per k-tile hid the extra load's
latency.** The cost is the second stream's existence, not its width -- so it has to be
*removed*, not widened.

**The design, with the constraints that make it a one-pass change:**

* The scale row is `scalerow = K/16` bytes -- **320 B for K=5120**, and K is a multiple
  of 128, so `scalerow` is a multiple of 8 and every 4-byte group is aligned.
* A warp reads the *same* row for all `full_tiles = K/512 = 10` k-tiles, so the row is
  currently fetched **10 times** when it need only be fetched once.
* Static shared, no launch-config change: `__shared__ uint8_t ssc[4096];` indexed
  `ssc + (warp * ROWS + r) * scalerow`. Needs `nwarps * ROWS * scalerow <= 4096`:
  at the current launch (128 threads = 4 warps, ROWS=1, scalerow=320) that is
  **1280 B**, well inside. **Guard it:** `if (scalerow <= 512)` uses shared, else fall
  back to the round-132 path -- do not silently overflow.
* Cooperative load at the top of the row loop: lanes `0..(scalerow/4-1)` each load one
  `uint32` (that is 80 lanes' worth at K=5120, so loop it), then `__syncwarp()`.
  Keep the loads 4-byte aligned and the loop unrolled.
* The k loop then reads `ssc[... + i*kWarp + lane]` -- a shared load, no global traffic.

**Expected signals:** `generate` **must be 16/16**; t=1 must fall below **100.79 ms**.
If the `__syncwarp()` and shared traffic eat the gain, revert and record -- the
ablation is still the only thing that has located this, and a negative here would mean
the 26% is not addressable from the scale path at all.
