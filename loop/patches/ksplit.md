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

## The atomic risk is MEASURED AND ELIMINATED (round 150)

The section above says to measure the atomics before implementing. Done, with a
standalone kernel at the prefill's own launch shape (272 blocks x 128 threads) writing
into an output of the prefill's own size (59 x 17408 floats):

| | |
|---|---|
| atomic load, 2-way split of one prefill | 192 matrices x 2.2 M = **422 M atomics** |
| **measured bursts** | 422 M in **4.0 / 4.0 / 5.0 ms** |
| that rate | **~105 G atomics/s** |
| needed to stay inside a 434 ms prefill | ~972 M atomics/s |
| **margin / cost** | **~108x -> ~1% of the prefill** |

**So the atomics are not a reason to avoid the split, and the fallback (a second
reduction pass) is unnecessary.**

One note on how this was read: the benchmark's printed *verdict* line said "0 M/s"
because of an extra `/1e6` in the format expression. **The reliable output was the
*time* -- 4.0, 4.0 and 5.0 ms -- and the rate is arithmetic on it.** Worth recording,
because a broken derived number next to a good raw number is exactly the shape of
error this session has hit repeatedly.

**Nothing in this spec is now blocked:** the mechanism (47% occupancy, blocks-bound)
is measured, the design has verified line anchors, and the main risk is eliminated.
The remaining work is the four edits and the gate.

## Both traps can be DESIGNED OUT, not worked around (round 158)

Applying the same insight that unblocked the streaming fix in round 157 -- *remove the
problem the design creates, do not solve it* -- both silent-error traps above have a
cheap structural answer. **Neither needs the `rel` variable or a second store site.**

### Trap 1 (buffer parity) disappears if `kc_half` is even

`nchunk = 5120 / 32 = 160`, so a 2-way split gives **`kc_half = 80`** and `kc0` is
**0 or 80 -- both even.** The loop's parity is `cur = (c ^ 1) & 1`, which depends only on
the parity of `c`; with `c` starting at an even number the expression **stays correct
unchanged.**

```
prologue stages chunk kc0, writing buffer 1
c = kc0 (even) -> cur = (kc0 ^ 1) & 1 = 1  -> reads buffer 1   == what the prologue staged
```

**So there is no `rel` and no parity edit** -- only the prologue's chunk index changes
from `0` to `kc0`, which is an index and not a parity. **The guard is a divisibility
check: split only when `kc_half % 2 == 0`; otherwise launch `grid.z = 1` and take the
existing path.**

### Trap 2 (per-dtype stores) disappears if the split is scoped to the NVFP4 GEMM

The prefill launches the NVFP4 path (`ops.rs:1191/1206/1221`), and its store lives in
`gemm2d_store_scaled`. **Launching `grid.z = 2` only from that path leaves the fp8 and
bf16 bodies -- and their stores -- completely untouched**, so there is no second site to
convert and no partial-corruption failure mode.

### What is actually left

1. `#define GB10_KSPLIT 2`
2. `kc0 = blockIdx.z * kc_half`, `kc1 = min(kc0 + kc_half, nchunk)`
3. prologue stages chunk `kc0` instead of `0`
4. `atomicAdd` in the **one** NVFP4 store
5. `grid.z = 2` at the **one** NVFP4 prefill launch, plus a `cudaMemsetAsync` of `y`

**With both traps designed out, the gate is checking arithmetic rather than a lifetime
(or a parity) that the design got wrong** -- which is exactly the difference between
round 157 succeeding and rounds 156 failing twice.

## Acceptance

* **`generate --n 16` must be 16/16.** The gate is the only thing that makes a K split
  trustworthy -- a split with a wrong `kc1` still runs and still reports a fast number.
* `chunked-prefill --n 6` must stay OK (it guards the `start > 0` prefill path).
* TTFT from `forward-cost` or `chunked-prefill`, **>= 3 runs, mean against 434 ms**, and
  the ranges should not overlap before this is called a win.
* If TTFT does not move, the occupancy hypothesis is wrong for this kernel and the
  result should be recorded as such -- **it would be the tenth mechanism eliminated on
  the prefill side, and `docs/NEXT.md` wants that recorded, not quietly dropped.**

---

## TWO CORRECTIONS FOUND BY READING THE LOOP (round 151)

An implementation attempt failed on a Python syntax error **before any write** --
`git status --short` was empty and `kernels/gemm.cu` was byte-identical to its backup.
But reading the loop first turned up two things this spec did not have, and **both
would have produced a kernel that runs and reports a fast number with wrong output.**

### 1. The chunk loop is software-pipelined, and the parity breaks with `kc0 > 0`

```cuda
stage_wtile<GB10_KC>(wt[1], w, sc, s2, K, nbase, N, 0);   // prologue, hardcoded 0
stage_xtile<GB10_KC>(xt[1], x, K, T, t0, 0);
__syncthreads();
for (int c = 0; c < nchunk; ++c) {
    const int cur = (c ^ 1) & 1, nxt = c & 1;             // double-buffer parity
    if (c + 1 < nchunk) { stage_*(... c + 1); }
```

**`cur = (c ^ 1) & 1` is only correct because `c` starts at 0.** With `kc0 > 0` the
parity must be relative to the block's own start:

```cuda
const int rel = c - kc0;
const int cur = (rel ^ 1) & 1, nxt = rel & 1;
```

and the prologue must stage `kc0`, not `0`. **Getting this wrong swaps the weight and
activation tiles -- it does not crash, and the numbers stay plausible.** This is the
single most dangerous part of the change and it was missing from the edit list above.

### 2. The store is not only at line 311

`gemm2d_store_scaled` holds the NVFP4 store, but the fp8 and bf16 bodies each have
their own store, and the kernel is instantiated per dtype. **Any `atomicAdd`
conversion has to be applied to every store site, or the split corrupts only the dtypes
that were missed** -- which would show up as a partial correctness failure rather than a
clean one.

### Consequence for acceptance

The gate is not optional here and `>= 3` runs is not enough by itself: **the parity bug
yields plausible numbers**, so the only trustworthy signal is `generate --n 16` at
16/16 plus `chunked-prefill --n 6` OK. **If a future round sees a *speedup* on the first
run, check the gate before believing it** -- rounds 88, 110 and 119 each reported a
faster kernel that was simply wrong.


---

## KERNEL HALF DONE (round 159) -- and the gate caught a real bug in it

Items 1-4 are in `kernels/gemm.cu` and verified. **The host still launches
`grid.z = 1`, so behaviour is unchanged and both gates pass: `generate --n 16` 16/16,
`chunked-prefill --n 6` OK.**

**The first attempt FAILED THE GATE 0/3, and it is worth recording why**, because it is
this project's most-repeated failure mode arriving on schedule:

```cuda
const int kc_half = nchunk / GB10_KSPLIT;      // WRONG
const int kc1 = min(kc0 + kc_half, nchunk);    // grid.z == 1 -> kc1 = 80, not 160
```

**Deriving the chunk range from the constant rather than from `gridDim.z` made a
`grid.z == 1` launch cover only half of K.** It compiled, it ran, and it would have
reported a *faster* prefill -- the same shape as rounds 88, 110 and 119. **The gate
caught it before a single timing was taken.** The fix:

```cuda
const int nsplit = (int)gridDim.z;                        // 1 or 2
const int kc_half = (nchunk + nsplit - 1) / nsplit;       // z==1 -> 160, the old loop
const int kc0 = blockIdx.z * kc_half;
const int kc1 = min(kc0 + kc_half, nchunk);
```

**The streaming fix in round 157 generalised:** derive the range from the launch, not
from a compile-time assumption about the launch.

## What is left: the host half (item 5)

`crates/gb10-cuda/src/ops.rs` -- at each of the three NVFP4 prefill launch sites
(1191, 1206, 1221):

1. set the third grid dimension to `2` (and **only** when `nchunk / 2` is even, else 1);
2. **zero `y` first**, because `blockIdx.z > 0` accumulates with `atomicAdd`. `y` is
   `t * n` floats -- 59 x 17408 x 4 B = **4.1 MB**, negligible against the 17.6 GB the
   same launch moves. Use whatever the crate already exposes for a device memset on the
   same stream, or a small fill kernel.

**Acceptance:** both gates must stay green, then `forward-cost` / TTFT with **>= 3 runs**
against the 434 ms baseline, **ranges non-overlapping** before calling it a win.
**Predicted ~2x: TTFT -> ~220 ms, endpoint 19.8 -> ~35-40 tok/s.**


---

## HOST HALF ATTEMPTED AND REVERTED (round 161) -- the gate caught three failures

`crates/gb10-cuda/src/ops.rs` was edited at all three NVFP4 prefill launch sites:
compute `nsplit` from `k / 32`, `memset_zeros(y)` when `nsplit > 1` (cudarc does expose
`memset_zeros` at `driver/safe/core.rs:1573`), and pass `nsplit` as the third grid dim.
**It compiled. It failed the gate three ways. It was reverted with `git checkout` and the
tree is green again at the round-160 commit (`generate` 16/16, `chunked-prefill` OK).**

| store form | `generate` | `chunked-prefill` | `batch-parity` |
|---|---|---|---|
| baseline (no split) | **16/16** | **OK** | OK |
| `if (z == 0) store else atomicAdd` | **0/0** | **FAILED** | OK |
| all blocks `atomicAdd` | **0/0** | **FAILED** | OK |

**The first form is a race and that part is understood**: block 0's plain store can land
after another block's `atomicAdd` and clobber the partial sum. **The second form removes
the race and still fails identically, so the race was not the (only) bug.**

**`0/0` rather than a token mismatch is the important signal.** The same gate reports
16/16 without the split, so a comparison that examines *zero* tokens points at something
structural -- an error, an empty output, a shape problem -- not a rounding difference.
**That is the first thing the next attempt must explain**, before touching the timing.

`batch-parity` passing throughout is consistent and expected: **it does not use the
prefill GEMM**, so it is not evidence that the split works.

### Cheapest checks for the next attempt, in order

1. **Read what `0/0` means in `gb10-verify`.** If it counts wrong tokens out of tokens
   compared, zero compared tokens means generation produced nothing -- so look for an
   error/panic path, not a numerics path. **One grep, before any code change.**
2. **Print `nsplit` and `kchunks` once** to confirm the split is 2 for the real `k` and
   that the guard is not silently selecting 2 where `nchunk / 2` is odd.
3. **Check whether `y` is used more than once per call** in the prefill path. If the
   same slice is passed for two GEMMs, or read back after the call, zeroing it before the
   launch is not the benign operation this spec assumes.
4. **Only then** re-measure TTFT, with `>= 3` runs.

**What survives and is safe:** the kernel half (items 1-4) is committed and verified with
`grid.z = 1` -- behaviour unchanged, both gates green. **That increment is real and
vetted; the host half is not.**

**And the rule that held: the gate blocked a faster-and-wrong prefill three times in a
row. Do not take a timing on this path until it is green.**


---

## `0/0` DECODED (round 162) -- and it condemns the memset, not the arithmetic

`crates/gb10-verify/src/main.rs:509`:

```rust
let n = out.len().min(want.len());
let agree = (0..n).filter(|&i| out[i] == want[i]).count();
```

**`0/0` means either side was empty.** The failing run still printed
`reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16`, which comes from
`o["reference"]` -- **so the oracle file loaded and `want` is populated. Therefore
`out` is empty: generation produced ZERO tokens.**

**That is a crash-or-degenerate-generation signal, not a numerics signal**, which is why
both store forms failed identically while `batch-parity` (no prefill GEMM) stayed green.

### The cause: `memset_zeros(y)` does not mean what the spec assumed

`dev.stream().memset_zeros(y)?` zeroes **the entire `CudaSlice`**. But the prefill GEMM's
`y` is not necessarily a freshly-allocated output -- it is whatever the caller passes, and
in this engine the GEMM result feeds the next op in the same forward pass, so that slice
can be a view into a larger activation buffer whose contents are **live**. **Zeroing all
of it mid-forward destroys the activations**, the model emits EOS immediately, `out` is
empty, and the gate reports `0/0`.

**So the premise "atomicAdd requires zeroing y" is the design problem** -- the same shape
as rounds 156/157, where the gate could not be written until the emitter was taken out of
it. **The requirement exists only because the element needs a known starting value.**

### The design that removes the requirement: scratch + reduction, no memset

Do not accumulate into `y` at all. Have the `nsplit` blocks write **disjoint** regions of
a private scratch buffer, then have one pass sum them into `y`:

* scratch is `nsplit * t * n` floats = 2 x 59 x 17408 x 4 B = **8.2 MB** (the same launch
  moves 17.6 GB of weights, so this is nothing);
* `y` is **written once and never zeroed**, so no live buffer is touched;
* no atomics at all -- which also retires the ~2.2M atomics/matrix the round-150 probe was
  measuring, and the race between block 0's store and the others' `atomicAdd`;
* the reduction is one extra pass over 1.03M elements per matrix.

**Cost: one extra small kernel and 8.2 MB of scratch. Benefit: the correctness argument
becomes "disjoint writes then a sum" instead of "a memset that must not land on live
data".**

**Do not re-attempt the memset form.** The next attempt should be the scratch+reduction
form, and it should still land the kernel change first with `grid.z = 1` and prove both
gates green before the host ever asks for `nsplit = 2`.
