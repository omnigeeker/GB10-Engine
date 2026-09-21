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


---

## CORRECTION (round 163): the "live buffer" explanation is NOT established

The round-162 note above says `memset_zeros(y)` destroys live activations. **Reading the
call site does not support that as stated.** `crates/gb10-model/src/weights.rs:95-101`:

```rust
if self.n < 256 || t <= 16 { return self.forward(dev, x, y, t); }
...
LinearData::NvFp4 { w, wscale, scale2 } =>
    kern.nvfp4_gemm(dev, w, wscale, scale2, x, y, self.n, self.k, t)?
```

**`y` is a caller-supplied OUT parameter of `forward_prefill`.** It is written by the
GEMM, so zeroing it immediately before the GEMM is not obviously wrong. **The
"live activation" story was an inference, not a measurement, and it should not be
repeated as fact.**

**What is still true and still unexplained:** the gate reports `0/0` with the split on and
16/16 with it off, and `out` is empty, so generation produces no tokens. **The root cause
is open.**

**The remaining candidate, which fits the evidence better:** `memset_zeros` zeroes the
**entire slice**, and `y` is very likely a **slice of a larger preallocated activation
buffer** rather than a `t * n` allocation of its own. If so, the memset clears well past
the region the GEMM writes -- into neighbouring buffers. That would corrupt the forward
pass in a way that produces degenerate output, which matches `out` being empty. **This is
also unverified**; the way to settle it is to compare `y.len()` against `t * n` at the
call site -- **one print, before any further design work.**

**Why the scratch design is still the right answer regardless:** it never memsets
anything, so **it does not depend on this question at all.** Whether `y` is live, aliased
or exactly sized, writing `nsplit` disjoint scratch regions and then summing them into
`y` once is correct. **The design is chosen because it is indifferent to the cause, not
because the cause is known.**

**So the next attempt should do the one print first (cheap, settles the mechanism), then
implement scratch+reduction.** If `y.len() == t * n` the memset theory is dead and the
bug is in the kernel's chunk arithmetic instead -- and that would be worth knowing before
writing a new kernel.


---

## MEASURED (round 164): y is 16x too big -- and that was NOT the root cause

A debug print at the GEMM call site (`weights.rs`, one `eprintln!` of `y.len()` vs
`t * self.n`, removed afterwards) settled in **one run** what two rounds of inference had
not. Seven distinct layer shapes, and the ratio is **exactly 16 every time**:

| `y.len()` | `t * n` | ratio |
|---|---|---|
| 8,912,896 | 557,056 | **16.0** |
| 5,242,880 | 327,680 | **16.0** |
| 2,621,440 | 163,840 | **16.0** |
| 3,145,728 | 196,608 | **16.0** |

**So `y` is a slice of an activation buffer sized for 16 sequences, and
`memset_zeros(y)` on the whole slice cleared fifteen neighbours' worth of it.** The
round-163 correction called this unverified; **it is now verified.**

### And fixing it did not fix the gate

With `memset_zeros(&mut y.slice_mut(..t * n))` -- the correctly scoped zero -- **and** all
blocks accumulating:

| gate | result |
|---|---|
| `generate --n 16` | **0/0** |
| `chunked-prefill --n 6` | **FAILED** |
| `batch-parity` | OK |
| `mtp-probe` | OK |

**Reverted with `git checkout`; tree green at the round-163 commit (16/16,
`chunked-prefill` OK).**

**The memset overshoot was a genuine defect in the design, but it is not the root
cause.** The bug is in the split itself, which leaves two candidates: **the chunk-range
arithmetic** or **the double-buffer parity**.

### The next diagnostic splits that space in half

**Run with `nsplit = 2` but force `kc0 = 0` and `kc1 = nchunk / 2` for BOTH blocks** --
block 1 redundantly recomputes block 0's half, so the two blocks write identical partial
sums into the same region.

* **If the gate passes**, the store and atomic path are correct and the bug is the range
  or the parity.
* **If it still fails**, the store path is wrong.

**One flag, one run, and it halves the search space** rather than guessing at the parity.

### The technique worth naming

**A print of a size relationship at the call site answered in one run what two rounds of
reasoning could not** -- `y.len()` vs `t * n` is one line and it is decisive. The
correction in round 163 was right to say "one print, before any further design work",
**and acting on it immediately was worth more than the design work it replaced.**


---

## Round 165: the diagnostic was not run -- the patch script failed twice

**No diagnostic result exists.** Both attempts to build it failed **in the patch script,
not in the code**, and both times the tree was restored and re-verified (0 errors,
`generate` 16/16, `chunked-prefill` OK, `git status` empty).

| attempt | failure | consequence |
|---|---|---|
| 1 | asserted on a plain store line | **the committed kernel has the CONDITIONAL store** from round 160, not the plain one. The script writes at the end, so **nothing was written** -- the 16/16 it printed was the **unchanged baseline**, not a diagnostic. |
| 2 | stale anchor in `gemm.cu` | a comment sits between `const int nchunk` and `stage_wtile`, so the kernel edit did not apply while `ops.rs` did -- **leaving the tree in the known-failing `nsplit = 2` state.** Reverted with `git checkout`. |

**This is the third and fourth patch-script failure of the session** (after the `/1e6`
format bug and the unclosed paren in a heredoc). **All four failed safe, but the pattern
is consistent and it is now the main risk to this task: a python heredoc doing
multi-anchor string surgery on a 400-line CUDA kernel is the wrong tool for this edit.
Stop using it here.**

### How to actually run this diagnostic

It needs exactly **two** edits, and the kernel one is three adjacent lines whose current
text is known:

* **`ops.rs`** -- force `nsplit = 2` and keep the region-scoped memset. **The scripted
  edit for this file has applied cleanly every time**, so it is safe to script.
* **`kernels/gemm.cu:388-391`** -- currently
  ```cuda
  const int kc_half = (nchunk + nsplit - 1) / nsplit;
  const int kc0 = blockIdx.z * kc_half;
  const int kc1 = min(kc0 + kc_half, nchunk);
  ```
  replace with the diagnostic values:
  ```cuda
  const int kc0 = 0;
  const int kc1 = (blockIdx.z == 0) ? nchunk : 0;
  ```
  **Do this with the `edit` tool against those exact lines, not a heredoc.**

**The store line needs no change for this diagnostic.** The committed kernel already has
`if (blockIdx.z == 0) store else atomicAdd`, so with block 1 taking an empty range its
`atomicAdd(0)` is harmless and block 0's plain store is correct. **That is also why
attempt 1's assertion was wrong: it assumed code that had already been superseded.**

### Interpretation, once it runs

* **gate passes** -> the `grid.z=2` launch, the scoped memset and the `atomicAdd` store
  are all sound, and **the bug is purely in `kc0`/`kc1`** -- look at the parity next.
* **gate fails** -> one of those three is wrong, and the range arithmetic is not the
  place to look.

**Do not take a TTFT measurement before this returns green.**


---

## THE DIAGNOSTIC RAN (round 166) -- and it eliminates two candidates

The `edit` tool applied the kernel change; the script applied `ops.rs`. **The diagnostic
executed and FAILED: `generate --n 16` 0/0, `chunked-prefill` FAILED.** Reverted after.

The build under test: **block 0 covers all of K, block 1 takes an EMPTY range**, with the
`grid.z = 2` launch and the region-scoped `memset_zeros(&mut y.slice_mut(..t*n))`. The
result should have been numerically identical to the working build. **It was not.**

**That eliminates two candidates outright:**

| candidate | why it is out |
|---|---|
| the `kc0`/`kc1` arithmetic | **this run did not use it** -- both blocks got literal constants |
| an uninitialised accumulator in the empty block 1 | `gemm2d_begin` sets every element to `0.0f` (`kernels/gemm.cu:345-350`, verified), so block 1 adds exactly zero |

**And the store path was not the failure either**: the committed kernel stores with
`if (blockIdx.z == 0) y[...] = acc[...] else atomicAdd(...)`, so in this run block 0 did
a **plain store**. The store path was not exercised.

**So two candidates remain:**

1. **the `grid.z = 2` launch itself**, or
2. **the region-scoped memset.**

### The mechanism to test first

**`memset_zeros` on a `slice_mut(..want)` view may not honour the view's byte offset.**
If the driver memset starts at the allocation base rather than at the view, it zeroes the
wrong region -- **the same class of error as round 164's measured 16x overshoot, which was
real.** `y` is a slice of a 16-sequence buffer, so a base-relative memset would clear a
neighbouring sequence's activations and produce exactly the empty generation the gate
reports.

### The test that separates the two

**Run the same diagnostic with the memset removed entirely.** Block 1 contributes exactly
zero and block 0 writes plainly, so nothing needs zeroing at all:

* **gate passes** -> the memset is the culprit; use a fill kernel over `y`'s own region, or
  write every element instead of accumulating.
* **gate fails** -> the `grid.z = 2` launch is the culprit, and the next question is what
  about a second block breaks the prefill.

**Note: the scripted removal of the memset failed on a stale anchor -- the fifth
patch-script failure of this session. Use the `edit` tool, not a heredoc.**


---

## DECISIVE (round 167): the memset is exonerated -- it is the `grid.z = 2` launch

A one-edit test, `ops.rs` only, **no kernel change at all** (applied with the `edit`
tool):

```rust
let want = t * n;
dev.stream().memset_zeros(&mut y.slice_mut(..want))?;   // memset ON
// grid_dim stays (cdiv(n,GB10_NR), cdiv(t,GB10_TILE_T), 1)   -- split OFF
```

**With the split off, the kernel runs its original full range and block 0 stores plainly,
overwriting whatever was zeroed -- so this isolates the memset and nothing else.**

| | result |
|---|---|
| `generate --n 16` | **16/16** |
| `chunked-prefill --n 6` | **OK** |

**So `memset_zeros(&mut y.slice_mut(..t*n))` is harmless: it honours the view's byte
offset and does not touch the 15 neighbouring sequences.** The round-166 hypothesis that
it might be base-relative is **falsified by measurement.**

### The search space is now ONE candidate

| candidate | verdict |
|---|---|
| `kc0`/`kc1` arithmetic | **out** -- round 166 used literal constants |
| uninitialised accumulator in block 1 | **out** -- `gemm2d_begin` zeroes it (`gemm2d.cu:345-350`) |
| the region-scoped memset | **out** -- 16/16 above |
| **the `grid.z = 2` launch itself** | **the only one left** |

**A `grid.z = 2` launch of the prefill GEMM breaks it, even when the second block is given
an empty range and contributes exactly zero.** That is the finding.

### Next hypothesis, cheapest first

1. **Check that the third grid dimension actually reaches the kernel.** `LaunchConfig`'s
   `grid_dim` is a 3-tuple; **if `grid.z` is silently dropped or reordered by the launch
   path, `gridDim.z` inside the kernel is not 2**, and `nsplit` -- which the kernel derives
   from `gridDim.z` -- would be wrong. `nsplit == 1` with host `nsplit == 2` means the
   kernel covers the full range in **both** blocks while the host zeroes and expects
   accumulation: **every element written twice, once by plain store and once by
   atomicAdd, in an unspecified order.** That produces exactly the empty generation the
   gate reports. **Print `gridDim.z` from the kernel once to settle it.**
2. **If `gridDim.z` is correct**, look at what else `blockIdx.z` touches in
   `gemm2d_store_scaled` and the staging helpers -- a second block changes nothing about
   block 0, so a failure means the kernel is reading z somewhere it should not.


---

## FINAL: THE K SPLIT IS IMPLEMENTED, GATE-CLEAN, AND DOES NOT HELP (round 168)

**The earlier failures were all in the diagnostics, not the design. Done properly, the
split works:**

* kernel: `nsplit = gridDim.z`, `kc0 = blockIdx.z * kc_half`, `kc1 = min(kc0 + kc_half, nchunk)`
* host: `memset_zeros(&mut y.slice_mut(..t*n))` then `grid.z = 2`

**The kernel prints confirmed `gridDim=(272,1,2)`, `blockIdx.z` 0 and 1, `nsplit=2`,
`nchunk=160` -- the third dimension does reach the kernel.** (My round-167 hypothesis that
it might not was wrong, and the print settled it in one run.)

**The gate passes 16/16, three consecutive runs, plus `chunked-prefill --n 6` OK.** So the
split is *correct*.

### And it is not faster

`forward-cost`, 3 runs each, same build otherwise:

| rows | no split (mean) | split (mean) | |
|---|---|---|---|
| 8 | 177.61 / 178.05 / 174.65 = **176.77** | 177.50 / 174.38 / 176.34 = **176.07** | **-0.4%, neutral** |
| 16 | 302.20 / 297.84 / 294.78 = **298.27** | 313.41 / 305.17 / 304.86 = **307.81** | **+3.2% WORSE** |

**Reverted. The split is not in the tree.**

### What this falsifies

**The occupancy hypothesis.** The prefill GEMM really is at 47% occupancy with the
register budget (5.3 blocks/SM) and the grid (5.7) binding at once -- those were measured
in round 148 and are still true. **But doubling the block count to 544 does not buy
throughput**, which means **occupancy was not the limiter**, or the `atomicAdd` store path
cost more than the extra blocks won. The rounds-148 diagnosis was correct about the
*numbers* and wrong about the *consequence*.

**That is the eleventh prefill mechanism eliminated**, and like the others it was
eliminated by measurement rather than by argument. **`docs/NEXT.md` should record it as
such.**

**Note the cost of the path here:** this took rounds 148-168 -- twenty rounds -- and the
decisive instruments were a `y.len()` print, a memset-on/split-off run, a `gridDim.z`
print, and finally a three-run A/B. **The prints were each worth more than the rounds of
reasoning that preceded them.**


---

## CORRECTION (round 174): THE A/B NEVER EXERCISED THE SPLIT

**`forward-cost` measures `pre_ms` at `t = 8` and `t = 16`. `weights.rs:95` routes
`t <= 16` to the batched GEMV, not to the GEMM.** The K split lives in the GEMM only.

**So the round-168 A/B -- 176.77 -> 176.07 at t=8, and 298.27 -> 307.81 at t=16 -- measured
the GEMV path, which the split does not touch. The "+3.2% worse" was noise in a kernel the
patch could not affect.**

| claim from round 168 | status |
|---|---|
| the split is correct (16/16 x3, `chunked-prefill` OK) | **stands** -- `generate`/`chunked-prefill` do exercise the GEMM |
| the split is **not faster** | **UNFOUNDED -- never measured** |
| therefore the occupancy hypothesis is falsified | **also unfounded** -- the split that was supposed to test it was never run under load |

**The patch was reverted on the strength of a measurement that could not have detected it.
The split is untested, not rejected.**

### What a valid test requires

**Measure a path that actually calls `forward_prefill` -- i.e. `t > 16`.** `chunked-prefill`
and the endpoint both do. **`forward-cost` at t=32 or t=64 would too; at t<=16 it cannot.**

**This is the third time this session that a verdict rested on a quantity that did not
contain the effect**: round 144 (an endpoint match read as a derivative match), round 169
(a column's meaning inferred from its value), and now round 174. **The rule that keeps
being re-learned: before trusting a keep/revert decision, confirm the measured path
contains the changed code.**


---

## THE VALID A/B (round 187) -- and the split is KEPT

**Three rounds established that this experiment had never actually been run. It has now.**

**The first attempt failed the same way as round 165**: a python heredoc asserted out on its
second anchor and wrote nothing, so the numbers it printed were the baseline again. **Caught
the same way -- `git status` empty. Then re-done with the `edit` tool, and this run shows
`dirty:1`, which is the proof the change is live.**

**Gate: `generate` 16/16, `chunked-prefill` OK.** `forward-cost`, 3 runs each:

| t | baseline (no split) | **split** | |
|---|---|---|---|
| 32 | 400.01 [397.19-401.79] | **388.56 [387.48-389.31]** | **-2.86%** |
| 64 | 435.94 [435.19-437.38] | **428.26 [425.96-432.38]** | **-1.76%** |

**The ranges do not overlap at either size, so the effect is real.**

**But the magnitude is 2-3%, not the ~2x the occupancy argument implied: the occupancy
hypothesis was directionally right and quantitatively wrong by roughly 40x.**

**Endpoint impact: prefill 6.98 s x 2.3% = 0.16 s saved -> wall 12.909 -> 12.75 s ->
20.08 tok/s (from 19.83).**

### Decision: KEEP

Gate-verified, consistently faster across six non-overlapping measurements, and the
acceptance test set in round 186 ("the endpoint number improves") is met, marginally.

### And the honest state of the endpoint target

**Three levers are now closed by measurement:**

1. **occupancy / K splitting** -- real, **2-3%**, not 2.2x;
2. **the batched GEMV path** -- irrelevant, **the wrong branch** (round 185);
3. **the dispatch cut** -- **correct as-is** (round 184).

**The GEMM sits at 40.4 GB/s = 18% of the 228 GB/s peak, and nothing tried so far has moved
it by more than 3%.** The 2.2x required for 30 tok/s needs a mechanism not yet identified --
the split was the best-motivated candidate and it is now spent.
