# Ready-to-apply: raise the GEMV batch cut to 32

**Round 119 proposed raising the `t <= 16` GEMV cut to cover a 59-token prefill.
That is impossible, and the check takes one grep:**

* `kernels/gemv.cu:340` -- `#define GB10_BATCH_MAX 16`
* `crates/gb10-cuda/src/kernels.rs:18` -- *"`GB10_BATCH_MAX` in `kernels/gemv.cu`.
  Larger batches fall back to the [GEMM]"*
* `crates/gb10-model/src/weights.rs:95` -- the cut itself, `if self.n < 256 || t <= 16`

A GEMV over 59 tokens needs `ceil(59/16)` = **4 weight passes**. At the GEMV's
measured 148 GB/s that is 4 x 119 = **476 ms**, against the GEMM's single 434 ms
pass. **The GEMM is correct for t=59, and the `t <= 16` cut is already in the right
place for the crossover it was chosen for.**

**But the arithmetic points somewhere better.** The two paths:

| path | weight passes | rate | cost |
|---|---|---|---|
| GEMV (`t <= BATCH_MAX`) | `ceil(t/16)` | 148 GB/s | `ceil(t/16) x 119 ms` |
| GEMM (`t <= 64`) | `ceil(t/64)` | 32 GB/s | `ceil(t/64) x 434 ms` |

The GEMV wins while `ceil(t/16) <= 3`, i.e. **t <= 48** -- everything in between is
currently paying 434 ms for work that could cost 238.

**The edit: `GB10_BATCH_MAX` 16 -> 32.** Then a 59-token prefill is 2 GEMV passes =
**238 ms against the current 434, ~45% faster**, and the endpoint's 16 serial
prefills (7.63 s) fall the same way. That is the largest single lever left, and it
attacks TTFT and endpoint concurrency together.

**The constraint, and the fix that has already been measured:**

`acc[ROWS][BMAX]` at `ROWS`=4/`BMAX`=32 is 128 registers -- too many. At
**`ROWS`=2/`BMAX`=32 it is 64**, the same as today. Round 104 measured `ROWS`=2 at
`BMAX`=16 as **neutral at B=4 and 12% worse at B=16**, so the cost of `ROWS`=2 is
known and small, while the prefill win is ~45%.

**Edits:**
1. `kernels/gemv.cu`: `#define GB10_BATCH_MAX 16` -> `32`.
2. `nvfp4_gemv_batch_tmpl<4, GB10_BATCH_MAX>` -> `<2, GB10_BATCH_MAX>` (and the fp8
   one if it has the same shape) -- **only if ptxas reports spills at `ROWS`=4**;
   try 4 first, since 128 registers did not spill for `acc[8][8]`.
3. `crates/gb10-model/src/weights.rs:95`: `t <= 16` -> `t <= 32`.
4. `crates/gb10-cuda/src/kernels.rs`: the batch launch must pick the GEMV for
   `B <= 32` (check that `GB10_BATCH_MAX` there matches).

## TESTED AND REJECTED (round 121)

Applied as steps 1 and 3 (`GB10_BATCH_MAX` 16 -> 32, `t <= 16` -> `t <= 32`):

| | baseline | with `BATCH_MAX`=32 |
|---|---|---|
| `generate` | 16/16 | **16/16** |
| TTFT | 434 ms | **433.1 ms (no change)** |
| `nvfp4_gemv_batch_kernel` | 128 regs, 256 B stack | 128 regs, **512 B stack** |

Reverted.

**The spec above was wrong, and the reason matters: `BATCH_MAX`=32 lets the GEMV
serve `t <= 32`, NOT `t = 59`.** The kernel takes `B` as a runtime parameter bounded
by the compile-time `BMAX`, so a 59-token prefill needs `BMAX >= 59` -- which means
`acc[ROWS][59]`, far past the register budget. **Serving an arbitrary `t` through the
GEMV needs a t-tiled outer loop in the GEMV path, which does not exist.** Raising the
constant cannot get there, and the crossover table above is therefore not actionable
as written.

The change is also not free: `BMAX`=32 grows the decode kernel's stack frame from 256
to 512 B even at `B`=16, and round 104 already measured that added pressure in this
kernel as harmful (12% at `B`=16). So it costs on the decode path and buys nothing on
the prefill path.

**What would actually work**, if a future round wants the GEMV for long prefills: a
loop over `t` in `BMAX`-sized tiles *inside* `forward`/`forward_prefill`, so a
59-token prefill becomes 2 calls at `B`=32. That is real code, not a constant, and it
must be measured against the GEMM's single pass rather than assumed.
