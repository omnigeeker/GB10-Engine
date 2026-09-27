# Prefill GEMM on bf16 tensor cores — implementation spec

## Why this is the whole remaining gap

Measured per-chunk decomposition of the prefill (see `comparison.md`):

| | GEMM | attention |
|---|---|---|
| 8K | 49.6 s (93%) | 4.4 s (7%) |
| 32K | 202.6 s (75%) | 65.2 s (24%) |

The prefilled GEMM is **~7 TFLOPS of fp32** on CUDA cores. The fp32 FMA
theoretical floor is 128 FMA/cycle/SM * 48 SM * 1.5 GHz = 9.2 TFLOPS, so the
kernel is already at ~76% of what fp32 can ever do. Getting to llama.cpp's
~43 TFLOPS needs tensor cores; there is no fp32 tuning that reaches it.

## Why it should still pass the token-exact gate

`generate` compares against `fixtures/oracle`, whose weights are described as
"dequantized from NVFP4 to bf16". Two facts make bf16 the natural arithmetic:

1. `stage_wtile` in `kernels/gemm.cu` already converts every weight to bf16
   before staging it to shared (`__bfloat16_as_ushort(__float2bfloat16_rn(...))`).
   The *existing* GEMM already consumes bf16 weights.
2. Any bf16 value is exactly representable in fp32, which is why the current
   mixed bf16-weight/fp32-accumulate kernel is bit-comparable to the oracle.

**CORRECTION (round 15), because the paragraph that used to be here was wrong.**
I wrote that a cuBLAS bf16 GEMM would differ from today only in accumulation
order, not operand precision. That is false, and the way it is false matters:

- The current kernel's weights are bf16 but its **activations are fp32**
  (`xt[2][GB10_KC][GB10_XSTRIDE]` is `float`), and its **output is fp32**
  (the accumulator is written as `float`). Only the weights are bf16.
- cudarc's `impl Gemm<half::bf16>` passes `CUDA_R_16BF` for **A, B *and* C**.
  So the safe API would round the activations to bf16 on the way in and the
  output to bf16 on the way out. Those are two real precision reductions that
  the oracle the gate compares against does not take.

So this is not the same class of change as the `GB10_KC`/`KSPLIT` variations.
It has to be validated empirically by `generate` against the oracle, and it may
well fail. Two ways to soften it, in order of preference:

1. Call `cublasGemmEx` through `cudarc::cublas::sys` directly with
   `C = CUDA_R_32F` (fp32 output) and bf16 inputs. That removes the output
   rounding and leaves only the activation rounding. `cudarc::cublas::sys` is
   public, so this is a small amount of unsafe FFI rather than a rewrite.
2. If the activation rounding is what breaks the gate, keep fp32 activations and
   use TF32 tensor cores (`CUBLAS_COMPUTE_32F_FAST_TF32`). That is roughly half
   of bf16 throughput, so ~40 TFLOPS, which puts 8K at ~13.1 s against
   llama.cpp's 10.58 s -- i.e. *not* a win. It is the fallback, not the plan.

Either way the honest statement is: the tensor-core path is a bet on the gate
tolerating reduced activation precision, and that bet has not been tested yet.

### Measured: the bet is won

It has now been tested, before writing any of the cuBLAS path. `gemm2d_outer_bf16`
in `kernels/gemm.cu` carries an off-by-default diagnostic (`-DGB10_SIM_BF16_ACT=1`,
define added in `crates/gb10-cuda/build.rs`) that rounds the activations to bf16
and back inside the existing fp32 GEMM -- exactly the rounding a bf16 cuBLAS GEMM
imposes on its input. With it on:

    gb10-verify generate --n 16 --repeat 8 --oracle fixtures/oracle
    -> oracle agreement: 16/16 (100.0%), exact match

**The gate tolerates bf16 activations.** Since activation error is the one that
propagates through all 64 layers, and it survives, the output rounding that
cudarc's safe API would additionally impose is a much smaller concern -- and the
`cudarc::cublas::sys` route with `C = CUDA_R_32F` removes it entirely anyway.

So the plan proceeds on the bf16 path, not the TF32 fallback. The diagnostic is
kept in the kernel, off by default and inert, because it is the cheapest possible
re-test of this question if anything downstream changes the numerics.

## Pieces

1. **cudarc feature.** `crates/gb10-cuda/Cargo.toml`: add `"cublas"` to the
   cudarc feature list. `CudaBlas::new(stream)` gives a handle;
   `impl Gemm<half::bf16> for CudaBlas` lives at
   `cudarc-0.19*/src/cublas/safe/gemm.rs:143` and takes `DevicePtr<half::bf16>`.

1b. **`s2` handling.** `s2` (the per-tensor scale) is not folded into the
   staged weights -- `stage_wtile` takes it and ignores it (`(void)s2`). The
   NVFP4 GEMM applies it as a post-scale on the fp32 accumulator
   (`gemm2d_store_scaled`, `acc[i][j] *= s2`). A dequant kernel must therefore
   **not** fold `s2` into the bf16 weights: dequantise with the group scale only,
   then scale the fp32 result of the GEMM. Folding it in would round each weight
   after scaling instead of scaling the fp32 sum, which is a different number.

2. **Dequant kernels** in `kernels/gemm.cu`, modelled directly on the existing
   staging code.

   **DONE for NVFP4 (round 17).** `dequant_nvfp4_to_bf16_kernel` is implemented,
   wired through `Ops::dequant_nvfp4_to_bf16`, and gated by a new
   `gb10-bench dequant-parity` that compares it against the host reference
   `dequant_nvfp4_row` on a synthetic 256x512 matrix covering all 16 E2M1 codes:

       dequant nvfp4 -> bf16   256 x 512  (256 rows, 131072 elements)
         bit-exact against dequant_nvfp4_row: 131072 / 131072
       dequant-parity: OK

   **DONE for FP8 and the activation cast too (round 18).**
   `dequant_fp8_to_bf16_kernel` and `f32_to_bf16_kernel` are in, wired, and the
   same gate now covers all three:

       dequant fp8   -> bf16   256 x 512  (131072 elements)
         bit-exact against e4m3_to_f32 * scale: 131072 / 131072
       f32 -> bf16 cast: OK
       dequant nvfp4 -> bf16   256 x 512  (256 rows, 131072 elements)
         bit-exact against dequant_nvfp4_row: 131072 / 131072
       dequant-parity: OK

   Note the asymmetry that had to be respected: fp8's single per-tensor scale is
   applied *inside* the dequantise (matching `stage_wtile_fp8`), while nvfp4's
   `s2` must not be.

   Three things that each cost a rebuild and are worth knowing:
   the entry point needs `extern "C"` (the other gemm kernels carry a comment
   explaining that C++ mangling otherwise makes the name unloadable); appending
   to a `.cu` file does not always invalidate the PTX -- `touch` it; and a bf16
   output must be compared against a bf16-*rounded* reference. The nvfp4 check
   passed without rounding only because E2M1 x E4M3 has 7 mantissa bits and so is
   exact in bf16; E4M3 x an arbitrary fp32 scale is not, and the unrounded
   comparison showed 130002/131072 "failures" that were entirely the harness. The
   residual 1039/131072 = 1/128 after that was exactly the two E4M3 NaN encodings
   (0x7F, 0xFF), which the CUDA and host decoders disagree on and which no real
   weight contains.
 Each takes the same packed weight/scales and writes bf16 to a
   **global** scratch buffer instead of shared:
   - `dequant_nvfp4_to_bf16` — copy the index math from `stage_wtile`
     (that is the authoritative on-device layout), write
     `__float2bfloat16_rn(dequantized)`.
     The CPU reference for the layout is `dequant_nvfp4_row` in
     `crates/gb10-bench/src/reference.rs:49` — read it first, it defines the
     group-16 scale and per-tensor `scale2` convention.
   - `dequant_fp8_to_bf16` — E4M3 to bf16, trivial.
   - bf16 weights need no kernel; pass the existing buffer directly.
   - `f32_to_bf16` for the activations, one buffer reused per GEMM.

3. **Scratch.** The largest matrix is the MLP gate/up at 17408 x 5120, which is
   **178 MB** as bf16. Allocate one scratch buffer of that size and reuse it for
   every matrix in the prefill. Do **not** cache dequantised weights across
   chunks — the whole point is that token generation is bandwidth-bound on the
   NVFP4 bytes, and a 178 MB bf16 cache would be a different (worse) tradeoff.

3b. **Epilogue (DONE and verified, round 20).** The cuBLAS output is bf16 (cudarc's safe `Gemm<half::bf16>`
   fixes A, B *and* C to `CUDA_R_16BF`), so `forward_prefill` still needs a
   bf16 -> fp32 convert-and-scale epilogue to apply `s2` and hand fp32 back to
   the rest of the layer. Rounding the output to bf16 is the same class of
   rounding as the activation rounding already measured to keep the gate at
   16/16, since the output is simply the next layer's activation.
   `bf16_to_f32_scaled_kernel` does the convert-and-scale (taking `has_scale` as
   an int so the caller can always pass a valid slice), and `u16_to_bf16_kernel`
   bridges `LinearData::Bf16`, which stores weights as `CudaSlice<u16>`, to the
   `CudaSlice<half::bf16>` that cuBLAS needs. `cublas-parity` now covers the
   whole pipeline:

       epilogue bf16->f32 * s2: exact: 816 / 816
       weights u16 -> bf16 operand: exact
       cublas bf16 layout: y[17,48] = x[17,32] * W[48,32]^T
         exact against host matmul: 816 / 816
       cublas-parity: OK

4. **Scratch budget, and a correction (round 21).** The dispatch needs three
   shared bf16 buffers, and working out their sizes turned up an error in my own
   accounting. `forward_normed` explicitly leaves the final-norm rows of every
   position in `state.normed` "with no `lm_head`", and `prefill_seq` then scores
   **only the last row**; `state.logits` is sized `vocab * n_seq`, not
   `vocab * t`. So `lm_head` is not in the prefill at all -- it is a one-row GEMV
   for the last token.

   I had been counting `lm_head` as ~6% of the prefill's GEMM FLOPs in the
   per-chunk arithmetic. It is not in that path, so the measured `G` is entirely
   layer projections; the correction goes the right way (there is slightly less
   work than I assumed) but the number was wrong.

   It also means no `t x vocab` buffer is needed. Had `lm_head` been scored for
   all rows it would want `y` at 2048 x 248320 = 2.0 GB in fp32 / 1.0 GB in bf16,
   which would have dominated the budget. The actual requirement is:

   | buffer | max shape | bf16 |
   |---|---|---|
   | `W` dequantised | 17408 x 5120 (MLP gate/up) | 178 MB |
   | `x` activations | 2048 x 17408 (MLP down) | 71 MB |
   | `y` output | 2048 x 17408 (MLP gate/up) | 71 MB |
   | | **total** | **321 MB** |

   All three are shared across every `Linear` and grown on demand, so the
   allocation is one-off. 321 MB against a 40 GB KV budget is not a constraint,
   which removes the main risk I had been carrying about this step.

4. **Dispatch (the only step left).** `Linear::forward_prefill` in `crates/gb10-model/src/weights.rs`
   — replace the GEMM branch (the `n >= 256 && t > 16` side of the existing
   crossover) with: dequant -> convert x -> `blas.gemm` -> (x stays fp32 for the
   rest of the layer). Keep `forward` (the NVFP4 GEMV) for `t <= 16` and for
   decode, so the OTPS win is untouched.

5. **Layout (DONE and verified, round 19).** `C[t, n] = X[t, k] * W[n, k]^T`,
   both row-major:
   `cublasGemmEx(OP_T, OP_N, m = n_out, n = t, k = k_dim, A = W (lda = k_dim),
   B = x (ldb = k_dim), C = y (ldc = n_out))`, compute type `CUBLAS_COMPUTE_32F`,
   scale type `CUDA_R_32F`. Note `m` is the *weight* dimension and `n` is the
   token dimension — that is the whole trick of this layout, and getting it
   backwards is the likely first failure. `Ops::cublas_gemm_bf16` implements
   this and `gb10-bench cublas-parity` gates the layout against a host matmul on
   integer-valued inputs (so bf16 products and fp32 sums are exact and the
   comparison is exact, not tolerance-based):

       cublas bf16 layout: y[17,48] = x[17,32] * W[48,32]^T
         exact against host matmul: 816 / 816
       cublas-parity: OK

   The gate also detects the specific silent failure worth fearing -- a wholesale
   transposed result from swapping `m`/`n` or the two transpose flags -- and says
   so explicitly rather than just reporting a mismatch count.

## Verification order

1. `gb10-verify generate --n 16 --repeat 8 --oracle fixtures/oracle` — the
   token-exact gate. This is the one that matters; fp32 accumulation order is
   allowed to differ, precision is not.
2. `gb10-verify forward-cost` — confirm the small-t shapes did not regress.
3. `gb10-verify prefill-shape --limit 8225` — read the per-chunk times. **The
   success criterion is that the per-chunk constant G drops from 12.4 s to
   ~2 s**; if it does not, the cuBLAS path is not being hit or is being
   launched per-matrix with the wrong shape.
4. Only then the end-to-end 8K / 32K runs.

## Measured: what bf16 tensor cores actually do on this part

`gb10-bench cublas-gemm` (added this round) runs a real cuBLAS bf16 GEMM with
fp32 accumulate at the shapes the prefill actually uses. This replaced an
assumption with a number, and the assumption was **too pessimistic by 2x**:

| shape (n x k x t) | ms | TFLOP/s | vs fp32 |
|---|---|---|---|
| mlp gate/up 17408x5120x2048 | 4.63 | **78.8** | 11.3x |
| mlp down 5120x17408x2048 | 4.10 | **89.1** | 12.7x |
| lm_head 248320x5120x2048 | 64.89 | 80.2 | 11.5x |
| attn q_proj 6144x5120x2048 | 1.53 | 84.0 | 12.0x |

I had been assuming ~43 TFLOPS, inferred backwards from llama.cpp's own prompt
eval time. This part does **~80 TFLOPS** of bf16, i.e. **11-13x** the current fp32
GEMM, not 6x. Applying that to the measured GEMM/attention split:

| | GEMM | attention | total | vs llama.cpp |
|---|---|---|---|---|
| 8K now | 49.6 s | 4.4 s | 54 s | 5.2x slower |
| **8K + tensor cores** | **4.4 s** | 4.4 s | **8.8 s** | **1.20x FASTER** |
| 32K now | 202.6 s | 65.2 s | 268 s | 6.0x slower |
| 32K + tensor cores | 18.0 s | 65.2 s | 83.2 s | 1.9x slower |
| 32K + tensor cores + 3x attention | 18.0 s | 21.7 s | **39.7 s** | **1.12x FASTER** |
| 32K + tensor cores + 6x attention | 18.0 s | 10.9 s | **28.9 s** | **1.54x FASTER** |

**So the tensor-core GEMM alone wins 8K cold TTFT outright, with no attention
work at all.** 32K then needs the GQA fusion, and it wins even at a pessimistic
3x rather than the hoped-for 6x.

That is a materially better position than the 43 TFLOPS assumption implied, and
it is the reason this benchmark exists as its own subcommand: the single most
consequential number in the whole plan turned out to be 2x off, and it cost one
20-line measurement to find out instead of a multi-round rewrite.

## Implemented and measured (rounds 22)

The dispatch is in: `Linear::forward_prefill` calls `forward_prefill_tensor_core`
by default (set `GB10_TC_GEMM=0` to fall back to the fp32 path). It was landed
opt-in and flipped to default-on once the suite cleared it: `generate` 16/16
against the oracle and `batch-parity` 16/16 over 16 sequences, **both with the
tensor-core path active** -- those are the two gates `loop/run_round.sh` runs. The three scratch buffers are per-call rather than a shared
persistent scratch, which relies on cudarc's caching allocator to make the repeat
allocations cheap; that kept the change local instead of threading a scratch
through 17 call sites.

**Correctness:**

    GB10_TC_GEMM=1 gb10-verify generate --n 16 --repeat 8 --oracle fixtures/oracle
    -> oracle agreement: 16/16 (100.0%), exact match

**Speed — per-chunk (`prefill-shape --limit 8225`):**

| chunk | start | fp32 GEMM | **tensor core** |
|---|---|---|---|
| 0 | 0 | 12.62 s | **4.72 s** |
| 1 | 2048 | 13.09 s | 5.02 s |
| 2 | 4096 | 13.80 s | 5.55 s |
| 3 | 6144 | 14.10 s | 5.97 s |
| 4 | 33 tokens | 0.48 s | 1.03 s |
| **total** | | **54.09 s** | **22.28 s (2.43x)** |

Fitting the same `t = G + k*(start + chunk/2)`: **G falls from 12.35 s to 4.51 s**,
a 2.74x cut in the constant term.

**Speed — end to end, through the server:**

| | fp32 GEMM | tensor core | |
|---|---|---|---|
| 8K cold TTFT | 54.88 s | **22.15 s** | **2.48x** |
| 8K warm TTFT | 0.03 s | 0.03 s | unchanged |
| 8K OTPS | 8.60 | 8.64-8.74 | unchanged |
| vs llama.cpp 10.58 s | 5.2x slower | **2.09x slower** | |

Warm TTFT and OTPS are unchanged exactly as intended -- the change touches only
the prefill GEMM, so the decode roofline is untouched.

### The measurement disagrees with the prediction, and that is the next lead

The plan predicted ~11x on the GEMM (7 -> 80 TFLOP/s) and an 8.8 s 8K cold TTFT.
It measured 2.74x and 22.15 s. The gap is real and worth naming rather than
rounding away:

The pipeline now does, per chunk, roughly **72 GB of pure zeroing** (`alloc_zeros`
is the only allocator cudarc exposes, and every buffer is fully overwritten right
after), plus ~44 GB of bf16 writes from the dequant, ~22 GB of NVFP4 reads, and
~44 GB of bf16 reads by the GEMM. That is ~180 GB per chunk against the ~18 GB the
fp32 GEMM needed, and at the measured ~200 GB/s it is ~0.9 s -- not enough on its
own to explain 4.51 s, so the zeroing and the four-kernel-per-matrix structure are
the first suspects but not yet proven ones.

The order to attack it, cheapest first:

1. **Drop the zeroing.** 72 GB per chunk of memset that is immediately written
   over. cudarc exposes no uninitialised allocator, so this needs a persistent
   scratch (which removes the zeroing *and* the allocation churn in one move).
2. **Hoist the scratch** into the 321 MB shared buffer computed above, which also
   removes per-call pool traffic.
3. **Fuse dequant into the GEMM** (or cache the dequantised weights) if 1 and 2
   are not enough -- but note the dequantised model is 44 GB and the NVFP4 one is
   22 GB, so caching it trades the decode roofline away and is likely wrong.

Even at 2.74x this is the largest single win of the session on the metric the
objective is about: 8K cold TTFT went from 5.2x slower than llama.cpp to 2.09x.

### The alloc_zeros hypothesis, measured (round 25)

The leading suspect for the gap between the predicted 11x and the measured 2.74x
was the per-call `alloc_zeros` in `forward_prefill_tensor_core`. `cublas-gemm` now
re-allocates the same three buffers per rep, exactly as the model does, and times
that separately from the GEMM:

| shape | GEMM | alloc_zeros only | overhead |
|---|---|---|---|
| mlp gate/up | 4.68 ms (78.0 TFLOP/s) | **2.44 ms** | **+52%** |
| mlp down | 4.17 ms (87.5 TFLOP/s) | 2.32 ms | +56% |
| lm_head | 60.57 ms (86.0 TFLOP/s) | 29.01 ms | +48% |
| attn q_proj | 1.50 ms (85.9 TFLOP/s) | 0.94 ms | +63% |

So the hypothesis is confirmed and it is much larger than my first estimate. I had
put the zeroing at ~67 GB per chunk, which at the nominal 228 GB/s read bandwidth
is ~0.33 s; the measured cost is a ~111 GB/s memset (i.e. the write path is about
half the read path, not equal to it) **plus** a fixed per-call allocation overhead
on top, for roughly 50% again on every GEMM.

Note this also sharpens the earlier note about `lm_head`: it is not on the prefill
path, so its 29 ms of allocation is not being paid per chunk -- it is in the table
only because it is the largest shape and therefore the clearest signal.

**What this does and does not explain.** At ~50% overhead on a 1.05 s GEMM budget,
this is ~0.5-0.6 s per chunk of the 4.44 s measured. It is real, it is now
quantified, and the fix is known (a persistent 321 MB scratch, which removes both
the zeroing and the allocation churn). But it is still only about an eighth of the
gap, so **a little over 2 s per chunk remains unexplained** and I am not going to
attribute it without evidence. The next measurement should be per-phase timing
inside `forward_prefill_tensor_core` (dequant / activation cast / GEMM / epilogue)
rather than more end-to-end fits, since four independent fits now agree with each
other and disagree with the phase model.

### Per-phase timing: the residual is the allocator, not the GEMM (round 26)

Four end-to-end fits agreed with each other and disagreed with the phase model, so
I stopped fitting the total and timed the phases directly. `gb10-bench tc-phase`,
MLP gate/up at t = 2048:

| phase | ms | share |
|---|---|---|
| `alloc_zeros (w,x,y)` | **4.458** | **39.0%** |
| `dequant_nvfp4_to_bf16` | 1.353 | 11.8% |
| `f32_to_bf16` | 0.291 | 2.5% |
| `cublas_gemm_bf16` | 4.761 | 41.6% |
| `bf16_to_f32_scaled` | 0.569 | 5.0% |
| **sum** | **11.433** | |

**The GEMM is only 41.6% of the pipeline, and the non-GEMM work is 1.40x the
GEMM.** The allocator alone is 39%.

This corrects round 25's estimate by a wide margin, and the earlier reasoning was
wrong for an instructive reason: I costed the zeroing as 270 MB of memset at the
nominal 228 GB/s read bandwidth and got ~1.2 ms. The measured 4.458 ms is
**~60 GB/s** -- so the memset is not the cost at all. The per-call allocation and
launch overhead dominates it, which is exactly what a bandwidth calculation cannot
see. The round-25 A/B (2.44 ms alloc against a 4.68 ms GEMM) was already pointing
here; splitting the phases shows the allocator is the single largest phase after
the GEMM itself.

**What that is worth.** If the allocator cost is removed (a persistent 321 MB
scratch, which removes the zeroing and the allocation churn together), the GEMM
pipeline keeps 6.975 / 11.433 = 61% of its time -- a **1.64x** cut on every
prefill GEMM. Against the measured 8K decomposition (22.15 s = ~4.4 s attention +
~17.75 s GEMM pipeline):

    4.4 + 17.75 * 0.61 = ~15.2 s

which would be **1.44x slower than llama.cpp's 10.58 s**, down from 2.09x. At 32K
the same scaling gives roughly 82 s against 44.55 s. That is the largest single
remaining lever, and unlike the previous two rounds it is now measured rather than
inferred -- one allocation change, no kernel work.

The residual after that would be the dequant (11.8%), which is the next candidate:
it is a pure streaming pass over the packed weights and should be closer to
bandwidth than 1.35 ms for 89 MB read + 178 MB written (197 GB/s, i.e. already
near the measured 228 GB/s peak, so it may genuinely have little left in it).
