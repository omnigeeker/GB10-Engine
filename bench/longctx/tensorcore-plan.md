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
   staging code. Each takes the same packed weight/scales and writes bf16 to a
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

4. **Dispatch.** `Linear::forward_prefill` in `crates/gb10-model/src/weights.rs`
   — replace the GEMM branch (the `n >= 256 && t > 16` side of the existing
   crossover) with: dequant -> convert x -> `blas.gemm` -> (x stays fp32 for the
   rest of the layer). Keep `forward` (the NVFP4 GEMV) for `t <= 16` and for
   decode, so the OTPS win is untouched.

5. **Layout.** `C[t, n] = X[t, k] * W[n, k]^T`, both row-major:
   `cublasGemmEx(OP_T, OP_N, m = n_out, n = t, k = k_dim, A = W (lda = k_dim),
   B = x (ldb = k_dim), C = y (ldc = n_out))`, compute type `CUBLAS_COMPUTE_32F`,
   scale type `CUDA_R_32F`. Note `m` is the *weight* dimension and `n` is the
   token dimension — that is the whole trick of this layout, and getting it
   backwards is the likely first failure.

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
