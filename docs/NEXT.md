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

### Step 2 attempted and reverted (round 57)

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
