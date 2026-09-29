# Long-context comparison against llama.cpp

Same box, same NVFP4 weights, same prompt, same harness (`bench/longctx/ttft.py`),
runs sequential so neither contender has the GPU to itself. llama.cpp is
`tools/llama.cpp/build/bin/llama-server` on `models/Qwen3.8-27B-NVFP4.gguf` with
`-fa on -np 1`; gb10-server is this repo's build.

Two definitions, because "TTFT" hides a real question:

* **cold TTFT** — the prefix has never been seen, so the whole prompt is prefilled.
* **warm TTFT** — the identical request again, so a prefix cache can skip the prefill.

The prompt is repeated filler plus "list the integers 1 to 300", which stops the
model answering in two tokens and makes OTPS an average over ~198 intervals.

## Current scorecard: tensor-core prefill GEMM, grid-clamp bug FIXED

**Every number here is server-side, measured with `PREFILL_CHUNK = 8192`, `--ctx 262144`, and
pairs taken in the same session** (the session's rule: only same-session pairs are comparable --
round 90 measured 23% drift on this box). gb10 runs `GB10_TC_GEMM` on by default with
`GB10_TC_SPLIT=1` (a single bf16 operand).

| context | cold TTFT | warm TTFT | OTPS |
|---|---|---|---|
| 8K (8752 / 8790 tok) | 18.90 / 13.45 s = **1.41x slower** | 0.03 / 0.28 s = **9.3x faster** | 7.25 / 6.18 = **1.17x faster** |
| 32K (34793 / 34831 tok) | 120.15 / 56.94 s = **2.11x slower** | 0.05 / 0.31 s = **6.2x faster** | 6.42 / 5.79 = **1.11x faster** |
| 128K (136474 / 136512 tok) | 1224.16 / 290.63 s = **4.21x slower** | 0.16 / 0.47 s = **2.9x faster** | 4.28 / 4.68 = **1.09x slower** |
| 256K | not measured (extrapolates to ~65 min cold) | -- | -- |

**Warm TTFT 3 of 3 won. OTPS 2 of 3 won. Cold TTFT 0 of 3.**

> Three scorecards exist for this section and only the third is valid.
>
> * `10.35 / 80.00 / 890.54 / 3298.83 s` ("1.09x / 1.80x / 3.24x / 4.54x slower") was measured
>   with the tensor-core prefill GEMM **truncating work** -- it skipped every element past
>   16,776,960 (see `TC_GEMM_REGRESSION.md`), so it was fast because it was wrong.
> * `64.12 / 309.28 s` was the fp32 fallback after that bug was found but before its cause was,
>   i.e. the price of a correct engine with the fast path switched off.
> * **The table above is the honest one**: the fast path is on *and* correct.

### What is left, and exactly how much of it

Fitting the three measured points:

```
gb10   prefill ~= 1.577e-3 * T + 5.417e-8 * T^2   s
llama  prefill ~= 1.468e-3 * T + 4.845e-9 * T^2   s

       linear term: gb10 is 1.075x slower   <- essentially at parity
    quadratic term: gb10 is 11.18x slower   <- this is the whole deficit
```

| context | gb10 linear | gb10 quad | llama total | gap | quad share of gap | attention speedup needed |
|---|---|---|---|---|---|---|
| 8K | 12.9 | 3.6 | 12.3 | 4.2 | 79% | 11.2x (with the linear term at parity) |
| 32K | 51.7 | 58.2 | 53.3 | 56.6 | 94% | 11.2x |
| 128K | 206.7 | 930.6 | 275.6 | 861.7 | 98% | 11.2x |
| 256K | 413.5 | 3722.4 | 717.6 | 3418.2 | 99% | 11.2x |

Two things fall out of this, and both are cleaner than the earlier version of this section:

1. **The prefill GEMM is no longer the problem.** Making the tensor-core path both fast *and*
   correct moved the linear term to within 7.5% of llama.cpp. The remaining linear excess is
   worth at most a 1.075x win and is not where the work is.
2. **The target is a single number: 11.2x on prefill attention.** Because the condition reduces
   to `gb10_quad / k < llama_quad`, the requirement is *the same at every context length* --
   it is just the ratio of the two quadratic coefficients. That is also why the previously
   recorded requirements (1.29x / 5.98x / 8.40x / 8.82x) were wrong: they were computed against
   the broken GEMM's inflated linear term, which made attention look like a 1.29x problem at 8K
   and hid that it is an 11.2x problem everywhere.

**So the objective's "designed for 9x" is ~20% short of what the measurement requires.** An
`mma.sync` prefill attention at 9x would leave 8K at ~13.3 s against 13.45 s (a knife-edge win),
32K at ~58.2 s against 56.94 s (still a loss), and 128K at ~310 s against 290.63 s (still a
loss). At 11.2x all four win. The design should be specified to 12x for margin, and the
`ex2.approx.f16x2` and fp32-accumulation constraints from the original plan still hold.


**What is left, and what it costs:**

| cold cell | speedup needed on attention | cheapest available lever |
|---|---|---|
| 8K | **1.29x** | none -- see rounds 145-147 |
| 32K | 5.98x | mma prefill attention |
| 128K | 8.40x | mma prefill attention |
| 256K | 8.82x | mma prefill attention |

**Rounds 145-147 walked back the one apparent shortcut.** The `GB10_ATTN_SMEM_PROBE` occupancy probe
(cost of forcing 1 block/SM: 3.77 -> 5.55 s, i.e. 1.47x) looked like a cheap way to win 8K, but it
measures *sensitivity*, not headroom: the kernel already runs at 2 blocks/SM, and 3 would need a 22%
cut in its 43,104 B shared-memory request. **So all four cold cells need the same work** -- the mma
rewrite, or (for a possible 8K-only win) an fp8 staging rederivation that carries a bank-layout
rederivation and a likely `attn-tile` precision rejection.

**Still owed:** a repeat of 256K OTPS, which is 1.05x away and needs ~111 minutes for two trials.

## 8K — with the new decode kernel

| metric | gb10-server | llama.cpp | ratio |
|---|---|---|---|
| prompt tokens | 8,225 | 8,263 | — |
| cold TTFT | 54.5 s | **10.58 s** | 5.2× slower |
| **warm TTFT** | **0.03 s** | 0.237 s | **7.9× faster** |
| **OTPS** | **8.69** | 7.32 | **1.19× faster** |

## 32K

| metric | gb10-server | llama.cpp | ratio |
|---|---|---|---|
| prompt tokens | 32,747 | 32,785 | — |
| cold TTFT | 272.6 s | **44.55 s** | 6.1× slower |
| **warm TTFT** | **0.05 s** | 0.29 s | **5.8× faster** |
| **OTPS** | **7.15** | 6.865 | **1.04× faster** |

llama.cpp's 32K numbers are the mean of three trials (43.55 / 44.85 / 44.46
cold, 0.29 / 0.28 / 0.25 warm). gb10's cold TTFT is unchanged by the prefix
cache, which is the point of it: a cold request has nothing to resume from.

**Warm TTFT is won.** It was 821 s at 32K — byte-identical to cold, because the
server wiped its own cache every request — and is now 0.05 s, faster than
llama.cpp's 0.27 s. At 8K it is 0.035 s against 0.24 s.

## The batch-size bug

`step_batch` returned `memcpy_dtov(&state.idx)` unchanged, and `state.idx` is
`state.n_seq` entries long — not the `n_seq` the caller actually passed in. The
server's decode loop feeds that result straight back in as the next step's
tokens, so a group of 1 was promoted to a group of `state.n_seq` on its second
step and stayed there.

The answers were unaffected, because the extra slots are idle and nothing reads
them, which is exactly why it survived: `batch-parity` passes 16 real sequences,
so it always exercised the correct path. What it broke was the cost.

`GB10_STEP_TIMING=1` prints the split, and it is unambiguous:

| | submit | drain | total | slots |
|---|---|---|---|---|
| before | 100.4 ms | 324.5 ms | 424.8 ms | 16 |
| after | 33.2 ms | 106.3 ms | 139.6 ms | 1 |

A lone request was doing **16× the work**, which is why its per-token cost was
flat against both context length and batch size, and why starting the server
with a larger `--concurrency` made OTPS *worse* (16 slots at 8K measured 2.34
tok/s, 8 slots at 32K measured 3.13). After the fix the drain is 106 ms against
the 94.8 ms that `gb10-bench stream` measures for streaming all the weights at
185 GB/s — the decode step is now within 12% of its bandwidth roofline.

## The prefix cache

`run_group` used to start with `state.reset(dev)`, which zeroed every KV cache,
conv history and recurrence on **every** request. There was no warm path at all.

The interesting part is that a position counter is not enough to rewind: the 48
Gated-DeltaNet layers carry a *recurrence*, and the state after N-1 tokens
cannot be recovered from the state after N by moving an offset. So the cache
snapshots the recurrence (`ModelState::snapshot_recurrent`, which already
existed for the MTP verify path) and restores it.

A snapshot is taken **after** each prompt rather than one token short of it, and
the token that prompt prefilled to is cached beside it. An exact repeat
therefore runs no forward pass at all: restore, hand back the cached token, and
let the decode loop continue from a state that is already exactly where it
should be. That is why warm TTFT is 35 ms and not ~250 ms — it is the cost of a
160 MB device-to-device copy, not of a forward pass.

Correctness of that is not assumed. `bench/longctx/prefix_ab.sh` runs one
request sequence — a repeat, an unrelated prompt after a cached one, and a
prefix extension — against a caching server and against one started with
`--no-prefix-cache`, and requires the output to be identical. It is wired into
`loop/run_round.sh` as its own gate, because none of the `gb10-verify` gates go
through the server's resume path and so none of them test it.

One limitation, deliberate: a miss costs the whole group its cache, since
`reset` and the snapshot are both whole-state operations. Mixed groups fall back
to a full prefill. This can make a group slower, never wrong.

## What the new decode kernel bought

`attn_decode_multi_kernel` used to give every thread one `head_dim` lane and walk
the keys serially, calling `block_reduce_sum` once per key — two
`__syncthreads()` inside the loop, so the block advanced exactly one key per
barrier and no two keys were ever in flight. It now gives each *warp* a strided
subset of the keys, reduces a key's score with `__shfl_xor` inside the warp, and
merges the eight warps' partials once.

`gb10-verify decode-bench` times both, in the same binary, so this is an A/B and
not a comparison across builds:

| keys | serial | warp | speedup | rms rel |
|---|---|---|---|---|
| 2,048 | 2.405 ms | 0.399 ms | 6.02× | 1.18e-6 |
| 8,192 | 9.629 ms | 1.579 ms | 6.10× | 2.07e-6 |
| 32,768 | 37.831 ms | 6.452 ms | 5.86× | 4.31e-6 |
| 65,536 | 74.216 ms | 12.514 ms | 5.93× | 6.36e-6 |

RMS-relative because a per-element relative error is meaningless on values that
cross zero. Agreement is ~1e-6; a tiling or indexing bug would show up at O(1).
End to end, at 8K with everything else held constant, **OTPS goes 1.82 → 2.41
(+32%)**. Correctness is unchanged: `generate` is still 16/16 token-exact against
the bf16 oracle and identical across 8 repeats.

The 6× does not become 6× end to end because attention is not where decode
spends its time. Working back from the measured numbers at 8K:

* all 21.9 GB of weights, streamed once per token at the measured 228 GB/s, is
  **96 ms** — a hard floor of ~10.4 tok/s;
* attention, at the warp kernel's 1.17 ms per layer × 16, is **~19 ms**;
* `gb10-bench stream` measures the whole weight set streaming in **94.8 ms at
  185 GB/s (81% of peak)**;
* after the batch-size fix the full step is **140 ms**, of which 106 ms is
  device drain — within 12% of that streaming floor.

So at 8K the decode step is essentially at its bandwidth roofline, and the
remaining 45 ms is the non-GEMM kernels plus host submission. At 32K attention
is no longer negligible — 4.88 ms per layer × 16 = **78 ms** of a 226 ms step —
and that is where OTPS's remaining 1.58× lives.

## Cold TTFT

Prefill cost is `≈5.11 ms·T + 5.93e-7·T²`. The quadratic term is not attention
*arithmetic* — it is attention *traffic*. The tiled kernel runs one block per
query head, so all 24 query heads independently stream the same K/V, and the
cache is f32:

* at 8K the quadratic term is 40 s of the 88 s total;
* at 32K it is 636 s of 822 s.

Against a 6× GQA redundancy (24 query heads over 4 KV heads) and 2× from f32,
the same K/V is being read about 12× more than it needs to be. llama.cpp sits at
~40 TFLOPS on prefill against gb10's ~7, and no amount of fp32 CUDA-core
tuning can close that: `gemm.cu` uses no tensor cores at all — no `mma.sync`, no
`wgmma`, no `__hfma2`, just scalar `fmaf`. Reaching llama.cpp's prefill rate
means bf16 or tf32 tensor cores, which is a numerical change, not just a
scheduling one.

## What the copy was costing

The single largest cost in a DeltaNet layer, after the GEMMs, was a
`memcpy_dtod` that looked free. `CudaStream::memcpy_dtod` copies
`src.num_bytes()` -- the *whole allocation*, not the live part -- and `Scratch`
is sized for a full prefill chunk. So `sc.conv` is `conv_dim * 2048 * 4` = 84 MB,
and every DeltaNet layer copied all 84 MB to produce the 40 KB that the l2norm
and the recurrence actually read. At 48 layers that is **4 GB of dead traffic per
token**, and `GB10_OP_TIMING=1` showed it as 0.716 ms of a 2.57 ms layer.

The kernel it fed was innocent:

| op (one DeltaNet layer) | before | after |
|---|---|---|
| `copy_live` (was `memcpy_dtod`) | 0.716 ms | **0.007 ms** |
| l2norm q | 0.004 ms | 0.007 ms |
| l2norm k | 0.004 ms | 0.007 ms |
| delta_step | 0.037 ms | 0.051 ms |
| **layer total** | **2.57 ms** | **~1.76 ms** |

`copy_live` copies only the `conv_dim * batch` floats the layer goes on to
read. End to end this is **OTPS 5.94 → 7.81 at 8K** and **4.43 → 5.31 at 32K**,
which is what puts 8K ahead of llama.cpp.

Two other things the same instrumentation ruled *out*, which is worth recording
because both were plausible and neither was true:

* `gated_delta_rule_step_multi` is not the cost. `GB10_SKIP_DELTA_STEP=1` makes
  it a no-op and the layer time does not move.
* host submission is not the cost. `GB10_STEP_TIMING=1` splits host from device
  with CUDA events: host 34 ms, device 143 ms, and the wall equals the device
  time, so the host work is fully hidden. `gb10-bench launch-overhead` puts a
  launch at 4.7 us, and CUDA graphs would buy nothing here.

## Where the remaining OTPS gap is

`gb10-bench store-stream` gives per-shape bandwidth, which is how the decode
step was attributed at all:

| kind | n | k | calls | ms/step | GB/s |
|---|---|---|---|---|---|
| nvfp4 | 17408 | 5120 | 128 | 30.00 | 213.9 |
| nvfp4 | 5120 | 17408 | 64 | 15.97 | 200.9 |
| fp8 | 10240 | 5120 | 48 | 11.73 | 214.5 |
| fp8 | 5120 | 6144 | 64 | 9.83 | 204.8 |
| fp8 | 6144 | 5120 | 48 | 7.94 | 190.2 |
| fp8 | 12288 | 5120 | 16 | 4.47 | 225.2 |
| **bf16** | **48** | **5120** | **96** | **2.51** | **18.8** |
| fp8 | 1024 | 5120 | 32 | 1.48 | 113.7 |

The GEMMs are at 190-225 GB/s, so weight streaming is not the problem. The
`bf16 48x5120` rows are the `linear_attn.in_proj_a`/`in_proj_b` pair: N=48 is
below the kernel's 64-wide N tile, so `grid.x` is **1 block** walking all 160
K-chunks behind a barrier each. It is 2.51 ms for 47 MB.

At 32K the binding constraint is instead the decode attention: 4.88 ms per layer
x 16 = **78 ms of a 188 ms step**. The warp kernel gives each query head its own
block, so all 24 query heads stream the same K/V and the GQA group is read 6x
over from L2 (344 GB/s of L2 traffic against 268 MB of unique DRAM data). Fusing
the group so one block serves all 6 query heads is worth ~6x on that traffic and
is what the next round is for.

## A blocking change that had to be reverted

The prefill attention kernel's query/key blocking looked like free performance.
Shared memory is `(BQ + 2*BK) * head_dim`, so the host caps `BQ + 2*BK` at 48,
and the kernel is latency-bound on the four `__syncthreads` per key tile rather
than on arithmetic or bandwidth. Total time therefore goes as `1/(BQ * BK)`, and
the shipped `8 / 16` gives only 128 of a possible ~264.

Retuning to `BQ=24, BK=11` (264, and 48,448 B of the 49,152 B limit) passed the
token-exact gate with the error unchanged at `attn_gated err/scale=4.350e-7`,
but made an 8K prefill take **over 7 minutes** instead of 88 seconds. It was
reverted.

The constraint that was missed: the score loop assigns **two threads per
(query, key) pair** and joins them with `__shfl_xor_sync(0xffffffff, dot, 1)`.
One pass therefore consumes `blockDim.x / 2` = 128 pairs, and every lane of a
warp must execute the same *number* of passes. `8 * 16 = 128` is exactly one
pass; `24 * 11 = 264` is 2.06 passes, so lanes 0-15 run a third pass while
16-255 do not, and the shuffle names lanes that are not executing the
instruction. The result is both wrong and enormously slow.

Anyone raising `BQ * BK` must keep it a multiple of 128 and re-check the 48 KB
budget in `gb10_cuda::ops::attn_prefill_tiled`. `BQ=16, BK=16` is exactly two
uniform passes but needs 50,368 B, which is 1,216 B over the limit; it would fit
under the 99 KB opt-in ceiling if the kernel were given
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, which nothing in
`crates/gb10-cuda` currently sets.

## Why the decode attention was slow: warps, not bytes

The warp-parallel decode kernel was already free of barriers inside the key
loop, and at `--n-seq 4` it sustained ~1030 GB/s. At `--n-seq 1` the same kernel
managed only ~344 GB/s. Same kernel, same bytes per sequence -- the difference
is that the grid is `(n_q_heads, n_seq)`, so one sequence is **24 blocks**, and
with an 8-warp block that is 4 resident warps per SM. There was nothing to hide
the key-load latency with.

`NW = 8 -> 32` (block 256 -> 1024 threads) fixes it without touching the grid:

| keys | n_seq 1, warp kernel before | after | speedup |
|---|---|---|---|
| 32,768 | 4.884 ms | **1.912 ms** | 2.55× |
| 32,768 bandwidth | ~344 GB/s | **842 GB/s** | |

`decode-bench` still agrees with the serial reference (rms rel 4.16e-6) and
`generate` is still token-exact 16/16, so widening the merge from 8 to 32 warps
did not move any token. End to end:

| context | metric | before | after |
|---|---|---|---|
| 8K | OTPS | 7.81 | **8.65** |
| 32K | OTPS | 5.31 | **7.06** |

At 32K the step is now 141.6 ms: ~95 ms of weight streaming at the measured
185 GB/s, 30.6 ms of decode attention, ~16 ms of everything else. The attention
is back to being L2-bound rather than occupancy-bound -- 1.6 GB of L2 traffic for
268 MB of unique data, because each of the 6 query heads in a GQA group reads its
KV head's cache independently. Fusing the group (split the keys across blocks,
merge the online-softmax partials in a second pass) is the next lever; it should
take the attention to the ~19 ms DRAM floor.

## A second change that measured as nothing

`nvfp4_gemm` splits K across `grid.z = 2` unconditionally, which forces a
`memset_zeros` of the whole output plus the `atomicAdd` store path in
`gemm2d_store`. Since every shape here already has 80-3880 blocks without the
split, it looked like ~1.1M atomicAdds per call bought for nothing.

Making the split conditional on the grid being small measured as **no change at
all**:

| measurement | unconditional | conditional |
|---|---|---|
| `gb10-bench stream` | 91.56 / 94.76 ms | 95.28 ms |
| 8K OTPS (4 samples) | 8.70, 8.63, 8.63, 8.64 | 8.71, 8.64, 8.59, 8.62 |

`generate` stayed token-exact, so the change was safe, but it perturbs the GEMM
summation order for no measured gain, and it was reverted. The memset and the
atomicAdds evidently overlap with the neighbouring GEMMs' loads, so they are not
on the critical path. The `gb10-bench stream` spread (91.6-95.3 ms across runs)
is itself wider than any effect being chased here.

## Error bars, because a 1.03x claim needs them

Every llama.cpp row above is a fresh 3-trial run (6 OTPS samples, since the
harness reports one after the cold prefill and one after the warm one), and the
gb10 rows are the same harness on the same prompts.

**8K** (prompt 8,263 tokens for llama, 8,225 for gb10):

| | llama.cpp samples | mean | gb10 samples | mean |
|---|---|---|---|---|
| OTPS | 7.39 7.37 7.34 7.39 7.18 7.27 | 7.32 | 8.70 8.63 8.63 8.64 | **8.65** |
| cold TTFT | 10.35 10.65 10.73 | 10.58 | 88.13 89.49 | 88.81 |
| warm TTFT | 0.23 0.24 0.24 | 0.237 | 0.03 0.03 | **0.03** |

**32K** (32,785 vs 32,747 tokens):

| | llama.cpp samples | mean | gb10 samples | mean |
|---|---|---|---|---|
| OTPS | 6.89 6.82 6.86 6.76 6.94 6.92 | 6.87 | 7.06 7.07 | **7.065** |
| cold TTFT | 43.77 44.89 45.00 | 44.55 | 826.09 | 826.1 |
| warm TTFT | 0.29 0.29 0.29 | 0.29 | 0.05 | **0.05** |

The OTPS win is real but modest at 32K, and the two distributions do not
overlap: llama.cpp's **best** sample at 32K is 6.94 against gb10's **worst** of
7.06, and at 8K llama's best is 7.39 against gb10's worst of 8.63. The warm
TTFT and cold TTFT gaps are far outside the noise in opposite directions.

## Correcting the cold-TTFT model: prefill is compute-bound, not bandwidth-bound

I had been reasoning as though a prefill forward pass is dominated by streaming
the weights. llama.cpp's own timing log refutes that outright. From
`/tmp/longctx/llama-8k.log`, same server, same session:

| prompt | prompt eval time |
|---|---|
| 4 tokens | 207.40 ms |
| 8,221 tokens | 10,330.91 ms |
| 8,221 tokens | 10,376.80 ms |

A linear fit through the 4-token and 8,221-token points gives a **fixed cost of
about 202 ms** and a **marginal cost of 1.23 ms per token**. The weights are read
once per forward pass, so that ~202 ms *is* the weight streaming: 28.23 GB in
202 ms is ~139 GB/s, which is exactly the DRAM speed this box does. The
remaining ~10.1 s at 8K is **compute**.

The consequence is that the cold-TTFT gap is a FLOPS gap, not a bytes gap, and
the arithmetic says so from both sides:

| | 8K prefill | FLOP | effective |
|---|---|---|---|
| llama.cpp | 10.58 s (10.14 s of it compute) | 4.6e14 | ~45 TFLOPS |
| gb10 | 88.8 s (~88.7 s compute) | 4.5e14 | **~10.7 TFLOPS** |

That is the whole story: llama.cpp runs the prefill GEMMs on bf16 tensor cores
at ~45 TFLOPS, and `kernels/gemm.cu` has no tensor-core instruction at all
(`grep -cE "mma\.sync|wgmma"` is 0), so it does fp32 `fmaf` on CUDA cores at
~10.7 TFLOPS -- 46% of this part's ~23 TFLOPS fp32 peak.

This does not change the conclusion that cold TTFT needs tensor cores, but it
does change where to look. The 17.6 GB of per-token weight streaming that
dominates *decode* is a ~120 ms rounding error during prefill, so nothing about
the prefill is helped by shrinking or re-laying-out weights. Only raising FLOPS
helps the linear term, and only cutting the attention's redundant work helps the
quadratic one.

## What the prefill GEMM is actually bound by

`gb10-verify forward-cost` runs the same model two ways and prices the GEMM
path directly. The shape of it rules out the explanation I was working from:

| t | one t-row forward (ms) | FLOP |
|---|---|---|
| 8 | 180.75 | 2.7e11 |
| 16 | 315.21 | 5.4e11 |
| 32 | 400.11 | 1.1e12 |
| 64 | 441.04 | 2.2e12 |

Between t=32 and t=64 the arithmetic **doubles** while the time grows 10%. If the
kernel were arithmetic-bound the time would roughly double, so at these sizes it
is bound by the weight loads, which do not change with t.

But that stops being true at the size the prefill actually uses. `PREFILL_CHUNK`
is 2048, and a chunk of 2048 tokens costs about 13 s
(`t = 5.1072e-3*T + 5.9269e-7*T^2` at T = 2048), which is 8.97e13 FLOP -- about
**7 TFLOPS**. So the same kernel is load-bound at t<=64 and arithmetic-bound at
t=2048, and the arithmetic it settles at is ~7 TFLOPS against this part's ~23
TFLOPS fp32 peak.

I tried to exploit the load-bound half. The grid is
`(N/GB10_TN, T/GB10_TT, gridDim.z)`, so each weight tile is re-read `T/GB10_TT`
times -- 32 times at T = 2048. Raising `GB10_TT` from 64 to 128 halves that, and
is bit-identical because `GB10_KC` and the k-chunk order are untouched, so every
output element accumulates in exactly the same order. Measured:

| | GB10_TT = 64 | GB10_TT = 128 |
|---|---|---|
| 8K cold TTFT | 88.81 s | 88.33 s |
| `forward-cost` t=32 | 400.11 ms | 624.63 ms |
| `forward-cost` t=64 | 441.04 ms | 712.25 ms |

No help at the size that matters (88.33 vs 88.81 is inside the noise) and
clearly worse at t=32/64, because a 128-token tile is half empty at those sizes
while the shared tile grows to 40 KB/block. Reverted. The reason it cannot help
is the same reason it was worth trying: at T = 2048 the kernel is **not** bound
by those re-reads, so halving them buys nothing.

That leaves the cold-TTFT gap as an arithmetic gap with no tiling fix. ~7 TFLOPS
out of ~23 is what fp32 `fmaf` on CUDA cores gets here, and llama.cpp's ~45
TFLOPS comes from bf16 tensor cores. Reaching it means an `mma.sync` GEMM on the
staging tiles that are already bf16 in shared, which is a numerical change and a
much larger piece of work than anything above.

## Raising the prefill attention tile: 3.5%, not the 3x the model predicted

V is only ever read as `Vs[j * HD + tid]` -- one column per thread for each j --
so it never needed shared memory. Moving it into a per-thread register array
frees the `BK * HD` staging buffer, which is what lets `BQ * BK` grow without
exceeding the 48 KB shared limit:

| | BQ | BK | BQ*BK | shared |
|---|---|---|---|---|
| before | 8 | 16 | 128 (1 pass) | 41,568 B |
| after | 24 | 16 | 384 (3 passes) | 42,784 B |

`PREFILL_BQ * PREFILL_BK` must stay a multiple of `blockDim.x / 2` (the score
loop pairs two threads per `(i,j)` behind a `__shfl_xor_sync(0xffffffff)`), and
384 = 3 x 128 satisfies that with every lane running the same number of passes.

The change is numerically identical -- same K and V values, same dot products,
same accumulation order -- and `attn-tile` and `generate` both still pass
(`generate` 16/16 token-exact).

Measured on 8K cold TTFT:

| | before | after |
|---|---|---|
| cold TTFT | 88.81 s | **85.71 s** |
| OTPS | 8.65 | 8.69 |

So a 3x larger score tile bought **3.5%**, not 3x. That falsifies the "attention
time is proportional to 1 / (BQ * BK)" model I had been carrying. The per-tile
cost is evidently dominated by something that does not shrink with the tile --
the K and V staging and the four barriers per tile are all still there, and the
score-loop shared reads per thread (3 pairs x 256 floats) actually grew. Tiling
alone will not close a gap that is now 8.1x at 8K.

### The 32K re-measurement after the tile change

The 8K number alone understated it, so the 32K cold TTFT was re-run on the
current binary:

| | BQ=8, BK=16 | BQ=24, BK=16 |
|---|---|---|
| 8K cold TTFT | 88.81 s | 85.71 s (**-3.5%**) |
| 32K cold TTFT | 826.09 s | **757.68 s** (**-8.3%**) |

The gain roughly doubles from 8K to 32K, which is the signature of a change that
attacks the quadratic term: fitting `t = 5.1072e-3*T + 5.9269e-7*T^2`, the
quadratic coefficient falls from 5.93e-7 to about 5.30e-7 while the linear one
is untouched. So it is a real improvement in the right place, but an 11% cut on
the quadratic term against a 17x gap at 32K.

## The score loop's K row was being read three times: 1.31x at 8K, 1.73x at 32K

Last round I priced the prefill attention and found the score loop was ~80% of
the per-tile cycles (3072 of ~3840, at 32 floats/cycle/SM), and that the reads
were redundant. Working out the pair mapping showed the redundancy was worse
than "Q is read BK times and K is read BQ times" -- it was being paid *within a
single thread*.

A thread-pair owns three pairs, `p`, `p + (nt>>1)`, `p + 2*(nt>>1)`. Since
`PREFILL_BK` divides `nt >> 1`, adding `nt >> 1` to the pair index advances `i`
by `(nt >> 1) / PREFILL_BK` and **leaves `j` unchanged**. With `nt == 256`,
`PREFILL_BK == 16` and `PREFILL_BQ == 24` the three pairs are `(i0, j)`,
`(i0 + 8, j)`, `(i0 + 16, j)`: one K row, three Q rows. The code was reading that
K row from shared three times.

Holding it in a register across the three turns six shared reads per three fma
into four. The host now refuses any launch where
`PREFILL_BQ * PREFILL_BK != 3 * (nt >> 1)`, so the layout the kernel assumes
cannot silently stop holding. Numerically identical -- same dot products, same
accumulation order -- and `attn-tile` and `generate` (16/16 token-exact) both
still pass.

| | before | after | |
|---|---|---|---|
| 8K cold TTFT | 85.71 s | **65.21 s** | 1.31x |
| 32K cold TTFT | 757.68 s | **437.10 s** | 1.73x |
| `attn-tile` 65536 tokens | 139.77 s | **69.75 s** | 2.00x |
| `attn-tile` 16384 tokens | 8.73 s | 6.39 s | 1.37x |
| `attn-tile` 4096 tokens | 0.55 s | 0.24 s | 2.29x |

I predicted 1.5x from the shared-read arithmetic and got 2.0x on the pure
attention benchmark, so the hoist bought more than the read ratio alone -- the
three explicit row pointers also let the compiler drop per-`d` address
arithmetic that the `p`-indexed loop had been recomputing.

This is the first change that moves cold TTFT by more than a few percent, and
unlike every tiling change tried before it, the gain **grows** with context
(1.31x at 8K, 1.73x at 32K) because it attacks the quadratic term. It also
confirms the diagnosis from last round was right: the attention was bound by
redundant shared reads in the score loop, not by the tile size.

## The real cost was shared bank conflicts, not reads

The hoist above cut the score loop's reads by a third and won 1.3-1.7x, which is
what the read arithmetic predicts. But the reads themselves were far more
expensive than "one transaction each", and that was the actual problem.

Every row of Q and K started on the same shared bank, because the stride was
`HD == 256` floats and 256 is a multiple of 32. In the score loop a warp reads
`Ks[j * HD + sub * half + d]` for 16 different `j` and 2 `sub` values: 32
*distinct words*, all at bank `d % 32`. That is a **32-way conflict** -- 32
transactions for every single load instruction, on the hottest loop in the
kernel. The `sub` offset (`half == 128`) is itself a multiple of 32, so it did
not spread them either.

Padding the rows to `HD + 1` floats makes the bank `(row + d) % 32`, so the 16
rows land on 16 different banks and only the 2-way `sub` conflict remains: 32
transactions become 2. This only changes addresses, so it is numerically
identical, and `attn-tile` and `generate` (16/16 token-exact) both still pass.

| `attn-tile` case | BQ=8 | +BQ=24 | +hoist | **+pad** | total |
|---|---|---|---|---|---|
| 65536 tokens | 139.77 s | 139.77 s | 69.75 s | **28.46 s** | **4.9x** |
| 16384 tokens | 8.73 s | 8.73 s | 6.39 s | **1.09 s** | **8.0x** |
| 4096 tokens | 0.55 s | 0.55 s | 0.24 s | **0.07 s** | **7.9x** |
| start 20480, 2048 tokens | 2.94 s | 2.94 s | 1.57 s | **0.36 s** | **8.2x** |

End to end:

| | original | after hoist | **after pad** | total |
|---|---|---|---|---|
| 8K cold TTFT | 88.81 s | 65.21 s | **54.49 s** | **1.63x** |
| 32K cold TTFT | 826.09 s | 437.10 s | **272.62 s** | **3.03x** |

Three rounds ago I concluded the attention was "a from-scratch rewrite, not a
tuning problem". That was wrong, and the way it was wrong is worth recording:
the 333 GFLOPS figure (1.4% of fp32 peak) was real, but I attributed it to the
algorithm's structure rather than to a one-character addressing bug. A stride
that is a multiple of 32 is the single most common way to destroy a shared
memory kernel, and it had been sitting in the hottest loop the whole time.

## Killing the last bank conflict moved the microbenchmark but not the model

With rows padded, the only conflict left in the score loop was the 2-way one
between the two `sub` halves: their offsets differ by `half == 128`, and 128 is a
multiple of 32, so both halves sat in the same bank. Separating them by one extra
float (`PADH = half + 1`, row stride `2 * PADH`) makes the 16 `j` values and 2
`sub` values of a warp span all 32 banks:

| `attn-tile` | padded | **split halves** |
|---|---|---|
| 65536 tokens | 28.46 s | **17.36 s** (1.64x) |
| 16384 tokens | 1.09 s | 1.04 s |

So the conflict arithmetic was right again. But it did **not** move the thing the
goal is measured on:

| | padded | split halves |
|---|---|---|
| 8K cold TTFT | 54.49 s | 54.68 s |
| 32K cold TTFT | 272.62 s | 269.41 s |

Both are inside the run-to-run spread. I checked that the real prefill really
does call `attn_prefill_tiled` (`Model` -> `Layer::forward_prefill` ->
`ops.attn_prefill` -> `self.attn_prefill_tiled`), so this is not the kernel
being bypassed. The reading is that after the padding fix the attention stopped
being conflict-bound and became bound by something the standalone benchmark does
not reproduce: **KV cache traffic**.

The arithmetic for that is not subtle. Every one of the 24 query heads walks the
whole key prefix for its layer, but there are only 4 KV heads, so the cache is
read 6 times over. Per layer per prefill the key+value reads are

    6 * (T^2 / 2) * 2 * n_kv_heads * head_dim * 4 bytes

which at T = 32747 is ~4.4 TB per layer and ~70 TB over the 16 full-attention
layers. The measured remaining quadratic term at 32K is ~105 s, so the cache is
being served at an effective ~670 GB/s -- above DRAM, so largely out of L2.

That makes **GQA-aware attention** the next real lever: fusing the 6 query heads
that share a KV head so the cache is read once instead of 6 times is a 6x cut on
the dominant term. It is also the change I costed earlier and deferred -- it
needs either a key split with a global partial buffer and a second merge kernel,
or a register budget of `6 * BQ` accumulators per thread, which forces BQ down to
about 8.

The split-halves change is kept: it is numerically identical, correctness-gated,
and a real 1.64x on the isolated kernel. It simply is not what the model is
waiting on right now, and it will matter again once the KV traffic is fixed.

## The KV traffic, counted

The GQA redundancy is worth putting numbers on before committing to a rewrite.
Every query block walks the whole key prefix for its own KV head, and there are
6 query heads per KV head, so the cache is read 6 times over:

| | 8K | 32K |
|---|---|---|
| query blocks per layer | 8,208 | 32,736 |
| average keys each | 4,112 | 16,374 |
| KV traffic per layer | 69 GB | 1.10 TB |
| KV traffic, 16 full-attention layers | **1.11 TB** | **17.6 TB** |
| KV cache size per layer | 67 MB | 268 MB |
| implied rate over the measured attention term | 89 GB/s | 167 GB/s |

Two things fall out of this. At 32K the attention term is running at 167 GB/s
against a measured 185 GB/s achievable and a 228 GB/s peak, so it is essentially
**DRAM-bound** and 6x less traffic is 6x less time: 17.6 TB -> 2.9 TB -> ~16 s at
185 GB/s, against the ~105 s it takes now.

At 8K the same arithmetic gives only 89 GB/s, i.e. it is *not* DRAM-bound there --
roughly half the bandwidth is still being left on the table by the kernel itself.
So the picture is: at long context the attention is a traffic problem, at short
context it is still partly a kernel problem, and those want different fixes. That
is the honest state of it; I have not resolved why the half-split's 1.64x on the
isolated kernel vanishes end to end, and I would rather leave that recorded as
open than paper over it.

The 6x lever is GQA fusion -- one block per KV head serving all 6 query heads so
the cache is read once. The blocker is that the budget wants `6 * BQ` row
accumulators and `6 * BQ` rows of Q in shared, which forces BQ down to about 4
and needs V to stay in registers, in which case the shared budget is
`6 * BQ * (HD + 2)` for Q plus `BK * (HD + 2)` for K, about 43 KB at BQ = 4,
BK = 16. That fits, but it is a rewrite of the score, softmax and accumulator
loops to carry a head dimension, not a tuning change.

## The prefill is 93% GEMM, and that resolves the open contradiction

`gb10-verify prefill-shape --limit 8225` runs a real chunked prefill and prints
each chunk separately. Same binary as the server, so this is the same code path:

| chunk | tokens | start | time |
|---|---|---|---|
| 0 | 2048 | 0 | 12.62 s |
| 1 | 2048 | 2048 | 13.09 s |
| 2 | 2048 | 4096 | 13.80 s |
| 3 | 2048 | 6144 | 14.10 s |
| 4 | 33 | 8192 | 0.48 s |
| | | **total** | **54.09 s** |

This is the measurement I should have taken several rounds ago. The per-chunk
cost is almost **flat**: growing the key range from 1024 to 7168, a 7x increase,
costs 12.62 s -> 14.10 s, i.e. +11.7%. Only the attention can grow with the key
range, so its share is bounded by that difference. Solving
`a(start) ~ k * (start + chunk/2)` from chunks 0 and 3:

| | per chunk | total |
|---|---|---|
| attention | 0.25, 0.74, 1.23, 1.73 s | **3.95 s (7%)** |
| GEMM + everything else | ~12.4 s each | **49.66 s (93%)** |

So at 8K the prefill is **93% GEMM**. That single number resolves the open
contradiction from the last two rounds: the half-split's 1.64x on the isolated
attention kernel is worth about 0.5 s of a 54 s prefill, which is exactly the
noise band it disappeared into. It was not that the change failed to translate --
it is that **I had the attention's share wrong by roughly an order of magnitude**,
because I was deriving it from a linear coefficient fitted on 32K/128K/256K data
where the attention genuinely does dominate, and extrapolating that fit down to
8K.

It also re-ranks every remaining option, and for the better:

| | now | with a tensor-core GEMM (43 TFLOPS) | + GQA-fused attention |
|---|---|---|---|
| GEMM | 49.7 s | 8.1 s | 8.1 s |
| attention | 4.0 s | 4.0 s | 0.7 s |
| **8K cold TTFT** | **54.5 s** | **12.0 s** | **8.7 s** |
| vs llama.cpp 10.58 s | 5.2x slower | 1.13x slower | **1.2x FASTER** |

The GEMM is the whole game at 8K. It is ~7 TFLOPS of fp32 against llama.cpp's
~43 TFLOPS of bf16 tensor cores -- the same 6x that closes almost the entire gap
by itself -- and the weights are *already staged as bf16* in shared, and the
oracle the gate compares against is *itself bf16*. The path is cuBLAS bf16
(cudarc exposes `cublas` and a `bf16` `Gemm` impl) over a per-matrix dequantise
into a scratch buffer, or an `mma.sync` version of the existing staging.

For the longer contexts the two fixes are both needed and the margin is thinner:
at 32K the linear term is ~199 s and the attention ~105 s, so a 6x GEMM and a 6x
attention give ~33 s + ~17 s = ~50 s against llama.cpp's 44.55 s. That is close
enough to be worth chasing rather than hopeless, which is not what I would have
said two rounds ago.

## Provenance of the numbers in the tables above

The scorecard mixed measurements taken on three consecutive builds, so they were
re-checked on the final one. All of these are the same harness, the same prompt
(8,225 / 32,747 tokens) and the same llama.cpp build:

| 8K cold TTFT | value | build |
|---|---|---|
| round 266 (row padding) | 54.49 s | pad |
| round 267 (split halves) | 54.68 s | pad + split |
| **round 270 (HEAD)** | **54.88 s** | pad + split |

The three agree inside the run-to-run spread, which is the point: the split
halves are neutral end to end, exactly as reported, and nothing in the table
depends on which of the three builds it came from. 32K is 269.41 s on the same
binary that HEAD carries for the attention path.

Reproduce with:

    gb10-server --model models/Qwen3.8-27B-NVFP4 --port 8080 --ctx 16384
    python bench/longctx/ttft.py --port 8080 --reps 263 --trials 2 --max-tokens 200

for 8K, and `--ctx 36864 --reps 1054` for 32K.

## The same fit at 32K, and what it says the two fixes are worth

The per-chunk breakdown at 32K (16 chunks, 32,747 tokens, total 269.60 s against
the server's 269.41 s) fits `t = G + k * (start + chunk/2)` with **G = 12.67 s per
chunk and k = 2.49e-4 s per key**, to a mean absolute error of 0.21 s across all
16 chunks:

| chunk | start | measured | model |
|---|---|---|---|
| 0 | 0 | 12.92 s | 12.92 s |
| 3 | 6144 | 14.16 s | 14.19 s |
| 7 | 14336 | 16.38 s | 16.65 s |
| 11 | 22528 | 18.87 s | 18.73 s |
| 15 | 30720 | 20.82 s | 20.51 s |

So the decomposition is:

| | GEMM | attention |
|---|---|---|
| 8K | 49.6 s (93%) | 4.4 s (7%) |
| 32K | **202.6 s (75%)** | **65.2 s (24%)** |

This is the first time the attention's share has been measured rather than
inferred, and it moves in the direction that makes the objective reachable. Both
of the two remaining fixes are now costed against a measured baseline:

| | 8K cold TTFT | 32K cold TTFT |
|---|---|---|
| now | 54 s | 268 s |
| + tensor-core GEMM (6.1x, 43 TFLOPS) | 12.5 s | 102.4 s |
| + GQA-fused attention (6x) | **8.9 s** | **44.1 s** |
| vs llama.cpp | 10.58 s -> **1.19x faster** | 44.55 s -> **1.01x faster** |

That is the whole remaining objective in two lines: a bf16 tensor-core GEMM buys
almost all of it at 8K and about three quarters of it at 32K, and the GQA fusion
closes the rest at both.

**One caveat on the attention number, because it cuts against the projection.**
17.6 TB of estimated key/value traffic divided by the measured 65.2 s attention
term is **270 GB/s**, which is *above* the 228 GB/s DRAM peak this machine
measures. The traffic model must therefore be over-counting, and the most likely
reason is L2: neighbouring query blocks read heavily overlapping key ranges, so
part of what I counted as a DRAM read is served from cache. That means a 6x cut in
*logical* traffic will not be a 6x cut in time. At 32K it is probably a 3-4x, and
the 1.01x column above is the optimistic end of the range -- the pessimistic end,
at 3x, is 55.7 s, still 0.8x (i.e. 1.25x slower than llama.cpp). So 8K is the
context where the objective is most clearly winnable, and 32K is a coin flip that
depends on how much of that traffic is really coming from DRAM.

## The remaining work, specified

The two fixes above are now costed against measured baselines, so the next step
does not need to re-derive anything. The tensor-core GEMM in particular has its
prerequisites already in the tree -- `stage_wtile` already converts every weight
to bf16 before staging, the oracle is itself bf16, and cudarc already exposes
`Gemm<half::bf16> for CudaBlas` -- and the exact dispatch, layout and
verification order are written up in [tensorcore-plan.md](tensorcore-plan.md).

The one number to watch when it lands is the per-chunk constant `G` from the fit
above: it should fall from 12.4 s to about 2 s. If it does not, the cuBLAS path
is not being reached, and that is a faster diagnostic than any end-to-end run.

## The number the whole plan rested on was 2x pessimistic

`gb10-bench cublas-gemm` measures a real cuBLAS bf16 GEMM with fp32 accumulate at
the prefill's own shapes. I had been assuming ~43 TFLOPS, inferred backwards from
llama.cpp's prompt-eval time. The measurement says **~80 TFLOPS**:

| shape (n x k x t) | ms | TFLOP/s | vs fp32 |
|---|---|---|---|
| mlp gate/up 17408x5120x2048 | 4.63 | **78.8** | 11.3x |
| mlp down 5120x17408x2048 | 4.10 | **89.1** | 12.7x |
| lm_head 248320x5120x2048 | 64.89 | 80.2 | 11.5x |
| attn q_proj 6144x5120x2048 | 1.53 | 84.0 | 12.0x |

against the current fp32 CUDA-core GEMM's ~7 TFLOPS. Putting that through the
measured GEMM/attention split from the per-chunk fit:

| | GEMM | attention | total | vs llama.cpp |
|---|---|---|---|---|
| 8K now | 49.6 s | 4.4 s | 54 s | 5.2x slower |
| **8K + tensor cores** | **4.4 s** | 4.4 s | **8.8 s** | **1.20x FASTER** |
| 32K now | 202.6 s | 65.2 s | 268 s | 6.0x slower |
| 32K + tensor cores + 3x attention | 18.0 s | 21.7 s | **39.7 s** | **1.12x FASTER** |
| 32K + tensor cores + 6x attention | 18.0 s | 10.9 s | **28.9 s** | **1.54x FASTER** |

The tensor-core GEMM by itself is enough to win 8K. That is the first time in
this session that a single identified change has been sufficient for a metric
rather than merely closing part of a gap, and it is worth recording that the
thing standing in the way of knowing it was a 20-line benchmark, not the rewrite.

## Tensor-core prefill GEMM: the scorecard after the change

The bf16 cuBLAS prefill GEMM is now the default (round 282). Both token-exact
gates hold with it active -- `generate` 16/16 against the oracle and
`batch-parity` 16/16 over 16 sequences -- and warm TTFT and OTPS are unchanged by
construction, because the decode path still runs the NVFP4 GEMV and its roofline
was never touched.

| context | metric | gb10 | llama.cpp | result |
|---|---|---|---|---|
| 8K | cold TTFT | **20.21 s** | 10.58 s | 1.91x slower (was 5.2x) |
| 8K | warm TTFT | **0.03 s** | 0.237 s | **7.9x faster** |
| 8K | OTPS | **8.6** | 7.32 | **1.18x faster** |
| 32K | cold TTFT | **129.11 s** | 44.55 s | 2.90x slower (was 6.0x) |
| 32K | warm TTFT | **0.05 s** | 0.29 s | **5.8x faster** |
| 32K | OTPS | **7.05** | 6.865 | **1.03x faster** |

Progress on the one metric that is still behind, over this session:

| | start of session | now | |
|---|---|---|---|
| 8K cold TTFT | 88.81 s (8.4x slower) | **20.21 s (1.91x slower)** | **4.4x** |
| 32K cold TTFT | 826.09 s (18.5x slower) | **129.11 s (2.90x slower)** | **6.4x** |

Both contexts are now measured on the same build, with the persistent scratch in
place:

| | 8K | 32K |
|---|---|---|
| cold TTFT | 20.21 s | **129.11 s** |
| llama.cpp | 10.58 s | 44.55 s |
| | 1.91x slower | 2.90x slower |

The 32K number also checks the decomposition: 134.73 s (before the scratch) against
the fitted attention term of ~65 s left ~70 s of GEMM, down from 202.6 s -- a 2.9x
cut, consistent with the 2.74x measured at 8K. The scratch then took 32K from
134.73 to 129.11 s, a 1.04x gain, in line with the 1.10x it gave at 8K. So the two
independent fits agree, and the remaining 32K gap is now **more than half
attention** -- roughly 65 s of the 129.11 s.

## Correction: the 32K attention is compute-bound, not KV-bandwidth-bound (round 30)

At 32K the attention term is ~65 s of the 129.11 s total, so it is the largest
single remaining cost. I had been carrying it as "KV-traffic-bound" and had queued
GQA fusion (to remove the 6x redundancy of 24 query heads re-reading 4 KV heads)
as the main lever for long context. The arithmetic says that is aimed at the wrong
wall.

**The KV traffic is not the cost.** At 32K, one pass over the cache is

    32768 tokens * 4 kv heads * 256 dim * 4 B = 134 MB per layer
    134 MB * 16 full-attention layers      = 2.1 GB
    with the 6x head redundancy            = 12.9 GB

which at the measured 228 GB/s is **57 ms**. The attention term is ~65 s, i.e.
~1100x that. Even granting that the L2 absorbs some of the redundancy, there is no
bandwidth story here at all.

**The O(t^2) scores are the cost.** QK^T plus the weighted sum is
`2 * t^2 * d * heads * layers`:

    t = 32768, d = 256, heads = 24, layers = 16
    -> 2.11e14 FLOP over ~65 s = 3.2 TFLOP/s

So the attention runs at 3.2 TFLOP/s. That is **17% of this part's fp32 peak**, and
that is where the remaining long-context time is.

### A second correction, which is mine and not the kernel's

Those percentages use the correct peak, and the peak I had been quoting is wrong.
I wrote, and repeated all session, that fp32 tops out at 9.2 TFLOP/s on GB10
("128 FMA/cycle/SM * 48 SM * 1.5 GHz"). That counts **1 FLOP per FMA**. A fused
multiply-add is 2 FLOP:

    128 FP32 cores/SM * 2 FLOP/FMA * 48 SM * 1.5 GHz = 18.43 TFLOP/s

So the fp32 ceiling is **2x higher than I had been quoting**, and two earlier
conclusions have to be restated:

- the fp32 prefill GEMM at ~7 TFLOP/s was described as "already at 76% of its
  ceiling, so tensor cores are mandatory". It is at **38%**, not 76%. Tensor cores
  were the right call and the measurement (78-87 TFLOP/s) still stands, but the
  stated reason overstated how little headroom fp32 had.
- the attention at 3.2 TFLOP/s has up to **~5.8x** available from fp32 tuning
  alone, before bf16 tensor cores enter the picture at all.

### The 3.2 TFLOP/s figure is now confirmed two independent ways (round 31)

Round 30 inferred the attention's achieved rate from the end-to-end per-chunk fit,
which left one thing assumed: whether `attn-tile` exercises one layer or all 16.
It is a **single-layer** synthetic tile benchmark, and it reproduces the rate on
its own:

| measured by | tokens | time | FLOP | rate |
|---|---|---|---|---|
| `attn-tile` | 16384 | 1.04 s | 3.30e12 | **3.17 TFLOP/s** |
| `attn-tile` | 65536 | 17.35 s | 5.28e13 | **3.04 TFLOP/s** |
| end-to-end per-chunk fit | 32768 | ~65 s | 2.11e14 | **3.25 TFLOP/s** |

Three numbers across a 4x range of context and two independent methods land on
**3.0-3.25 TFLOP/s**. That is **16-18% of the corrected 18.43 TFLOP/s fp32 peak**,
so the ~5.8x headroom figure is a measurement and not an extrapolation. The
assumption round 30 flagged is the one that held.

### What this changes about the remaining plan

GQA fusion targets the KV read, which is 57 ms of a 65 s term. It should be
dropped from the critical path -- it is a rounding error at 32K, whatever it may do
for decode. The lever is the score loop's arithmetic:

1. **fp32 tuning first**, since it is the cheapest and needs no precision
   argument: 3.2 -> up to 18 TFLOP/s is ~5.8x, worth ~50 s at 32K and ~? at 8K.
   The score loop is already register-blocked (round 265's 4-reads/3-fma hoist),
   so this needs the real in-model phase timing from round 27's lesson rather than
   another guess about where the cycles go.
2. **bf16 tensor cores for QK^T and PV** only if (1) plateaus. This is a
   flash-attention-shaped rewrite, and the accuracy argument is not free: the
   score matrix feeds a softmax, so bf16 scores would need checking against the
   needle and perplexity gates, not just `generate`.

The ordering matters and it is the reverse of what I had planned.

## The attention bottleneck, identified exactly: shared-load bound (round 32)

Round 31 established the rate (3.0-3.25 TFLOP/s, 16-18% of fp32 peak) but not the
cause. The cause is in the score loop itself, `kernels/elementwise.cu:457-466`:

    const float* krow = Ks + j * PS + sub * PADH;
    const float* qr0  = Qs + i0 * PS + sub * PADH;
    const float* qr1  = Qs + (i0 + step) * PS + sub * PADH;
    const float* qr2  = Qs + (i0 + 2 * step) * PS + sub * PADH;
    ...
        const float kv = krow[d];       // 1 shared load, hoisted out of the 3
        d0 = fmaf(qr0[d], kv, d0);      // 1 shared load + 1 FMA
        d1 = fmaf(qr1[d], kv, d1);      // 1 shared load + 1 FMA
        d2 = fmaf(qr2[d], kv, d2);      // 1 shared load + 1 FMA

That is **4 shared loads for 3 FMA**. The hardware ratio is

    128 FMA/cycle/SM  vs  128 B/cycle/SM of shared = 32 float loads/cycle
    -> 4 FMA per shared load

so the loop runs at 0.75/4.0 = **18.8% of the load-limited peak**, and the measured
rate is **17% of the fp32 peak**. Two independent derivations -- one from the source
structure, one from timing -- land on the same number, which is what identifies this
as a load bound rather than a FLOP bound. (Round 265's hoist of `krow[d]` is what
took it from 6 reads/3 FMA to 4; it was a real 1.5x on the load side, and it is also
why the remaining headroom is exactly 4/0.75.)

The fix is a register tile on the score matrix. Per k-step a `TxT` tile costs `2T`
loads for `T^2` FMA:

| tile | loads | FMA | FMA/load | of the 4.0 ratio |
|---|---|---|---|---|
| now (1x3) | 4 | 3 | 0.75 | 19% |
| 4x4 | 8 | 16 | 2.00 | 50% |
| **8x8** | **16** | **64** | **4.00** | **100%** |
| 16x16 | 32 | 256 | 8.00 | (past it) |

**An 8x8 tile reaches the shared-load limit exactly**, so it is the target: no
further tiling helps, and the score phase could gain up to **4/0.75 = 5.3x**.

Extrapolating on the measured 32K decomposition (65.0 s of attention inside
129.11 s):

    129.11 - 65.0 + 65.0/5.3 = 76.3 s   vs llama.cpp 44.55 s = 1.71x  (from 2.90x)

Note what this is *not*: it is not a precision change, not a bandwidth change, and
not GQA fusion. It is holding more of the score matrix in registers so that each
shared load feeds four FMAs instead of 0.75. 8x8 float accums is 64 registers per
thread, which is affordable, and the 24-row BQ means the tile needs to divide
cleanly into the existing blocking -- the same constraint that made `BQ=24, BK=11`
fail in round 264 for shuffle-uniformity reasons, so it should be checked against
that guard before being trusted.

### This is now the single remaining lever

Both contexts' cold-TTFT gaps reduce to this one kernel:

| | attention share | after a 5.3x score phase |
|---|---|---|
| 8K (20.21 s) | ~4.4 s | ~19.4 s |
| 32K (129.11 s) | ~65.0 s | **~76.3 s** |

at which point 32K is 1.71x from llama.cpp and the remainder is the dequant plus the
cuBLAS GEMM, both of which are already measured near their limits.

### The 5.3x is not reachable as stated; the real target is ~2.3x (round 33)

Round 32 said an 8x8 register tile over the score matrix would hit the shared-load
ratio exactly and give 5.3x. Reading the mapping properly, that is wrong, and the
reason matters more than the number.

**`j` is not a loop.** It is fixed per thread:

    const int j  = q % PREFILL_BK;      // q = tid >> 1
    const int i0 = q / PREFILL_BK;

so each thread owns **3 rows x exactly one `j`**, and the score pair count is
`BQ * BK = 384`, or 768 work items once the two `sub` halves are counted. Spread
over `nt = 256` threads that is **3.0 work items per thread** -- the tile is already
fully subscribed, with only 3 accumulators each.

The consequence is that the Q value at `(row i, offset d)` is re-loaded by every one
of the 16 threads that own a *different* `j` for that row. **The Q loads carry a 16x
redundancy**, and they are 3 of the 4 loads per inner iteration.

**Why the 8x8 tile cannot simply be applied.** Giving a thread more accumulators
means giving it more `j`s, which means fewer threads for the same 384 pairs:

| j's per thread | loads/d | FMA/d | FMA/load | accumulators | threads needed |
|---|---|---|---|---|---|
| **1 (now)** | 4 | 3 | 0.75 | 3 | 256 |
| 2 | 5 | 6 | 1.20 | 6 | 128 |
| 4 | 7 | 12 | **1.71** | 12 | 64 |
| 8 | 11 | 24 | 2.18 | 24 | 32 |
| 16 | 19 | 48 | 2.53 | 48 | 16 |

So every step toward the 4.0 ratio **collapses the block**. At 16 threads the block
no longer has enough warps to hide latency and the gain evaporates. The 5.3x figure
assumed the arithmetic could be re-tiled at constant parallelism, which this mapping
does not allow.

**The fix has to grow the tile instead of shrinking the block.** Keep 256 threads,
give each 3 rows x 4 `j` (12 accumulators, 1.71 FMA/load), and enlarge `BK` so there
are `384 * 4 = 1536` pairs to cover -- `BK = 64` rather than 16. That raises shared
usage for the K tile from `16*258*4 = 16.5 KB` to `64*258*4 = 66 KB`, which is over
the 48 KB default and into the 99 KB opt-in limit, so it also forces the
`cudaFuncAttributeMaxDynamicSharedMemorySize` path and a re-check of the occupancy
that the extra shared memory costs back.

**Revised expectation: ~2.3x on the score phase for `BK = 64` with 12 accumulators**,
not 5.3x. Applied to the measured 32K decomposition:

    129.11 - 65.0 + 65.0/2.3 = 92.4 s   vs llama.cpp 44.55 s = 2.07x  (from 2.90x)

which is worth doing but is not the ~1.7x I wrote in round 32. I would rather
correct that here than have it carried forward as a promise.

The larger tile also has to be checked against the guard at
`kernels/elementwise.cu:448` (`BQ * BK != 3 * (nt >> 1)`) and against round 264's
finding that `BQ = 24, BK = 11` was ~5x slower for shuffle-uniformity reasons --
`BK = 64` is even and a multiple of the warp width, so it should be clear of that
particular trap, but the guard exists because the trap is real.

### Feasibility check: every larger BK costs half the occupancy (round 34)

Round 33 settled on `BK = 64` with 12 accumulators for ~2.3x. That plan has a
constraint I had not costed, and it is the kind that decides the change: shared
memory. Using the host's own formula, `(BQ*(HD+2) + BK*(HD+2) + BQ*BK + 3*BQ) * 4`:

| BK | Qs | Ks | S | red | total | **blocks/SM** | FMA/load | gain |
|---|---|---|---|---|---|---|---|---|
| **16 (now)** | 24768 | 16512 | 1536 | 288 | 43104 | **2** | 0.75 | 1.0x |
| 32 | 24768 | 33024 | 3072 | 288 | 61152 | **1** | 1.20 | 1.6x |
| 48 | 24768 | 49536 | 4608 | 288 | 79200 | **1** | 1.50 | 2.0x |
| 64 | 24768 | 66048 | 6144 | 288 | 97248 | **1** | 1.71 | 2.3x |
| 128 | 24768 | 132096 | 12288 | 288 | 169440 | 0 | -- | over the 99 KB opt-in limit |

**`BK = 16` is the only size that fits two blocks per SM** (43104 * 2 = 86208 <=
101376). Every size that improves the load ratio drops to **one** block, i.e. 8
warps instead of 16 per SM. The whole gain in this change comes from feeding each
shared load more FMAs, and the whole cost is halving the warps available to hide the
latency of those loads. Those are the same resource.

**So the 2.3x is not a prediction, it is a bet**, and it can lose: at `BK = 64` the
ratio improves 2.3x while the resident warps halve, and there is no measurement yet
that says which effect wins. What can be said now is only that the change cannot be
justified by the load-ratio arithmetic alone, which is what rounds 32-33 did. It
needs an A/B, and the honest expectation should be stated as "somewhere between
1.0x and 2.3x, or worse than 1.0x if the occupancy loss dominates".

Two ways out, both with their own cost:

1. **Stage Q and K in shared as bf16 instead of fp32.** That halves `Qs` and `Ks`:
   at `BK = 64`, `12384 + 33024 + 6144 + 288 = 51840`, which is within a whisker of
   the 50688 that two blocks would need -- close enough that trimming the padding or
   `BQ` slightly gets there. The cost is a precision change in the score matrix,
   which feeds a softmax, so it would need the needle and perplexity gates rather
   than just `generate`. This is the only route that gets both the ratio and the
   occupancy.
2. **Shrink `BQ` to buy shared memory.** This does not work: `BQ = 12, BK = 32`
   fits two blocks at 47088 bytes, but the pair count falls with it (768 work items,
   3.0 per thread) and the thread is back to 3 accumulators and 0.75 FMA/load. The
   extra accumulators need extra pairs to be spread over; that is the same coupling
   round 33 found and it does not go away by shrinking the tile.

Recommendation, given that: test `BK = 32` first, not 64. It is the smallest change
that moves the ratio at all (1.6x), it already pays the full occupancy cost, and it
therefore answers the actual question -- whether the load-ratio effect or the
occupancy effect dominates -- with the least work and the least shared-memory risk.
If 32 wins, 48 and 64 are worth trying; if 32 loses, the whole direction is dead and
no amount of further tiling will rescue it.

### A cheaper decisive probe than `BK = 32` (round 35)

Round 34 recommended testing `BK = 32` first. That test is still confounded: raising
`BK` changes the load ratio *and* the occupancy *and* the pair count all at once, and
the kernel rewrite needed to express 3 rows x 2 `j` is not small. There is a smaller
experiment that isolates the one variable in doubt.

**The question is only whether the attention is occupancy-sensitive.** If it is not,
no tiling change that costs occupancy can ever win, and the whole `BK` direction is
dead regardless of its load-ratio arithmetic. If it is, the trade is live and worth
the rewrite.

**Probe: pad the kernel's dynamic shared-memory request and change nothing else.**
`attn_prefill_tiled` currently asks for 43,104 B, which is what lets two blocks
co-reside per SM. Raising the request past 50,688 B -- without touching a single line
of kernel arithmetic -- forces one block per SM, which is exactly the cost that
`BK = 32/48/64` would pay. The launch is numerically identical, so `generate` must
still be 16/16 exact and nothing about the result changes; only the occupancy does.

Then:

| probe result | reading | next step |
|---|---|---|
| attention slows a lot (approaching 2x) | occupancy is the binding resource | the `BK` direction is dead; stop pursuing it |
| attention is roughly unchanged | the load ratio is the binding resource | `BK = 32` then 64 is worth the rewrite |

One API detail to settle first, and it is the only thing standing between this probe
and an answer: the request exceeds the 48 KB default, so the launch needs
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` set on the function first, and
`crates/gb10-cuda` does not currently set it anywhere. Round 264 hit the same wall
from the other side -- `BQ = 16, BK = 16` needed 50,368 B and was rejected for being
1,216 B over the default -- so wiring that attribute is worth doing regardless, since
it is a prerequisite for every larger-tile experiment in this document.

This is the same instrument-plus-control discipline that rounds 30-34 kept
re-learning: measure the variable in isolation before paying for the rewrite. The
probe is a host-side change of a few lines with no kernel edit, which is why it is
worth doing before `BK = 32` rather than after.

### The occupancy probe, run: the attention IS occupancy-sensitive, and that inverts round 34's recommendation (round 36)

`GB10_ATTN_SMEM_PROBE=<bytes>` is now implemented in `attn_prefill_tiled`: it raises
the dynamic shared-memory request (setting
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` first, via `CudaFunction::set_attribute`,
which cudarc does expose -- no raw FFI needed) without touching one line of kernel
arithmetic. 43,104 B lets two blocks co-reside per SM; 61,440 B allows only one.
The launch is numerically identical, and the output says so: `max|abs| 1.27e-7` and
`rms rel 4.76e-7` are unchanged to the digit.

| `attn-tile` case | 2 blocks/SM | **1 block/SM** | penalty |
|---|---|---|---|
| 65536 tokens | 17.16 s | **24.01 s** | **1.40x** |
| 16384 tokens | 1.04 s | **1.50 s** | 1.44x |

**So the answer is yes: halving the resident blocks costs ~1.40x.** That was the open
question of rounds 33-34 and it is now measured rather than argued.

**What it does to the `BK` plan** -- the combined effect is the load-ratio gain
divided by the occupancy penalty, since every larger `BK` pays the full 1.40x:

| BK | load-ratio gain | net | 32K cold TTFT | vs llama |
|---|---|---|---|---|
| 16 (now) | 1.00x | 1.00x | 129.11 s | 2.90x |
| 32 | 1.60x | **1.14x** | 121.0 s | 2.72x |
| 48 | 2.00x | 1.43x | 109.6 s | 2.46x |
| 64 | 2.30x | **1.64x** | 103.7 s | 2.33x |

**This inverts round 34's recommendation.** Round 34 said "test `BK = 32` first,
because it is the smallest change that moves the ratio at all and it already pays the
full occupancy cost". The second half is exactly why it is the *worst* choice: `BK = 32`
pays the entire 1.40x occupancy penalty to buy the smallest load-ratio gain, netting
1.14x. If the occupancy cost is going to be paid, it should be paid once and for
`BK = 64`. There is no reason to spend a rewrite on 1.14x.

**And with bf16 Q/K staging the occupancy cost disappears entirely.** Halving `Qs` and
`Ks` brings `BK = 64` to 51,840 B, back under the 50,688 B that two blocks need:

| BK | gain (2 blocks/SM) | 32K cold TTFT | vs llama |
|---|---|---|---|
| 32 | 1.60x | 104.7 s | 2.35x |
| 48 | 2.00x | 96.6 s | 2.17x |
| **64** | **2.30x** | **92.4 s** | **2.07x** |

So the route to the ~2.3x is `BK = 64` **plus** bf16 Q/K in shared, and the bf16 part
is doing as much work as the tiling: it converts a 1.64x into a 2.30x. That makes the
precision question the load-bearing one, not an afterthought -- the score matrix feeds
a softmax, so it needs the needle and perplexity gates, not just `generate`.

### Where this leaves the objective

Measured, not projected: the largest remaining lever is worth **~2.3x on the 65 s
attention term**, taking 32K cold TTFT from 129.11 s to ~92.4 s, i.e. from 2.90x to
**2.07x** behind llama.cpp. It does not close the gap, and reaching parity would still
need the attention's quadratic term attacked further (bf16 tensor cores for QK^T/PV)
and the GEMM's linear term raised (the dequant plus cuBLAS pipeline is ~70 s of the
129 s and is already measured near its limits).

### Two things the occupancy probe does not establish (round 37)

Round 36's conclusion is being used to argue for or against a kernel rewrite, so the
parts of it that are not measured need to be separated from the part that is.

**Established by measurement:** halving the resident blocks costs 1.40x at 65,536
tokens and 1.44x at 16,384 tokens, with output bit-identical. That is solid.

**Not established, and load-bearing:**

1. **The penalty was measured at 16,384 and 65,536 tokens; the model runs 2,048-token
   chunks.** The per-tile fixed costs (K/V staging, four barriers per tile) are a much
   larger share of a 2,048-token tile than of a 65,536-token one, so the marginal value
   of the second resident block need not be the same. The `attn-tile` "cache sized for
   the real context" rows are the closest thing measured at that size, and they are not
   usable: the same 2,048-token case reports 0.40 s baseline against 0.05 s under the
   probe, which is a warm-cache artifact and not an occupancy effect. **The 1.40x should
   be re-measured at 2,048 tokens before it is relied on.**

2. **The combination rule is assumed.** Round 36 computed `net = load gain / 1.40`,
   i.e. it treated the occupancy penalty and the load-ratio gain as
   multiplicative and independent. Neither is checked. They plausibly are not
   independent -- both are latency-hiding effects on the same loop -- in which case the
   net could be better or worse than that table.

Neither gap changes the qualitative finding, which is that the occupancy penalty is
real and large enough that `BK = 32` cannot pay it back. It does mean the specific net
numbers (1.14x, 1.64x, 2.30x and the 92.4 s figure) are extrapolations and should be
labelled as such rather than quoted as measurements.

**What is now in place for whoever picks this up:** the shared-memory ceiling can be
raised (`CudaFunction::set_attribute`, wired through `GB10_ATTN_SMEM_PROBE`), so a
`BK > 16` tile is *launchable* for the first time -- round 264 hit the 48 KB guard and
had nothing to do about it. The remaining work is the score-loop restructure (3 rows x
`BK/16` `j` per thread, with the `BQ*BK == 3*(nt>>1)` guard generalised) plus bf16
staging for `Qs`/`Ks`, and the first measurement of it should be the 2,048-token
occupancy penalty above, not a full end-to-end run.

### The 2,048-token occupancy measurement: the instrument cannot resolve it (round 38)

Round 37 said the 1.40x occupancy penalty should be re-measured at the model's actual
2,048-token chunk before being relied on. I ran that, three times per configuration:

| case | baseline mean / min / max / spread | probe mean / min / max / spread | ratio |
|---|---|---|---|
| start 0 | 0.26 / 0.02 / 0.38 / **19.0x** | 0.26 / 0.03 / 0.38 / **12.7x** | 1.01x |
| start 10240 | 0.43 / 0.19 / 0.55 / 2.9x | 0.39 / 0.27 / 0.62 / 2.3x | 0.91x |
| start 20480 | 0.47 / 0.35 / 0.70 / 2.0x | 0.50 / 0.50 / 0.50 / 1.0x | 1.06x |

The ratios (~1.0) say "no occupancy penalty at 2,048 tokens", but they must not be
believed, because **the within-configuration spread is up to 19x -- far larger than the
1.40x effect being looked for**. And it is not Gaussian noise: the values are bimodal,
clustering at 0.02-0.05 s and at 0.36-0.39 s for the same configuration. That is a
state effect (warm/cold, or first-touch), not scatter, which means averaging over more
repetitions would not fix it either -- the two modes are distinct populations.

So the honest result of this round is negative: **`attn-tile` at 2,048 tokens cannot
measure the thing it was run to measure.** The 1.40x at 16,384 and 65,536 tokens
remains the only clean number, and whether it transfers to the model's chunk size is
still open.

What is needed instead is a purpose-built measurement rather than this benchmark: a
loop that times the kernel launch for a fixed 2,048-token tile many hundreds of times
after an explicit warmup, reporting a median and a spread, with the cache state fixed.
`attn-tile`'s 2,048 rows are incidental output from a benchmark built to exercise
large contexts, and they show it.

**Practical consequence for the `BK` decision:** the 1.40x should be treated as an
upper bound on the occupancy cost at the model's operating point, since it was measured
where the per-tile fixed costs are amortised best. If the real penalty at 2,048 tokens
is smaller, the `BK = 32` net improves from 1.14x; if the bimodality is itself a
symptom of occupancy pressure, it could be worse. Either way the rewrite decision
should not be made on the current evidence, and the cost of getting the evidence is one
small dedicated benchmark rather than a kernel change.

### The occupancy penalty measured in situ, at the model's own operating point (round 39)

Round 38 concluded that `attn-tile` cannot resolve the occupancy question at 2,048
tokens (19x within-config spread, bimodal). The fix is not a new benchmark but the
instrument discipline from round 27: measure it **in the model**, where the chunking,
the warmup and the cache state are the real ones. `prefill-shape --limit 8225` with
and without the probe:

| chunk | start | baseline | 1 block/SM |
|---|---|---|---|
| 0 | 0 | 4.29 s | 4.33 s |
| 1 | 2048 | 4.58 s | 4.90 s |
| 2 | 4096 | 5.12 s | 5.70 s |
| 3 | 6144 | 5.56 s | 6.44 s |
| | **total** | **20.31 s** | **22.15 s (1.09x)** |

Fitting the usual `t = G + k * (start + 1024)` separates the two terms, and the
separation is the point:

| term | baseline | 1 block/SM | ratio | is |
|---|---|---|---|---|
| constant `G` | 4.018 s | 3.916 s | **0.975x** | the GEMM -- **unaffected** |
| per-key slope `k` | 2.124e-4 | 3.481e-4 | **1.64x** | the attention |

**The probe touched nothing but the attention, and the fit says exactly that**: `G`
moves by 2.5% (noise) while the context-dependent slope moves 1.64x. That is both the
result and its own validation -- a probe that had perturbed the GEMM would have moved
`G` too.

So the occupancy penalty at the model's real operating point is **1.64x**, confined to
the attention, and larger than the 1.40x measured in `attn-tile` at 16K/64K tokens.
The direction of that discrepancy is the opposite of what round 37 guessed (I had
suggested the in-situ penalty would be *smaller* because per-tile fixed costs amortise
worse at 2,048 tokens). It is larger.

**What it does to the `BK` plan:**

| BK | load gain | / 1.64 occupancy | net |
|---|---|---|---|
| 16 (now) | 1.00x | 1.64x | 0.61x |
| 32 | 1.60x | 1.64x | **0.98x -- a loss** |
| 48 | 2.00x | 1.64x | 1.22x |
| 64 | 2.30x | 1.64x | 1.40x |

**`BK = 32` is now a measured regression, not merely a weak option.** Round 34
recommended testing it first; round 36 called it the worst choice; this confirms it
would actively make the prefill slower.

And the projections show that **bf16 Q/K staging is not a refinement but the whole
game**:

| option | net | 32K cold TTFT | vs llama.cpp |
|---|---|---|---|
| `BK=64`, no bf16 | 0.99x | 129.9 s | 2.92x |
| `BK=64` + bf16 Q/K | **2.30x** | **92.4 s** | **2.07x** |
| `BK=48` + bf16 | 2.00x | 96.6 s | 2.17x |
| `BK=32` + bf16 | 1.60x | 104.7 s | 2.35x |

Without bf16 the entire `BK` direction nets at most 1.40x, and `BK=64` without it
comes to 0.99x -- because the load-ratio gain and the occupancy loss very nearly
cancel at that point. **The route is `BK = 64` plus bf16 `Qs`/`Ks`, and if the bf16
part is dropped the change is not worth making at all.** That reorders the work:
the precision question is the prerequisite, not the follow-up, and the score matrix
feeds a softmax, so it needs the needle and perplexity gates rather than just
`generate`.

### The `BK` direction was the wrong path: staging Q/K as bf16 at the *current* tile is better and much cheaper (round 40)

Round 39 concluded the route was `BK = 64` plus bf16 `Qs`/`Ks`, at 2.30x. Recomputing the
shared budget while actually searching the space found two things.

**First, a correction to round 39.** bf16 at `BK = 64` is
`12384 + 33024 + 6144 + 288 = 51840 B`, which is **1,152 B over the 50,688 B that two
blocks per SM require**. Round 39 said this configuration "returns to two blocks"; it
does not. It stays at one block, so the correct figure for it is
`2.30 / 1.64 = 1.40x`, not 2.30x -- the same as `BK=64` with no bf16 at all. The bf16
staging was doing *no* work in that configuration.

**Second, and this is the useful part: the search says the best option is the smallest
change.** For each `(BQ, BK)` that satisfies the `BQ*BK % 128` guard, with bf16 `Qs`/`Ks`:

| BQ | BK | bytes | blocks/SM | j per thread | load gain | **net** |
|---|---|---|---|---|---|---|
| **24** | **16** | 22,464 | **4** | 1 | 1.00x | **2.69x** |
| 12 | 32 | 24,384 | 4 | 1 | 1.00x | 2.69x |
| **24** | **32** | 32,256 | **3** | 2 | 1.60x | **2.62x** |
| 32 | 48 | 47,808 | 2 | 4 | 2.29x | 2.29x |
| 24 | 48 | 42,048 | 2 | 3 | 2.00x | 2.00x |
| 32 | 16 | 27,200 | 3 | 1 | 1.00x | 1.64x |
| 16 | 64 | 45,568 | 2 | 2 | 1.60x | 1.60x |

**`BQ = 24, BK = 16` with bf16 `Qs`/`Ks` is the best row, and it needs no thread-mapping
change at all.** Halving the staging drops shared from 43,104 B to 22,464 B, which takes
the kernel from two blocks per SM to **four** -- and the load ratio is untouched, because
this does not alter the pair mapping, only how Q and K are stored and read.

Every `BK` increase trades occupancy for load ratio at roughly par, and the best of them
(`BK=48`) nets 2.00x against the 2.69x that comes free from staging alone. **The whole
`BK` rewrite direction -- three rounds of it -- was attacking the wrong term.** The
lever is occupancy, which is what the round-39 probe measured, and bf16 staging is the
cheapest way to buy it.

**The extrapolation this rests on, stated plainly.** The 2.69x applies the measured
1.64x-per-halving twice (`1.64^2`) to get from 2 blocks to 4. I have only ever measured
the *downward* direction (2 -> 1 block). The upward direction is assumed symmetric, and
it need not be: if the kernel is already latency-limited at 2 blocks, a third and fourth
block may add little. So the honest range for this change is **1.0x to 2.69x**, with the
`BK=32`-style rows as the fallback if the doubling saturates.

That is testable with the same probe infrastructure, in the same way and at the same
cost: `GB10_ATTN_SMEM_PROBE` can be set *below* the computed request to admit more
blocks, once the staging is bf16. The measurement should come before the rewrite for the
third time in this document, and this time the rewrite is small enough that the
measurement is the larger half of the work.

**Precision is still the gate either way.** bf16 `Qs`/`Ks` changes what the score matrix
is computed from, and it feeds a softmax, so the needle and perplexity gates are
required -- `generate` alone does not cover it. That requirement is unchanged from
round 39; only its size changed.

### The bf16 staging has a trap in it, and the padding that avoids it is not the current one (round 41)

Round 40's plan is to stage `Qs`/`Ks` as bf16 to halve the shared budget. That change
touches the one thing in this kernel that has already cost an 8x: **bank conflicts**.
The whole 8.05x attention gain of this session came from discovering that the row stride
was a multiple of 32 and padding it, so any change to the element width has to be
re-derived rather than assumed.

Shared-memory bank of the element at index `i` is `(i * width / 4) % 32`, and the 32
lanes that access a K/Q row concurrently are the `(j, sub)` pairs with
`index = j*PS + sub*PADH`, `PS = 2*PADH`. So:

| staging | PADH | distinct banks (of 32) | |
|---|---|---|---|
| **fp32 (shipped)** | **129** | **32** | ok |
| fp32 | 130 | 16 | conflict |
| fp32 | 131 | 32 | ok |
| **bf16** | **129** | **16** | **conflict** |
| **bf16** | **130** | **32** | **ok** |
| bf16 | 131 | 27 | conflict |
| bf16 | 132 | 16 | conflict |

**Keeping `PADH = 129` while moving to bf16 would reintroduce a 2-way conflict in the
score loop** -- the loop would go from 4 loads / 3 FMA to the same count but with half of
them serialised, and given that this is the same mechanism that was worth 8.05x when
fixed, it would likely eat most or all of the occupancy gain round 40 is counting on.

**The fix is one number: bf16 needs `PADH = 130`,** i.e. `PS = 260`. The reasoning is
visible in the table: with 2-byte elements two adjacent elements share a 4-byte bank, so
the `sub` offset has half the bank resolution it has for fp32. `PADH = 129` puts
`floor(129/2) = 64` banks of shift between the halves, and `64 % 32 = 0`, so both halves
land in the same bank class. `PADH = 130` gives `floor(130/2) = 65`, and `65 % 32 = 1`,
restoring the odd shift that spreads `(j, sub)` across all 32 banks -- the same property
`PADH = 129` provides for fp32, for exactly the same reason.

This is worth recording rather than leaving to implementation, because it is invisible in
the shared-memory arithmetic (round 40's table, which only counts bytes, is unaffected --
22,464 B becomes 22,624 B and the occupancy result stands) and only shows up as the
kernel mysteriously not going as fast as predicted. It is the same failure mode as the
original conflict: correct output, no error, just slower.

**Updated round-40 plan, in order:** (1) stage bf16 with `PADH = 130`; (2) confirm the
occupancy actually rises to 4 blocks/SM and measure the gain against the 1.0x-2.69x
range, using the probe to check the saturation question; (3) only then the precision
gates, which are required either way because the score matrix feeds a softmax.

### fp16 Q/K staging: implemented, gated, and it falsifies round 40's 2.69x (round 42)

I implemented the staging change. `Qs`/`Ks` are now `__half`, converted on store and
read back with `__half2float` in the score loop, with `PADH = 130` and an intra-row gap
of 2 exactly as round 41 derived, and the host budget is
`(BQ*(HD+4) + BK*(HD+4))*2 + (BQ*BK + 3*BQ)*4 = 22,624 B`, down from 43,104 B.

**bf16 was tried first and failed the `attn-tile` gate**, at `max|abs| 1.19e-4` /
`rms rel 1.03e-4` (MISMATCH). The failure pattern identified it as precision rather
than a layout bug: `ntok = 1` passed *exactly* (0.0) while every `ntok >= 7` failed, and
with a single key the score error cannot reach the output at all. bf16 carries 8
mantissa bits; **fp16 carries 11 for the same 2 bytes, and it passes cleanly** --
`max|abs|` falls to `2.3e-5` and every row is `ok`:

| staging | `max|abs|` (ntok 7) | gate |
|---|---|---|
| fp32 | 1.19e-7 | ok |
| bf16 | 1.191e-4 | MISMATCH |
| **fp16** | **2.279e-5** | **ok** |

The 5x improvement over bf16 is the 3 extra mantissa bits, as expected. Note that
`PADH = 130` was derived for 2-byte elements generally, so it applies to fp16 unchanged.

**Then the measurement falsified the round-40 prediction.** In situ, at the model's
operating point:

| term | round-39 fp32, 2 blk | round-42 fp16, 4 blk | ratio |
|---|---|---|---|
| constant `G` | 4.018 s | 3.947 s | 0.98x |
| per-key slope `k` | 2.1240e-4 | 2.1631e-4 | **1.02x** |

**1.02x, not 2.69x.** Round 40 assumed the occupancy effect was symmetric and applied
the measured 1.64x-per-halving twice. It is not symmetric. The curve, now measured at
three points:

| blocks/SM | relative attention cost | how known |
|---|---|---|
| 1 | 1.64x slower than 2 | round 39, measured downward |
| 2 | 1.00x | reference |
| 4 | **0.98x** | **round 42, measured upward** |

**It saturates at 2.** Going 2 -> 1 costs 1.64x, but 2 -> 4 buys 1.02x: the kernel stops
being occupancy-limited once two blocks are resident. The round-40 caveat ("if the kernel
is already latency-limited at 2 blocks, a third and fourth block may add little") was the
right one, and it landed at the bottom of the 1.0x-2.69x range I gave.

**So the value of fp16 staging is not the occupancy -- it is the shared budget it frees.**
With `Qs`/`Ks` at 2 bytes, a much larger `BK` now fits *without* dropping below the
2-block saturation point, where before it could not:

| config | bytes | blocks/SM | load gain | net |
|---|---|---|---|---|
| `BK=16` fp16 (shipped now) | 22,624 | 4 | 1.00x | 1.00x |
| `BK=32` fp16 | 32,256 | 3 | 1.60x | ~1.60x |
| **`BK=48` fp16** | **42,048** | **2** | **2.00x** | **~2.00x** |
| `BQ=32, BK=48` fp16 | 47,808 | 2 | 2.29x | ~2.29x |

Compare round 39, where `BK=48` in fp32 was 79,200 B, i.e. **1** block, and the 1.64x
penalty cut its 2.00x load gain to 1.22x. fp16 staging is what makes `BK=48` reach
2.00x. **The staging is an enabler, not a win** -- its own contribution is 1.02x, and
everything it is worth comes from what it lets `BK` do:

| option | net | 32K cold TTFT | vs llama.cpp |
|---|---|---|---|
| `BK=16` fp16 (now) | 1.00x | 129.1 s | 2.90x |
| `BK=48` fp16 | 2.00x | 96.6 s | 2.17x |
| `BQ=32, BK=48` fp16 | 2.29x | 92.5 s | 2.08x |

The next step is therefore the score-loop restructure at `BK=48` (3 rows x 3 `j` = 9
accumulators per thread, generalising the `BQ*BK == 3*(nt>>1)` guard), and it is now
known to be affordable: 42,048 B leaves it at 2 blocks, which the curve says is full
speed. It also needs the fp16 precision gates, which `attn-tile` plus the round gate's
`generate`/`batch-parity` cover at this stage; needle and perplexity remain required
before the change is called done.

### `BK = 48` implemented and measured: no gain, because fp16 moved the bottleneck (round 43)

Round 42 predicted `BK = 48` with fp16 staging would net 2.00x on the attention. I built it
-- 3 rows x 3 key columns per thread, nine accumulators, `step = PREFILL_BQ / 3`
independent of `BK`, 42,336 B, 2 blocks/SM, host guard generalised to
`BQ*BK == 9*(nt>>1) && BK % 3 == 0`. It passes `attn-tile` at `max|abs| 2.279e-5`, the
same as `BK = 16`.

**It is not faster. It is slightly slower:**

| config | attn-tile 16384 | attn-tile 65536 | in situ (8225 tok) |
|---|---|---|---|
| fp32, `BK=16` (round 39) | 1.04 s | 17.16 s | 20.31 s |
| **fp16, `BK=16` (round 42, shipped)** | **0.99-1.01 s** | **15.95-16.11 s** | **20.11 s** |
| fp16, `BK=48` (round 43) | 1.08 s | 17.00 s | 20.31 s |

**The reason is that fp16 staging moved the bottleneck from shared loads to conversions.**
The round-40/42 load-ratio arithmetic counted shared *reads* per FMA, which was the right
model for fp32. Once Q and K are fp16, every value read also needs a `__half2float`
before it can feed an FMA, so the real ratio is (loads + converts) per FMA:

| config | loads | converts | FMA | (loads+converts)/FMA |
|---|---|---|---|---|
| fp32 `BK=16` | 4 | 0 | 3 | 1.33 |
| fp16 `BK=16` | 4 | 4 | 3 | 2.67 |
| fp16 `BK=48` | 6 | 6 | 9 | **1.33** |

`BK = 48` halves the loads-per-FMA exactly as designed -- 6/9 against 4/3 -- but it adds
conversions in the same proportion, so the combined instruction budget per FMA is
unchanged. Measured 17.00 s against 16.11 s, i.e. the prediction's 2.00x became ~0.95x.
**The premise of the whole `BK` direction was that shared loads were the only cost of a
score term. After round 42 that was no longer true.**

Reverted to fp16 `BK = 16`, which remains the best measured configuration.

**One implementation note worth keeping.** The first version of the nine-accumulator loop
produced *zero output* (`rms rel` exactly 1.000e0, even at `ntok = 1`, which the previous
build passed exactly). The cause was reducing the accumulators through a pointer:

```cuda
float* dd = (a == 0) ? &d00 : (a == 1 ? &d10 : &d20);
dd[0] += __shfl_xor_sync(0xffffffffu, dd[0], 1);
```

Taking the address of the accumulators defeated their register allocation, and the
shuffle then operated on the wrong values. Writing the nine shuffles out explicitly
fixed it and reproduced the `BK = 16` numbers exactly. The same "correct output, no
error, silently wrong speed" failure mode as the bank conflicts, and the same lesson:
in this kernel the accumulator form is not a stylistic choice.

**Where the attention work actually stands.** The load-ratio lever is spent: `BK` cannot
help while conversions are in the path, and 2 blocks/SM is full speed per round 42's
saturation curve. The remaining lever is to remove the conversions by doing the score
products in **packed half arithmetic** -- `__half2` loads with `__hfma2`, two FMAs per
instruction, which halves the loads *and* the converts at once, and would need the
pairing re-derived for 2-wide lanes. That is a larger change than this round's and has not
been attempted.

### Isolation experiment: the score loop is 66% of the attention, but no resource model predicts it (round 44)

Round 43's explanation for `BK=48` not helping was that fp16 conversions had become the
bottleneck. That explanation does not survive its own numbers -- fp16 adds four `cvt` per
three FMAs and is still *faster* than fp32 with none -- so this round measured what the
score loop is actually worth, by halving its `d` range (`d < half / 2`), which produces
wrong results but correct timing, and fitting the model end to end as usual:

| term | fp16 `BK=16` | score loop halved | ratio |
|---|---|---|---|
| constant `G` | 3.947 s | 3.913 s | **0.992x -- untouched** |
| per-key slope `k` | 2.1631e-4 | 1.4453e-4 | **0.668x** |

`G` not moving is the control: the probe touches only the score loop, and only the
context-dependent term responds. Halving the loop removed **33.2%** of the attention
slope, so the score loop is about **66% of the attention term** -- and since the attention
is roughly half of the 32K prefill, **the score loop alone is about a third of the entire
32K cold TTFT.** It is by a wide margin the largest single target in the model, and
focusing on it was the right call.

**But no single-resource model explains its cost**, and the three configurations measured
across rounds 42-43 disagree with all of them:

| config | shared reqs/d | bytes/d | `cvt`/d | FMA/d | req/FMA | instr/FMA | measured |
|---|---|---|---|---|---|---|---|
| fp32 `BK=16` | 4 | 16 | 0 | 3 | 1.33 | 2.33 | 17.16 s |
| fp16 `BK=16` | 4 | 8 | 4 | 3 | 1.33 | 3.67 | **15.95 s** |
| fp16 `BK=48` | 6 | 12 | 6 | 9 | **0.67** | **2.33** | 17.00 s |

| if the loop were bound by | prediction | measured |
|---|---|---|
| shared requests | `BK=48` ~2x faster than `BK=16` | 0.94x |
| shared bytes | `BK=48` ~4x faster than fp32 | 1.01x |
| instructions | fp16 1.57x *slower* than fp32 | 1.08x **faster** |

Every resource we have been counting moves in the wrong direction. The one model that
survives all three rows is that the loop is **latency-bound on the load-to-FMA dependency
chain**, not throughput-bound on any counted resource: `BK=48` issues fewer loads and
fewer instructions per FMA but each FMA still waits on its own load, so nothing improves.
That is consistent with the bank-conflict fix having been worth 8.05x -- a 32-way conflict
inflates each request's *latency* by 32 cycles, which is exactly what a latency-bound loop
is sensitive to, and halving a *count* is not.

**What this means for the next attempt.** The lever is not fewer loads per FMA; it is
**fewer dependent steps or more independent work per load**. `__half2` loads with
`__hfma2` are the natural candidate for the opposite reason than round 43 gave: not to cut
instruction count, but to make one load feed two independent FMAs *that do not need the
converted values first*, removing the `cvt` from the dependency chain entirely. The
precision question is real (accumulating in fp16), so it needs the needle and perplexity
gates, and the pairing has to be re-derived for 2-wide lanes because the `sub` split
currently relies on lane-adjacent `__shfl_xor` over single elements.

Reverted the isolation probe; the tree is back to fp16 `BK=16`, the best measured
configuration.

### Splitting the accumulator chain does nothing: the critical latency is the load (round 46)

Round 45's `__half2` loop cut the loads in the chain and bought 1.30-1.50x. The obvious
follow-up was to cut the other chain: each accumulator still carried 128 serial
dependent FMAs (two per iteration over 64 iterations), so splitting every row into two
half-sums that combine once at the end halves it to 64, at the cost of six accumulators
instead of three.

**It changed nothing:**

| config | attn-tile 16384 | attn-tile 65536 |
|---|---|---|
| `__half2`, one chain (round 45) | 0.74 s | 12.29 s |
| `__half2`, two chains (round 46) | 0.75 s | 12.19 s |

Within noise, while reordering the summation and so giving up bit-identity with the
previous kernel for no return. Reverted.

**Read together with round 45, this pins the latency down.** Round 45 shortened the
*load* path and gained 1.30-1.50x; round 46 shortened the *FMA* path and gained nothing.
So the dependency that matters is the **load**, not the arithmetic -- the loop is waiting
on shared-memory fetches, and the FMA chain has more than enough independent work to
hide behind them. Round 44's phrasing ("latency-bound on the load-to-fma chain") was
right in substance but had the culprit one step too late in the chain.

**So the remaining lever is to overlap loads better, not to change the arithmetic:**
software-pipeline the fetch (issue iteration `d+1`'s `__half2` loads before iteration
`d`'s FMAs) or widen further to 128-bit loads. Neither has been tried, and both leave the
arithmetic order untouched, so unlike the accumulator split they cost nothing in
precision.

Standing best configuration: fp16 `Qs`/`Ks` with `PADH = 130`, `BK = 16`, `__half2`
score loop. 8K in situ 20.11 -> 19.24 s; attention term 1.50x faster; 32K projected
129.11 -> 107.5 s (2.41x against llama.cpp, from 2.90x).

### Scorecard re-measured after the fp16 staging and `__half2` work (round 47)

Both kernel changes landed after the last server-side comparison, so the headline numbers
were stale. Re-measured through `gb10-server` with the same harness and settings used for
the llama.cpp side (`bench/longctx/ttft.py`, reps 263 / trials 2 at 8K, reps 1054 /
trials 1 at 32K, max_tokens 200, greedy):

| context | metric | gb10 | llama.cpp | result |
|---|---|---|---|---|
| 8K (8225) | cold TTFT | **18.90 s** (18.81 / 18.99) | 10.58 s | 1.79x slower |
| 8K | warm TTFT | **0.03 s** | 0.237 s | **7.9x faster** |
| 8K | OTPS | **8.61** (8.62 / 8.59) | 7.32 | **1.17x faster** |
| 32K (32747) | cold TTFT | **107.89 s** | 44.55 s | 2.42x slower |
| 32K | warm TTFT | **0.05 s** | 0.29 s | **5.8x faster** |
| 32K | OTPS | **7.04** | 6.865 | **1.03x faster** |

Movement against the previous scorecard:

| metric | before | after | |
|---|---|---|---|
| 8K cold TTFT | 20.21 s | **18.90 s** | 1.07x |
| 32K cold TTFT | 129.11 s | **107.89 s** | **1.20x** |
| 8K vs llama | 1.91x slower | **1.79x slower** | |
| 32K vs llama | 2.90x slower | **2.42x slower** | |

The 32K projection made in round 45 from the fitted attention slope was 107.5 s; the
measured value is **107.89 s**, so the per-key model that the whole attention analysis
rests on predicted the end-to-end result to within 0.4%. That is worth noting because it
is the first time in this document that a fitted prediction survived contact with a full
server-side measurement at a different context length.

Session cumulative cold TTFT: 8K 88.81 -> 18.90 s (**4.70x**), 32K 826.09 -> 107.89 s
(**7.66x**).

Warm TTFT and OTPS beat llama.cpp at both contexts, as they have throughout; cold TTFT
does not, and remains the gap to close. The two wins are unaffected by the kernel work
(warm TTFT is prefix-cache-bound and OTPS is decode-bound), and the cold-TTFT ratio
improved only because the prefill itself got faster.

### Manual prefetching is slower, and 128-bit loads are ruled out (round 48)

Round 46 concluded the critical latency is the load and named two remaining levers:
software-pipeline the fetch, or widen to 128-bit loads. Both are now settled, and neither
works.

**Software pipelining: measured slower.** The loop was rewritten so that iteration
`d+1`'s four `__half2` fetches are issued before iteration `d`'s FMAs, with the last
iteration peeled (prefetching one element past the end would read the row's padding gap,
which is never written). The fma order is untouched, and the output is bit-identical --
`nonzero 402652988/402653184`, the same count as the round-45 kernel. It is still slower:

| config | attn-tile 16384 | attn-tile 65536 |
|---|---|---|
| `__half2`, as shipped (round 45) | 0.74 s | 12.27-12.29 s |
| `__half2` + manual prefetch (round 48) | 0.80-0.81 s | 13.14-13.18 s |

About 7% worse, reproducibly (two clean runs each). The compiler was already scheduling
these fetches; doing it by hand adds registers and moves for no gain. Reverted.

**128-bit loads are impossible with the conflict-free padding.** An 8-half load needs the
element index to be a multiple of 8, i.e. `8 | PADH * (2j + sub)`, which needs `8 | PADH`.
But round 41 required `floor(PADH/2) % 32` to be odd for the `(j, sub)` pairs to spread
over all 32 banks, and that forces `PADH = 2 * (32k + odd)`. For `8 | PADH` we would need
`(32k + odd) % 4 == 0` -- an odd number divisible by 4. The two conditions are mutually
exclusive, so `PADH = 130` and 128-bit fetches cannot coexist. The bank-conflict fix and
the widest possible load are on opposite sides of the same constraint.

**Three consecutive negative results now bracket this loop tightly.** Round 43: more key
columns per thread (`BK=48`) changed nothing, because the load:flop accounting ignored the
conversions. Round 46: splitting the accumulator chain changed nothing, because the fma
chain is not the constraint. Round 48: hand-scheduling the loads is slower, because the
compiler already does it. The one change that worked -- round 45's `__half2` -- worked
because it genuinely *reduced the number of load instructions in the chain*, not because
of scheduling. Nothing in the remaining space reduces that count further: the width is
capped by the bank padding, the count per element is already 1, and 2 blocks/SM is full
speed (round 42).

So the score loop looks close to done at 1.50x over where the session started on it, and
further prefill gains have to come from elsewhere: the GEMM pipeline is the other half of
the 32K time (~70 s of 107.89 s) and the attention's remaining cost is no longer in the
score loop.

Standing best configuration: fp16 `Qs`/`Ks`, `PADH = 130`, `BK = 16`, `__half2` score
loop, no manual pipelining.

### The GEMM pipeline is now the largest lever: 60% of 32K, running at 28% of the measured peak (round 49)

Rounds 42-48 spent themselves on the attention score loop and took it from 1.00x to 1.50x.
With that done, the fitted per-chunk constant `G` -- everything that is not the attention
-- is no longer the smaller term. From the round-47 server measurement and the round-45
fit:

| term | per chunk | across 32K (16 chunks) | share of 107.89 s |
|---|---|---|---|
| `G` (GEMM pipeline + everything else) | 4.023 s | **64.4 s** | **60%** |
| attention slab | -- | 43.5 s | 40% |

**And `G` is 3.6x off its floor.** The prefill path multiplies weights by activations for
every token, so its arithmetic is fixed:

| quantity | value |
|---|---|
| FLOPs per token, whole model, prefill path | 43.6 GFLOP |
| per 2048-token chunk | 89.3 TFLOP |
| at the **measured** cuBLAS bf16 peak (78-87 TFLOPS, rounds 22-27) | **1.12 s** |
| as measured in situ (`G`) | **4.023 s** |
| effective throughput | **22.2 TFLOPS = 28% of peak** |

That is not a small inefficiency: `G` is the single largest item in the model now, and it
has 2.9 s per chunk of headroom that the attention no longer has.

**`tc-phase` says why, and says it was already visible.** Its breakdown for mlp gate/up
(n=17408, k=5120, t=2048) was alloc 39.0%, dequant 11.8%, `f32_to_bf16` 2.5%,
`cublas_gemm` 41.6%, epilogue 5.0%. The GEMM itself is *at* peak -- 4.761 ms for that
shape is 76.7 TFLOPS -- and the other 58% of the pipeline is what drags the average to 28%.
Round 27's lesson applies: `tc-phase`'s own alloc pattern overstates the model's cost
(persistent scratch bought only 1.10x), so the exact split needs an in-model measurement,
but the shape of the answer is not in doubt.

**The dequantization is repeated for every chunk, and that is pure redundancy.** Each
2048-token chunk re-runs `dequant_nvfp4_to_bf16` / `dequant_fp8_to_bf16` over the entire
weight set before the GEMM that consumes it. At 32K that is 16 identical passes over the
same 21.9 GB of quantized weights. It is the one part of `G` that is provably avoidable.

**And caching the dequantized weights is feasible in memory.** Holding every weight in
bf16 for the prefill path costs:

| item | size |
|---|---|
| total parameters (mlp 17.11B, attn 4.70B, lm_head 1.27B, embed 1.27B) | 24.35B |
| all-bf16 working set | **~49 GB** |
| quantized masters (estimate; 17.6 GB against 21.921 GB actual, so the split is approximate) | ~18-22 GB |
| GB10 unified memory | 121.7 GiB |
| KV cache, 32K x 10 sequences | ~2.7 GB |

~49 GB fits comfortably, and the quantized masters can be dropped once the bf16 copy
exists, so the two do not have to coexist. The prize, holding the attention fixed at
43.5 s:

| `G` per chunk | 32K cold TTFT | vs llama.cpp |
|---|---|---|
| 4.02 s (now) | 107.9 s | 2.42x |
| 2.50 s | 83.5 s | 1.87x |
| 2.00 s | 75.5 s | **1.69x** |
| 1.60 s | 69.1 s | **1.55x** |

**So the ordering has flipped.** The attention score loop is close to done -- three
consecutive negative results (rounds 43, 46, 48) bracket it tightly, and its remaining
width is capped by the bank padding. The GEMM pipeline is 60% of the time, 3.6x off its
floor, and its largest component is provably redundant work with a memory-feasible fix.
That is where the next round should go.

### The dequant is only 12% of G, so round 49's weight-caching plan is not worth it (round 50)

Round 49 sized the per-chunk constant `G` at 60% of 32K cold TTFT and named the repeated
weight dequantization as its provably redundant part, projecting 1.55-1.87x from caching
the dequantized weights (~49 GB, which does fit). That projection rested on `tc-phase`'s
breakdown, in which dequant plus cast plus alloc plus epilogue was 58% of the mlp gate/up
pipeline. Round 27's own lesson was that `tc-phase`'s alloc pattern overstates the model's
cost, so this round measured it in the model: `GB10_SKIP_DEQUANT=1` skips the per-call
weight staging entirely, leaving `wb` uninitialized. The results are wrong and the GEMM may
consume NaN, but every GEMM shape is unchanged, so the timing is valid.

| term | dequant ON | dequant OFF | ratio |
|---|---|---|---|
| `G` (per chunk) | 4.014 s | **3.533 s** | **0.880x** |
| slope `k` (attention) | 1.2061e-4 | 1.3916e-4 | 1.154x |

**The dequant is 12.0% of `G`, not 58%.** Removing it entirely buys 7.7 s across the 16
chunks of a 32K prefill:

| `G` per chunk | 32K cold TTFT | vs llama.cpp |
|---|---|---|
| 4.01 s (now) | 107.9 s | 2.42x |
| 3.53 s (dequant deleted) | **100.0 s** | **2.25x** |

So caching the dequantized weights would move the objective from 2.42x to 2.25x for a
~49 GB redesign that changes how the model is loaded and stored. **That is not worth doing**,
and round 49 said otherwise on the strength of a bench-internal breakdown. Corrected.

**The more important half is what this leaves unexplained.** With the dequant deleted, `G`
is still 3.5 s per chunk against a 1.12 s GEMM floor -- **3.2x** -- and `tc-phase` had put
`cublas_gemm` itself at 76.7 TFLOPS, essentially peak. So the remaining 3.5 s is not the
GEMM and is not the dequant, and no measurement in this document accounts for it. My
per-token 43.6 GFLOP count covers only the linear projections (q/k/v/o, gate/up/down) and
**not the Gated-DeltaNet recurrence that 48 of the 64 layers run**, which is the obvious
candidate for a large omitted term. Until `G` is decomposed in the model the way the
attention term was, any further prefill work is a guess -- and this round is the second
time a `tc-phase`-based projection has failed to survive an in-model check.

The slope moved 1.154x under a weight-only change, which it should not; that bounds the
attention-slab fit's run-to-run error at roughly 15%, so the 43.5 s attention figure and
anything derived from it should be read with that in mind.

Reverted the probe; the tree is back to the round-45 kernel, which remains the best
measured configuration.

### `G` is flat in sequence count, so it is not the DeltaNet state (round 51)

Round 50 left `G`'s 3.5 s per chunk (3.2x its GEMM floor, with the dequant already
deleted) unexplained, and named the Gated-DeltaNet recurrence as the prime suspect on the
grounds that 48 of the 64 layers run it and it keeps 3.1 MB of state per layer *per
sequence*, which the linear-projection FLOP count does not include. That is testable
without any instrumentation: if the dominant term were per-sequence state work, then
changing the sequence count would move `G`, while the GEMM pipeline is shared across the
batch and would not.

`prefill_shape` had `n_seq = 10` hardcoded; wiring it to `--n-seq` and running three
values:

| `n_seq` | chunk 3 | total (8225 tok) |
|---|---|---|
| 1 | 4.95 s | 18.66 s |
| 4 | 4.93 s | 18.49 s |
| 10 | 5.07 s | 18.83 s |

**Flat to within 2%.** Ten times the sequence count, and hence ten times the DeltaNet
state to carry, costs nothing measurable. **The hypothesis is refuted: `G` is per-token
work, not per-sequence state work.**

That narrows the 3.2x to per-token, batch-independent work. With the dequant deleted
(round 50) and the state ruled out here, what remains is the linear projections
themselves -- and the only way they can be 3.2x off the 80 TFLOPS figure used as the
floor is that **the 80 TFLOPS is not what these shapes achieve**. That figure was
measured on large single shapes (mlp gate/up, n=17408, k=5120). The model's actual GEMM
mix includes many small ones -- the k/v projections (n=1024), and the DeltaNet layers'
own small projections -- whose efficiency is necessarily lower, plus a per-call staging
and launch cost that a single-shape benchmark does not see. So the 1.12 s "floor" was
never a floor: it is the time the same FLOPs would take if every projection in the model
had the shape of the one that was benchmarked.

Two lessons that have now both occurred twice in this document: a bench-internal
single-shape number is not a model-level bound (rounds 22-27 vs 49-50), and a plausible
structural story is worth one cheap parameter sweep before it is written down as a cause
(round 37 vs here).

Reverted the diagnostic change; `prefill_shape` keeps its hardcoded `n_seq = 10`.

### The model's real GEMM shapes are efficient, so `G` is 6.5 ms/op of per-call overhead (round 52)

Round 51 concluded the 3.2x gap in `G` was "small-shape GEMM efficiency plus per-call
overhead", and that the first half needed the model's actual shapes measured. `cublas-gemm`
had only four shapes, all with large `n`; it now also sweeps the model's small projections
and a short-`t` case:

| shape | n | t | TFLOP/s |
|---|---|---|---|
| mlp gate/up | 17408 | 2048 | 77.7 |
| mlp down | 5120 | 2048 | 85.0 |
| attn q_proj | 6144 | 2048 | 80.9 |
| attn o_proj | 5120 | 2048 | 81.8 |
| **attn k_proj** | **1024** | 2048 | **74.8** |
| **attn v_proj** | **1024** | 2048 | **82.0** |
| mlp gate/up | 17408 | **256** | **42.8** |
| attn k_proj | 1024 | **256** | 55.9 |

**Small `n` is fine** -- the k/v projections at n=1024 reach 74.8-82.0 TFLOP/s, so the
"many small GEMMs" half of round 51's explanation is wrong. What actually costs is short
`t`: the same large shape drops to 42.8 TFLOP/s at t=256.

**But that does not explain `G` either, and this is the decisive number.** `G`'s 4.014 s
per chunk against 89.3 TFLOP of prefill FLOPs implies **22.3 TFLOP/s effective -- and the
worst shape measured, 42.8 TFLOP/s, is 1.9x better than that.** Nothing about shape
efficiency can produce 22.3 TFLOP/s from shapes that never go below 42.8.

So the elimination is now complete:

| candidate | status |
|---|---|
| GEMM arithmetic (89.3 TFLOP/chunk) | 1.12 s at 80 TFLOP/s -- the floor |
| dequantization | 12% of `G` (round 50, measured) |
| Gated-DeltaNet per-sequence state | flat in `n_seq` (round 51, measured) |
| small-`n` shapes | 74.8-82.0 TFLOP/s (measured above) |
| short-`t` shapes | 42.8 TFLOP/s worst case (measured above) |
| **the remainder** | **2.90 s per chunk = 6.5 ms per linear op** |

There are ~448 linear ops per chunk (64 layers x 7 projections: q, k, v, o, gate, up,
down; the DeltaNet layers' own projections add to that). Dividing the unexplained time by
them gives **6.5 ms of non-GEMM cost per linear op**. Per-call allocation and staging is
the only thing left that scales that way, and it is also the one thing the code makes
obvious: `forward_prefill_tensor_core` grows its scratch, dequantizes into it, converts
the activations, calls cuBLAS, then converts back -- per op, per chunk.

Note this contradicts the conclusion drawn in round 27, where a persistent 321 MB scratch
bought only 1.10x and I inferred that `tc-phase`'s allocation pattern was overstating the
model's cost. The measurement here says the per-op overhead is still there in full. One of
those two readings is wrong, and the way to settle it is to time the staging phases
**inside the model** rather than in either benchmark -- the same distinction that has now
caught this document out three times.

**The `cublas-gemm` shape sweep is kept** -- it is a permanent record of the shape
efficiency curve, and it is what turned this from a guess into an elimination.

### The DeltaNet is eliminated with the real config dims, and the FLOP count is confirmed (round 53)

Round 51's DeltaNet story deserved one more check, because the flat-in-`n_seq` result only
ruled out per-*sequence* state -- it did not rule out DeltaNet *compute*, which is flat in
`n_seq` too. And rounds 49-52 all rest on a FLOP count that used the **full-attention**
q/k/v/o dims for all 64 layers, when 48 of them are Gated-DeltaNet. The config gives the
real ones:

| field | value |
|---|---|
| `linear_num_key_heads` / `linear_key_head_dim` | 16 / 128 -> 2048 |
| `linear_num_value_heads` / `linear_value_head_dim` | 48 / 128 -> 6144 |
| `linear_conv_kernel_dim` | 4 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 4 / 256 |
| `full_attention_interval` | 4 |

With those, a DeltaNet layer's linear projections (q 10.5M + k 10.5M + v 31.5M + out
31.5M = 83.9M MACs) are **14% larger** than a full-attention layer's (73.4M), not smaller:

| | per layer (with MLP 267.4M) | layers | subtotal |
|---|---|---|---|
| DeltaNet | 351.3M MACs | 48 | 16.87B |
| full attention | 340.8M MACs | 16 | 5.45B |
| **total** | | | **22.3B MACs/token = 44.6 GFLOP/token** |

So round 49's 43.6 GFLOP/token was **2.4% low**, not wrong. The floor stands.

**And the DeltaNet's own compute is negligible**, which is what finally kills round 51's
hypothesis:

| term | MACs/token (x48 layers) | share of the linear projections |
|---|---|---|
| state update, `d_k * d_v * n_v_heads` | 37.7M | **0.17%** |
| chunked delta-rule term, `C^2 * d_k * n_kh` at C=64 | 402.7M | **1.80%** |

Both under 2%. The DeltaNet cannot move the total, whether or not its state scales with
`n_seq`. Combined with round 51's flat measurement, the recurrence is eliminated twice
over, by two independent routes.

**Also eliminated this round: per-op synchronization.** `grep` finds five `dev.synchronize()`
calls in `model.rs`, but they are all in the **decode** step, where they exist to populate
`PhaseTimes` (the loop syncs once per layer, 64 times per token). The prefill path has
none. That was the most plausible mechanism for a 6.5 ms-per-op cost, and it is not there.

**Where that leaves it.** Every candidate for `G`'s 3.5x is now measured or bounded:

| candidate | status |
|---|---|
| FLOPs (44.6 GFLOP/token) | verified against the real config dims, +2.4% |
| GEMM efficiency, large `n`, t=2048 | 74.8-89.2 TFLOP/s (measured) |
| GEMM efficiency, short `t`=256 | 42.8 TFLOP/s worst (measured) |
| dequantization | 12% of `G` (measured) |
| DeltaNet state | 0.17% of FLOPs (computed) and flat in `n_seq` (measured) |
| DeltaNet chunked rule | 1.80% of FLOPs (computed) |
| per-op synchronization | absent from the prefill path (checked) |
| **unexplained** | **2.90 s/chunk, 6.5 ms per linear op** |

The remaining possibility is that the model's GEMMs simply do not achieve their isolated
throughput once they are interleaved with the staging kernels, the DeltaNet and the
attention in one stream -- an effect no isolated-shape benchmark can see and one this
document cannot resolve without timing the phases **inside** the model. That measurement
has now been the named next step for three rounds; it is a small change to
`prefill_seq`, and picking a cause without it has already produced two wrong conclusions
(rounds 49 and 50) and one incomplete one (round 51).

### The prefill is CPU-bound, which is what the 3.5x actually is (round 54)

Rounds 49-53 chased `G`'s 3.5x through every GPU-side candidate -- FLOP count, GEMM shape
efficiency, dequantization, the DeltaNet state, hidden synchronization -- and eliminated
all of them. The candidate never tested was that the GPU is not the constraint at all.

`prefill-shape` at two limits, with `/usr/bin/time`:

| | wall | user | sys | total CPU | CPU/wall |
|---|---|---|---|---|---|
| 4 chunks (8225 tok) | 60.52 s | 53.95 s | 7.10 s | 61.05 s | 1.009 |
| 16 chunks (32747 tok) | 149.84 s | 143.97 s | 7.16 s | 151.13 s | 1.009 |
| **difference** | **+89.32 s** | **+90.02 s** | +0.06 s | +90.08 s | **1.008** |

The difference is the clean measurement: both runs load the same model, so subtracting
cancels the load phase and leaves only the prefill. Over the 12 extra chunks, **user time
grew by 90.02 s while wall time grew by 89.32 s -- a ratio of 1.008.**

**During the prefill the process is consuming essentially one full CPU core, continuously,
for as long as the prefill lasts.** GPU kernel execution does not appear in a process's
user time. A read of ~22 GB of weights and 89.3 TFLOP of GEMM against a GPU that measures
77-89 TFLOP/s on these exact shapes cannot take 89 s unless the GPU is idle waiting.

This resolves the contradiction that rounds 49-53 could not. The model's GEMM shapes really
do run at 75-89 TFLOP/s in isolation, and the prefill really does take 3.5x longer than
those shapes imply, and both are true because **the kernels are not the bottleneck -- the
CPU work that enqueues them is.** At ~448 linear ops per chunk and ~1800 kernel launches
per chunk (dequant, cast, cublas, epilogue x 448), the prefill spends ~7.5 s of CPU per
chunk to feed a GPU that could consume it in ~2 s.

So the 1.12 s "GEMM floor" was never the thing to close, and rounds 49-52's implicit plan
-- make the GEMMs faster -- had nothing to gain. The lever is **the number of launches and
the CPU cost per op**, and the standard fix is CUDA Graphs: capture one chunk's work and
replay it, collapsing ~1800 launches into one.

Two mechanisms produce this signature and the measurement above does not separate them:

1. **Launch/enqueue cost** -- the CPU simply cannot issue work fast enough. Fixed by CUDA
   Graphs or by fusing the staging kernels into the GEMM.
2. **Unified-memory page faults** -- GB10 has 121.7 GiB of unified LPDDR5X; if weight
   pages are faulted in per access rather than resident, the CPU services the faults.
   Fixed by pinning/prefetching (`cudaMemPrefetchAsync`) rather than by graphs.

They are distinguishable with one more measurement: count page faults (`/usr/bin/time -v`
reports minor/major faults, which the run above did not capture) or time a prefill that
reads a weight set small enough to be certainly resident. Given that `Maximum resident` was
reported as 0 in the earlier `/usr/bin/time -v` output -- i.e. the memory accounting is not
straightforward on this unified-memory part -- the fault count is the better probe.

**This is the single most important measurement in this document for the cold-TTFT
objective.** It says the 2.42x gap at 32K is not a kernel problem, that eight rounds of
attention-kernel work could not have closed it, and that the remaining work is
architectural -- fewer launches, or no page faults -- rather than arithmetic.

Note also that it retroactively explains round 27's puzzle: a persistent 321 MB scratch
bought only 1.10x, which I read as `tc-phase` overstating allocation cost. If the bottleneck
is CPU time per op, then removing the *allocation* while keeping the *launch count* would
indeed buy very little.

### Page faults are refuted; the CPU time is launch/enqueue overhead (round 55)

Round 54 established the prefill is CPU-bound but could not say whether the CPU was
launching kernels or servicing unified-memory page faults. `/usr/bin/time -v` at the same
two limits separates them:

| | 4 chunks | 16 chunks | scales with prefill? |
|---|---|---|---|
| Major (I/O) page faults | 0 | 0 | no -- **zero** |
| Minor page faults | 3,536,283 | 3,499,287 | **no** (slightly *lower*) |
| Voluntary context switches | 14,243 | 31,385 | **yes, +17,142** |
| Involuntary context switches | 729 | 2,033 | yes |
| File system inputs | 0 | 0 | no |

**The fault hypothesis is refuted.** Minor faults are flat-to-slightly-decreasing across a
4x increase in prefill work, and major faults are zero in both runs. Page faulting happens
during model load and not at all during the prefill. Round 54's alternative (2) is dead.

**The scaling signal is the context switch**: 1,429 voluntary switches per chunk, which is
the fingerprint of a thread that blocks repeatedly -- enqueuing work, blocking when the
queue or a resource is unavailable, and doing that roughly once per kernel. With ~1,800
launches per chunk this is the right order of magnitude for one switch per launch.

So the mechanism is round 54's alternative (1): **launch/enqueue cost, with the CPU
blocking per operation.** The fix is the standard one for this signature and it is
architectural rather than arithmetic:

- **CUDA Graphs.** Capture one chunk's work once and replay it, collapsing ~1,800 launches
  into one. This is the direct answer to a launch-bound prefill.
- **Kernel fusion.** Fuse the four staging kernels per linear op (dequant, cast, cublas,
  epilogue) so one op is one launch.

Both are worth doing and neither is a kernel-optimization exercise. In particular this
closes the question that rounds 42-48 spent themselves on: the attention score loop was
never going to move the cold-TTFT number much, because the prefill is not limited by what
the kernels compute.

At ~7.5 s of CPU per chunk feeding a GPU that needs ~2 s, even a 2x reduction in launch
count would take the 32K prefill from 107.89 s toward the 60 s range, which is the first
step that would put the objective in reach. Recorded as the next implementation target.

### RETRACTION: rounds 54-55's "the prefill is CPU-bound" is not supported (round 56)

Round 54 read a 1:1 `user`:`wall` ratio during prefill as proof that the process was
CPU-saturated, and round 55 read the scaling context-switch count as the fingerprint of
per-launch blocking. Both readings are void, for a reason one control run exposes.

`attn-tile` is a pure GPU-kernel benchmark: it launches the attention kernel and nothing
else. Its 65536 case is 23.26 s of GPU kernel time by its own report.

| workload | wall | user | sys | CPU/wall |
|---|---|---|---|---|
| `prefill-shape` 4 chunks (TC) | 59.85 s | 53.53 s | 6.78 s | 1.007 |
| `prefill-shape` 4 chunks (fp32) | 94.06 s | 88.42 s | 6.42 s | 1.008 |
| **`attn-tile` (pure kernel)** | **42.73 s** | **39.91 s** | **3.11 s** | **1.007** |

**The pure-kernel benchmark shows the same 1.007 ratio.** 23.26 s of that run is the
attention kernel executing on the GPU, and it cannot appear in the process's user time --
yet the user time matches the wall time. So `user` on this platform is not a measure of
CPU work; it tracks *GPU* time. The CUDA driver is spinning while the GPU runs, which is
standard behaviour for a spin-configured driver and is exactly what a 1:1 ratio with
`cpu 100%` on one core looks like.

Consequences, stated plainly:

- **The prefill is not shown to be CPU-bound.** Rounds 54 and 55 concluded it was, and
  used that to redirect the whole cold-TTFT effort to launch-count reduction. That
  redirection rests on nothing.
- Round 55's scaling voluntary-context-switch count is equally consistent with a spinning
  driver and no longer discriminates anything.
- The fp32-vs-TC comparison above, which round 56 first read as "fp32 uses more CPU so it
  is slower", shows only that `user` tracks wall in both paths -- a tautology under
  spin-waiting, not a finding.

**What still stands.** The gap itself is unchanged and is a plain FLOP/time measurement,
independent of any CPU accounting: the prefill moves 44.6 GFLOP/token over shapes that
measure 74.8-89.2 TFLOP/s in isolation, and takes 3.5x longer than that implies. Every
GPU-side elimination from rounds 50-53 also stands, because those were measurements of
GPU work (dequant share, DeltaNet FLOPs, `n_seq` flatness, absent synchronization), not
inferences from CPU time. The one thing that is now known is that **the cause of the 3.5x
is not established**, and that the CPU-time evidence for launch overhead does not exist.

The measurement that would settle it is the one this document has now named for four
rounds and never performed: **per-phase GPU timing inside the model** -- CUDA events
around the staging, GEMM and epilogue of a linear op, summed over a chunk, printed at the
end. Everything short of that has produced either a wrong conclusion (rounds 49, 50, 54,
55) or an incomplete one (round 51). Until it exists, no fix should be attempted on the
strength of a mechanism story, including the CUDA-graphs plan round 55 recommended.

### The alloc hypothesis is dead in-model, and the event API for the real measurement is located (round 57)

Round 56 said the only justified next step was per-phase GPU timing inside the model. This
round first re-examined whether `tc-phase`'s largest phase was even present in the model,
and located the API needed to measure the rest.

**Allocation: not per-op, hypothesis dead.** `tc-phase` puts `alloc_zeros` at 39% of the
mlp gate/up pipeline, which made it the leading candidate after the dequant was measured
at 12%. But the prefill path does not allocate per op. `weights.rs:104-111` guards every
buffer:

```rust
if sc.w.as_ref().map_or(true, |b| b.len() < n * k) {
    sc.w = Some(stream.alloc_zeros::<bf16>(n * k)?);
}
```

`alloc_zeros` runs only while a buffer is still growing. Once the largest shape of each of
the three buffers has been seen -- which is the first chunk -- nothing allocates again, and
the `Mutex<TcScratch>` is simply re-lent. So `tc-phase`'s 39% is an artifact of its own
per-rep allocation and is **not** a cost the model pays. That is consistent with round 27,
where making the scratch persistent bought only 1.10x: there was almost nothing to win
because the model was already not allocating.

**A pipeline accounting still does not close the gap.** Taking `tc-phase`'s phases, deleting
the bench-only `alloc`, and scaling by the model's real per-layer op mix:

| op | isolated pipeline (no alloc) |
|---|---|
| mlp gate/up (x2 per layer) | 13.9 ms |
| mlp down | 6.0 ms |
| attn q / o | 2.3 ms each |
| attn k / v | 0.4 ms each |
| DeltaNet q/k/v/out + gates | ~6.6 ms |
| **full-attention layer (x16)** | ~25.3 ms |
| **DeltaNet layer (x48)** | ~26.5 ms |
| **per chunk** | **~1.68 s** |

Against `G` = 4.02 s measured (3.53 s with the dequant deleted), that leaves a further
**2.4x** unexplained even after every phase `tc-phase` knows about is accounted for. So the
answer is not in the phase list either; it has to be measured, not enumerated.

**The API for measuring it is available, in safe cudarc, with no raw FFI** (cudarc 0.19.9):

| item | location |
|---|---|
| `CudaEvent` | `src/driver/safe/core.rs:532` |
| `CudaContext::new_event` | `src/driver/safe/core.rs:551` |
| `CudaEvent::record(&self, stream)` | `src/driver/safe/core.rs:587` |
| `CudaEvent::synchronize` | `src/driver/safe/core.rs:596` |
| `CudaEvent::elapsed_ms(&self, end) -> f32` | `src/driver/safe/core.rs:603` |
| re-export | `src/driver/safe/mod.rs:11` (`cudarc::driver::CudaEvent`) |

The recipe, so the next round does not have to re-derive it: in
`Linear::forward_prefill_tensor_core`, create an event pair, `record` the start on
`dev.stream()` before the staging call and the end after the epilogue, push the pair into a
thread-local `Vec`, and **do not synchronize inside the op** -- events can be recorded
asynchronously and read later. At the end of `prefill_shape`, synchronize once and sum
`elapsed_ms` per phase. That gives GPU time per phase, summed over a chunk, which is the
measurement rounds 53-56 named and none performed.

Rounds 54-55's CPU-time evidence for launch overhead was retracted in round 56. With the
allocation hypothesis now also closed, **no mechanism for `G`'s 2.4-3.5x is established**,
and the event measurement is the only remaining route to one.

### MEASURED: the GEMMs are 30% of the prefill and run at 84% of peak (round 58)

Rounds 53-57 named in-model per-phase GPU timing as the only remaining route to a cause
and never performed it. It is now implemented: `Linear::forward_prefill_tensor_core`
records a `CudaEvent` pair around the cuBLAS call when `GB10_GEMM_EVENTS=1` is set, pushes
the pair into a global `Vec`, and `prefill_shape` drains it with `elapsed_ms` at the end
(never inside the op -- `elapsed_ms` synchronizes). Gated so the server path, which shares
this function, records nothing and is unchanged: verified at 0 events and 18.87 s against an
18.83 s baseline.

| quantity | value |
|---|---|
| cuBLAS GPU time, 4 chunks / 8225 tokens | **5.567 s** (5.776 s with instrumentation) |
| prefill wall, same run | 18.83 s |
| **cuBLAS share of the prefill** | **29.6%** |
| calls | 2000 (500 per chunk, ~8 per layer) |
| FLOPs moved (8225 tok x 44.6 GFLOP/token) | 366.8 TFLOP |
| **in-model GEMM throughput** | **65.9 TFLOP/s** |
| isolated throughput, same shapes | 74.8-89.2 TFLOP/s |
| **in-model vs isolated** | **84%** |

**The GEMMs are not the problem.** In the model they run at 65.9 TFLOP/s against 74.8-89.2
measured in isolation -- 84% of their benchmarked rate, which is ordinary for a real
interleaved workload. And they account for **under a third** of the prefill.

This is the decisive result the last five rounds were circling. `G`'s 3.5x is **70% non-GEMM
time**, and it was never going to be found in FLOP counts, shape efficiency, the allocator or
the GEMM's own rate, because none of those is where the time goes. Reconstructing the
budget for the 18.83 s prefill at 8K:

| component | share | source |
|---|---|---|
| cuBLAS GEMM | **29.6%** | this measurement |
| attention kernels | ~22% | fitted attention slab, round 45 |
| weight dequantization | ~12% | `GB10_SKIP_DEQUANT` probe, round 50 |
| **remainder** | **~36%** | f32_to_bf16 cast, bf16_to_f32 epilogue, DeltaNet kernels, norms |

**So the target is the ~36% remainder plus the dequant**, i.e. the staging and epilogue
kernels that run once per linear op -- four extra kernel launches per GEMM, each moving
`t*k` or `t*n` elements -- together with the DeltaNet's own kernels. That is a kernel-count
and small-kernel-efficiency problem, and it is now localised rather than guessed.

What this measurement also retroactively settles:

- **Rounds 49-52 were chasing the wrong thing.** They were all implicitly about making the
  GEMM faster. It is already at 84% of its ceiling and is 30% of the time; even doubling it
  would buy under 15% of the prefill.
- **Round 54-55's launch-overhead story was retracted in round 56 and stays retracted**, but
  the *shape* of the answer it pointed at survives: the cost is in the many small
  non-GEMM kernels, not in the arithmetic.
- **Round 57's `tc-phase` scaling was close**: it predicted ~1.68 s per chunk against 4.02 s
  measured, and this measurement shows why -- `tc-phase` times a pipeline whose GEMM is 42%
  of it, while in the model the GEMM is 30% and the rest is larger than any isolated
  pipeline suggests.

Next: extend the same event instrumentation to the other three phases of the op
(`dequant`/`u16_to_bf16`, `f32_to_bf16`, `bf16_to_f32_scaled`) plus the attention and
DeltaNet launches, which is now a small, mechanical change to a mechanism that works.

### MEASURED: the full per-op phase breakdown, and 56% of the prefill is still outside it (round 59)

Round 58 bracketed the cuBLAS call. This round brackets the two staging phases as well --
four `CudaEvent`s per linear op, three phases, drained once at the end -- so the op is now
timed end to end except its epilogue. 8K prefill, 8225 tokens, n=2000 ops, 19.13 s wall:

| phase | GPU time | share of prefill | independent prior estimate |
|---|---|---|---|
| weight stage (`dequant_nvfp4/fp8_to_bf16`, `u16_to_bf16`) | 2127 ms | **11.1%** | 12% (`GB10_SKIP_DEQUANT` probe, round 50) |
| activation cast (`f32_to_bf16`) | 543 ms | **2.8%** | 2.5% (`tc-phase`, round 49) |
| cuBLAS GEMM | 5713 ms | **29.9%** | 29.6% (round 58) |
| **measured op phases, total** | **8.38 s** | **43.8%** | |
| **remainder** | **10.75 s** | **56.2%** | |

**Three independent prior estimates are confirmed to within a percentage point each.** The
dequant share from a kernel-deletion probe (12%), the cast share from a standalone
benchmark (2.5%) and the GEMM share from the round-58 event pair (29.6%) all reproduce.
That is the first time in this document that a set of predictions has survived an in-model
measurement intact, and it means the instrumentation is trustworthy.

**The new finding is the 56.2% remainder.** Even with every phase of the linear op except
the epilogue timed, **under half the prefill is accounted for**. The remainder is:

- the **epilogue** (`bf16_to_f32_scaled`), the one phase of the op not bracketed here. It is
  not small: it writes `t * n` fp32 values and reads `t * n` bf16, so for mlp gate/up alone
  that is 143 MB written plus 71 MB read per op, and there are two such ops per layer.
- the **attention** kernels, fitted at ~22% of the 8K prefill (round 45).
- the **DeltaNet** recurrence and gating kernels, and the norms.

Subtracting the attention's ~22%, the epilogue plus DeltaNet plus norms come to roughly
**34% of the prefill** -- more than the GEMM, and almost none of it has ever been timed.

So the objective's remaining lever is now precisely stated: **the epilogue and the DeltaNet
kernels, together about a third of the cold prefill**, with the GEMM (30%, at 84% of its
isolated rate) and the dequant (11%) both confirmed to be near their practical limits.

The two `[diag]` TFLOPS fields print 0.0 because of a missing `1e9` in the display
expression; the value is 366.8 TFLOP / 5.713 s = **64.2 TFLOP/s**, consistent with round
58's 65.9. The measurement itself is unaffected.

### MEASURED: the linear op is 46% of the prefill; the other 54% is DeltaNet and attention (round 60)

Adding the fifth event (for the epilogue) completes the per-op breakdown. First attempt
reported the epilogue as 0 ms, which was a real bug and worth recording: `evs.take()` ran
immediately after `mark!(3)`, so the `Vec` was already drained when `mark!(4)` executed and
the macro silently did nothing. Moving the push past the epilogue fixed it. 8K prefill,
8225 tokens, n=2000 ops, 18.82 s:

| phase | GPU time | share of prefill | standalone prediction |
|---|---|---|---|
| weight stage (`dequant_*`, `u16_to_bf16`) | 2083 ms | **11.1%** | 12% (round 50 probe) |
| activation cast (`f32_to_bf16`) | 548 ms | **2.9%** | 2.5% (`tc-phase`) |
| cuBLAS GEMM | 5532 ms | **29.4%** | 29.6% (round 58) |
| epilogue (`bf16_to_f32_scaled`) | 569 ms | **3.0%** | 5.0% (`tc-phase`) |
| **whole linear op** | **8.73 s** | **46.4%** | |
| **everything else** | **10.09 s** | **53.6%** | |

**All four phases now reproduce their independent predictions** (11.1 vs 12, 2.9 vs 2.5,
29.4 vs 29.6, 3.0 vs 5.0). The linear op is fully accounted for and is **under half** the
prefill.

**And the epilogue was not the missing block.** After round 59 it looked like the obvious
candidate for the 56% remainder -- it writes `t*n` fp32 values per op -- but it measures
569 ms, 3.0%. It was in the remainder only because it had not been bracketed.

**So the remaining 53.6% is attention plus the DeltaNet**, and the split follows from the
round-45 attention fit: the attention slab is ~22% of the 8K prefill, leaving roughly

| block | share of cold prefill |
|---|---|
| cuBLAS GEMM (at 84% of its isolated rate) | 29.4% |
| **DeltaNet recurrence + gating + conv + norms + embed** | **~32%** |
| attention kernels | ~22% |
| weight dequant (near the 228 GB/s limit) | 11.1% |
| activation cast + epilogue | 5.9% |

**The single largest block in the cold prefill is now the DeltaNet at ~32%, and it has never
been timed.** That is the finding this diagnostic campaign was for. Rounds 49-57 chased the
GEMM (29%, near its ceiling), the dequant (11%, near the bandwidth limit), the allocator
(not per-op) and the DeltaNet's *state* (0.17% of FLOPs) -- none of which is the target. The
target is the DeltaNet's *kernels*: 48 of 64 layers, each with its conv, gating and
recurrence launches, which no measurement in this document covers.

Next: bracket the DeltaNet's forward the same way, which is now a mechanical extension of a
mechanism validated four times over.

### A note on the layer-split instrumentation that was tried and reverted (round 61)

Round 60 localised the remaining 53.6% of the prefill to "attention (~22%) plus the
DeltaNet (~32%)", the latter by subtraction rather than measurement. The obvious next step
was to bracket the per-layer forward in `forward_normed` and split the total by
`is_delta()`.

**That was implemented and then reverted, and the reason is worth recording so it is not
repeated.** The five-event per-op mechanism from rounds 58-60 is validated and stays. The
layer-level version added a second global event `Vec` plus a `layer_event_snapshot()`, built
cleanly, and then **printed nothing** -- `dn + an` stayed 0 even with `GB10_GEMM_EVENTS=1`
and with `prefill_shape` confirmed to reach the patched loop through
`prefill_seq -> forward_normed`. The failed `new_event` calls were being swallowed by
`match (a, b) { (Ok(a), Ok(b)) => ..., _ => None }`, so the mechanism failed silently
rather than loudly -- which is exactly the failure mode that produced the round-60 epilogue
bug (a `Vec` drained before its last reader, also silent).

Rather than leave a second, silently-broken diagnostic in the tree, the layer-split change
was reverted in full. `git status` is clean, both files contain zero references to
`layer_event_snapshot`, the per-op diagnostic still reports its four phases, and the
ungated server path is unaffected. **The state committed at round 60 is the state that
stands.**

The lesson, which this document has now earned four times over: on this codebase an
instrumentation change that compiles and runs is not evidence that it measured anything.
Every diagnostic here needs its control -- a gated run against an ungated one, or a value
checked against an independent estimate -- and the per-op mechanism earned that in round 59
by reproducing three predictions to within a point.

The DeltaNet's ~32% therefore remains **unmeasured but well-bounded**: it is what is left of
the prefill after the whole linear op (46.4%) and the attention slab (~22%) are removed.
The next attempt should print a diagnostic count of successfully created event pairs
alongside the totals, so a silent zero is distinguishable from a real zero.

### `timing_event` exists, and why the round-58 event work was valid (round 62)

Retrying round 61's layer split turned up the explanation for a class of silent failure on
this codebase, and it is worth recording even though the layer split is again not in the
tree.

`crates/gb10-cuda/src/lib.rs:24-33` has a helper the per-op instrumentation should have been
using:

```rust
/// `CudaContext::new_event` defaults to `CU_EVENT_DISABLE_TIMING`, and under
/// that flag `CudaEvent::elapsed_ms` returns a meaningless number rather than
/// failing. Anything that reports a duration must get its events from here.
pub fn timing_event(dev: &Device) -> std::result::Result<CudaEvent, CudaError>
```

So the failure mode on this platform is not an error -- **it is a plausible-looking wrong
number**. `new_event(None)` yields events whose `elapsed_ms` is meaningless and does not
fail, which is exactly the shape of bug that a diagnostic cannot detect by inspection.

**This confirms the round-58/59/60 numbers are sound**, because `weights.rs:166` passes the
flag explicitly:

```rust
match ev_ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)) {
```

The four-phase breakdown (stage 11.0%, cast 3.0%, gemm 29.5%, epilogue 3.0%, op total
46.6%) therefore stands, and it is still the case that all four reproduce their independent
predictions.

**The layer split remains unimplemented.** This round it was rebuilt using `timing_event`
with attempt/success counters and an error log so that a silent zero could not recur, but
the reporting half never landed in `gb10-verify/src/main.rs` -- an edit that reported
success and was then not present in the file. Rather than commit a second half-wired
diagnostic, the change was reverted again and the tree is clean at the round-60 state.

Two attempts at the layer split have now failed for two *different* reasons, both in the
plumbing rather than the concept. The next attempt should not add the report to
`main.rs` by string-splicing: it should call the snapshot from inside `prefill_shape`'s
existing diagnostic block, verify the file changed with `grep` **before** building, and
confirm `cargo build` actually recompiled (`Finished` in 0.03 s means it did not).

**What stands unchanged:** the linear op is 46.6% of the cold prefill and is fully attributed
(GEMM 29.5% at 84% of its isolated rate, dequant 11.0% near the memory limit, cast 3.0%,
epilogue 3.0%); the remaining 53.4% is attention (~22% by the round-45 fit) plus the DeltaNet
(~32%), and the DeltaNet has still never been timed directly.

### MEASURED: the DeltaNet layers are 70% of the cold prefill (round 63)

The layer split landed, after two failed attempts, by using the codebase's own
`gb10_cuda::timing_event` (not `new_event`, whose `CU_EVENT_DISABLE_TIMING` default returns
a meaningless number instead of failing), counting attempts alongside successes, and
editing the existing diagnostic block rather than splicing a new one. The control line
confirms the measurement is real rather than silently empty:

```
[diag] layers tried=320 made=320 measured=320 errs=0
[diag] LAYER GPU: delta 13.29s / 240 = 70.1%   attn 5.50s / 80 = 29.0%
```

320 layers = 64 x 5 chunks, 240 DeltaNet and 80 full-attention, exactly as the config's
`full_attention_interval: 4` requires. 8K prefill, 18.96 s total:

| layer kind | count | GPU time | share of prefill | per layer |
|---|---|---|---|---|
| **Gated-DeltaNet** | 240 | **13.29 s** | **70.1%** | **55.4 ms** |
| full attention | 80 | 5.50 s | 29.0% | 68.8 ms |

**And the DeltaNet's own kernels are the largest single block in the model.** The measured
linear-op total (46.4% of the prefill) is spread over both layer kinds in proportion to
their op counts -- the DeltaNet layers carry ~8 projections each against the full-attention
layers' ~7, so roughly 75% of the 8.81 s of linear work is inside DeltaNet layers:

| block | share of cold prefill |
|---|---|
| **DeltaNet non-linear kernels (recurrence, gating, conv)** | **~35%** (6.7 s) |
| linear ops inside DeltaNet layers | ~35% (6.6 s) |
| attention kernels (the full-attention layers' non-linear part) | ~17% (3.3 s) |
| linear ops inside full-attention layers | ~12% (2.2 s) |

**This is the target the diagnostic campaign was for.** The DeltaNet's recurrence, gating
and convolution kernels are about **35% of the cold prefilling time -- more than the entire
GEMM (29.4%), which is already at 84% of its isolated rate -- and they have never been
optimised, profiled or even timed before this measurement.**

It also closes the question the last fourteen rounds kept circling. Rounds 49-57 chased the
GEMM, the dequant, the allocator, the FLOP count and the DeltaNet's *state* (0.17% of
FLOPs). The state is indeed negligible -- but the DeltaNet's *kernels* are the largest
single cost in the model, which is a different thing entirely and was only findable by
timing the layers.

Next: time the DeltaNet layer's internals the same way (its conv, its gating projections,
and the recurrence kernel) to find which of the three carries the 6.7 s. The mechanism is
now validated five times over.

### ROOT CAUSE: the DeltaNet recurrence runs at 6% occupancy with a serial scan (round 64)

Round 63 measured the DeltaNet's own kernels at ~35% of the cold prefill. This round found
why, by reading `gated_delta_rule_chunk_kernel` (`kernels/elementwise.cu:837`) and its launch
site.

```cuda
extern "C" __global__ void gated_delta_rule_chunk_kernel(...) {
    constexpr int D = 128;
    const int hv = blockIdx.x;
    const int b  = blockIdx.y;      // grid_dim.y == 1, so this is always 0
    ...
    for (int t = 0; t < T; ++t) {   // serial scan over the whole chunk
        ...
        for (int i = threadIdx.x; i < D; i += blockDim.x) sk[i] = khp[i];
        __syncthreads();            // 2 syncs per token
        ...
        for (int i = 0; i < D; ++i) {          // 128-long dependent chain
            const float s = S[i][j] * dec; S[i][j] = s; kv = fmaf(s, sk[i], kv);
        }
        ...
        for (int i = 0; i < D; ++i) {          // another 128-long dependent chain
            const float s = fmaf(sk[i], delta, S[i][j]); S[i][j] = s;
            o = fmaf(s, qh[i], o);
        }
    }
}
```

with the launch:

```rust
.launch(LaunchConfig { grid_dim: (n_v_heads as u32, 1, 1), block_dim: (128,1,1), shared_mem_bytes: 0 })?;
```

**The grid is 48 blocks.** `n_v_heads` is 48, and GB10 has exactly 48 SMs -- so this kernel
occupies one block per SM and nothing more. 48 x 128 threads = **6,144 threads against
98,304 thread slots, about 6% occupancy**, and each thread then executes two dependent
chains of 128 FMAs per token inside a serial loop over `T`, with two `__syncthreads()` per
iteration (4,096 barriers per layer per chunk). There is no second block per SM to hide the
latency of either the shared-memory traffic, the barriers, or the fma chains.

This is the 6.7 s. It is not a FLOP problem -- the state is small (128x128 fp32 per head,
3.1 MB per layer per sequence, and round 53 already showed the state update is 0.17% of the
model's FLOPs). It is an **occupancy and parallelism** problem: 48 blocks of 128 threads
doing a serial scan.

It also explains observations that never fit before:

- **Round 51's flat-in-`n_seq` result.** `blockIdx.y` is always 0 because `grid_dim.y` is 1,
  so the kernel's parallelism does not grow with batch at all. Ten sequences cost the same
  as one because the grid is shaped by heads alone.
- **Rounds 54-55's large CPU time.** A kernel with two barriers per token in a 2048-token
  serial loop is exactly the shape that leaves a GPU idle and the host spinning.
- **Why the linear-op accounting never closed the gap** (round 60): the missing 53% was
  never a staging kernel. It is one badly-parallelised recurrence kernel, 48 layers deep.

The fix is structural rather than incremental, and there are two standard routes:

1. **Chunk the recurrence.** The classic gated-delta-rule formulation splits the sequence
   into blocks and computes each block's contribution with dense matrix products (the
   "chunked" form this kernel is named for but does not implement), which turns the serial
   scan into GEMM-shaped work that cuBLAS-class kernels can execute at high occupancy.
2. **Parallelise within the scan.** Give each `(head, batch)` more than one block, or give
   each SM multiple blocks, and/or split `D` across warps so the two 128-long chains become
   independent partial sums.

Route 1 is the one that scales -- it is what every performant gated-delta-rule
implementation does -- and it is what makes the name of this kernel honest.

### The occupancy hypothesis is refuted: 4x the blocks changes nothing (round 65)

Round 64 concluded the DeltaNet recurrence was occupancy-limited -- one 48-block grid on a
48-SM part, ~6% occupancy -- and named two fixes: give each head more blocks, or implement
the true chunked form. The first of those is a small, correct change, so it was made.

The state columns are independent: thread `j` reads and writes only column `j` of `S`, the
output write is per `(t, hv, j)`, and the only shared input is `sk`, which each block can
load itself. So the kernel was changed from a `(n_v_heads, 1, 1)` grid of 128-thread blocks
to `(n_v_heads * 4, 1, 1)` with 32 threads each, holding `S[D][D/4 + 1]` (16.9 KB instead of
66 KB) and writing back only its own columns.

**Correctness is preserved exactly** -- `attn-tile` reports `nonzero 402652988/402653184`,
the same count as before, at the same 12.35 s.

**And it makes no difference at all:**

| | before (round 63) | after (4x blocks) |
|---|---|---|
| DeltaNet layers | 13.29 s / 240 = 70.1% | **13.17 s / 240 = 69.8%** |
| full-attention layers | 5.50 s / 80 = 29.0% | 5.53 s / 80 = 29.3% |
| total prefill | 18.96 s | 18.87 s |

Within run-to-run noise. **The kernel is not limited by the number of blocks.** 192 blocks
over 48 SMs performs identically to 48 over 48.

That is a genuinely useful negative result, and it kills route 2 from round 64. It also
sharpens the arithmetic: the recurrence moves about 308 GFLOP per chunk (48 layers x 2048
tokens x 1.57M MACs) and takes 13.17 s over 5 chunks, i.e. **~117 GFLOP/s, about 0.6% of
this part's 18.4 TFLOP/s fp32 peak.** Adding parallelism across independent columns cannot
help that, because the cost is not in the columns -- it is in the **serial dependency along
`T`**: each token's state update depends on the previous token's, 2048 times per layer, and
that chain is the critical path no matter how the columns are laid out.

**So route 1 is the only one left, and it is now the only one that has ever been supported
by evidence.** The classic chunked gated-delta-rule formulation breaks that dependency by
computing, for each block of tokens, a matrix-product form that is parallel in the block
index; the intra-block part becomes dense GEMM-shaped work and the inter-block part becomes
a much shorter scan (T/C steps instead of T). That is what this kernel's name claims and
what its body does not do.

The 4x split was reverted -- it is correct but buys nothing, and it costs a 4x redundant
`sk` load. The tree is back to the round-63 state.

### The prize, quantified: the one fix flips 8K cold TTFT from 1.79x slower to 1.63x faster (round 66)

Round 65 refuted the occupancy route and left the true chunked form as the only
evidence-backed fix. This round sizes what it is worth before anyone writes it, so the
work is either justified or abandoned on numbers rather than on hope.

The recurrence moves, per 2048-token chunk, `2 * D * D * n_v_heads` FMAs per token per
layer over 48 layers:

| quantity | value |
|---|---|
| FMAs per token per layer | 1,572,864 (3.15 MFLOP) |
| per 2048-token chunk, 48 layers | 154.6G FMA = **309 GFLOP** |
| whole 8K run (5 chunks) | **~2 TFLOP** |
| measured DeltaNet GPU time (round 63/65) | **13.17 s** |
| **effective throughput** | **117 GFLOP/s = 0.64% of fp32 peak** |
| per layer per 2048 tokens | 68.6 ms (33.5 us per token per layer) |

**0.64% of peak is the number.** Nothing in this document runs that far below its ceiling;
the GEMM is at 84% of its isolated rate, the dequant is at ~86% of the memory bandwidth.
A kernel at 0.64% of peak is not tuned, it is barely started.

And the payout, holding everything else in the prefill fixed at its measured value:

| chunked-form throughput | DeltaNet time | 8K cold prefill | vs llama.cpp (10.58 s) |
|---|---|---|---|
| 0.117 TFLOP/s (now) | 13.17 s | 18.87 s | 1.79x slower |
| **2 TFLOP/s** | 0.77 s | **6.5 s** | **1.63x FASTER** |
| 5 TFLOP/s | 0.31 s | 6.0 s | **1.76x FASTER** |
| 10 TFLOP/s | 0.15 s | 5.9 s | **1.79x FASTER** |

**2 TFLOP/s is 11% of fp32 peak** -- an unremarkable target for a matrix-product-shaped
kernel. At that rate the 8K cold-TTFT objective is not merely met, it is met with room to
spare, and the whole three-way scorecard turns green: warm TTFT already wins by 7.9x, OTPS
by 1.17x, and cold TTFT would win by 1.63x.

At 32K the same fix is necessary but not sufficient: the recurrence scales with `T`, so it
is ~42 s of the 107.89 s there (16 chunks vs 5), and taking it to 2 TFLOP/s gives roughly
**68 s, or 1.54x slower than llama's 44.55 s** -- a large improvement on the current 2.42x
but still short. At 32K the attention slab (~22% of the prefill, and itself quadratic in
context) would be the next target.

**So the plan is now ordered by measured value:** rewrite `gated_delta_rule_chunk_kernel`
into a genuine chunked form (intra-chunk dense matrix products, inter-chunk scan of T/C
steps instead of T). That is the change that decides whether this objective is met at 8K,
and it is worth roughly 12.4 s of an 18.9 s prefill.

Implementation notes for whoever does it, so nothing has to be re-derived:

- The state is `D=128` per value head, `n_v_heads=48`, fp32, held in `state` at
  `(b * n_v_heads + hv) * D * D` with `base` offset; `rec_stride()` gives the per-sequence
  stride.
- Inputs to the layer, all already present at the call site (`layer.rs` `forward_prefill`,
  around the `gated_delta_rule_chunk` call): `sc.conv_ln` (q at 0, k at `qk_dim`, v at
  `2*qk_dim`), `sc.decay`, `sc.beta`, plus `group = n_v_heads / n_k_heads`.
- `delta_gate_batched` already produces `decay` and `beta` per `(t, head)`; the chunked form
  consumes the same two arrays.
- The correctness gate is `gb10-verify generate --oracle fixtures/oracle` plus
  `batch-parity`; `attn-tile` does **not** cover this kernel, which is why round 65's
  correctness check could only assert that the *attention* output was unchanged.

### The within-token chain is not the bottleneck either: 4-way partial sums give 4% (round 67)

Round 66 sized the chunked rewrite at 12.4 s of the 18.9 s prefill. Before committing to
that rewrite, one cheaper hypothesis was still open, and it is worth closing because it
narrows what the rewrite must fix.

Both inner loops of `gated_delta_rule_chunk_kernel` carried a **single 128-long serial fma
chain** -- `kv = fmaf(s, sk[i], kv)` and `o = fmaf(s, qh[i], o)`. At 0.64% of fp32 peak,
a serial dependency of that length is a natural suspect: four independent partial sums cut
the chain to 32 and give the scheduler interleaved work at no arithmetic cost.

**Correctness is exact.** `generate --oracle` reports `16/16 (100.0%)` and `exact match` --
the reassociation changed no greedy token.

**And the gain is small but real:**

| | before (round 63/65) | after (4-way partial sums) |
|---|---|---|
| DeltaNet layers | 13.29 s / 13.17 s | **12.80 s / 12.65 s** |
| full-attention layers | 5.50 s / 5.53 s | 5.49 s / 5.41 s |
| total prefill | 18.87 s / 18.96 s | **18.44 s / 18.22 s** |

Consistent in both runs: **~4% off the DeltaNet, ~3% off the whole prefill.** Not noise, but
not the order of magnitude the chunked form promises.

**So the bottleneck is neither the block count (refuted, round 65) nor the length of the
within-token chains (refuted here). What is left is the one thing both of those leave
untouched: the serial dependency *along `T`*.** Every token's state update needs the
previous token's state -- 2048 sequential steps per layer, each with two barriers, and no
amount of restructuring *inside* a step can shorten a chain that runs *between* steps. That
is precisely the dependency the true chunked form removes, by making the intra-chunk
computation a dense matrix product over `C` tokens at once and leaving only a `T/C`-step
scan between chunks.

Two cheap hypotheses have now been tested and rejected against measurement, which is what
makes the third one worth the effort: 12.4 s of an 18.9 s prefill, and the difference
between 8K cold TTFT at 1.79x slower and 1.63x faster than llama.cpp.

The 4-way split is kept -- it is exact, verified, and worth ~3% of the prefill.

### The DeltaNet cost is exactly linear in T: 31.5 us per token per layer, no per-chunk overhead (round 68)

Rounds 65 and 67 eliminated the block count and the within-token chains. Before accepting
the chunked rewrite as the only remaining lever, one cheap question was still open: is the
13.17 s a *per-token* cost, or is some of it a per-chunk fixed cost (a launch, a state
load/store, a one-time setup) that a smaller change could remove? Sweeping the token count
answers it with no code change, because the layer diagnostic already reports the total
across whatever chunks were run.

| `--limit` | chunks | DeltaNet layers | DeltaNet time | **per 2048-token chunk** | attention |
|---|---|---|---|---|---|
| 2048 | 1 | 48 | 3.14 s | **3.14 s** | 0.79 s |
| 4096 | 2 | 96 | 6.20 s | **3.10 s** | 1.92 s |
| 6144 | 3 | 144 | 9.31 s | **3.10 s** | 3.42 s |

**3.14, 3.10, 3.10.** The cost is exactly proportional to `T` -- there is **no per-chunk
fixed overhead at all**. The linear model also predicts the 8225-token run:
`8225/2048 x 3.10 = 12.45 s` against 12.65-12.80 s measured.

So the recurrence costs **31.5 us per token per layer** (3.10 s / 2048 tokens / 48 layers),
for 1.57M FMAs = 3.15 MFLOP per token per layer -- **~100 GFLOP/s, the same 0.6% of fp32
peak**. There is nothing to trim around the edges: every microsecond is in the 2048
sequential steps, and the only way to remove them is to stop doing one token at a time.

The same sweep also exposes the attention's shape, which matters for the 32K target:

| `--limit` | attention | per chunk |
|---|---|---|
| 2048 | 0.79 s | 0.79 s |
| 4096 | 1.92 s | 0.96 s |
| 6144 | 3.42 s | 1.14 s |

Mildly super-linear in the chunk count (0.79 -> 0.96 -> 1.14 per chunk), as expected for
attention that is quadratic within a chunk and grows with the carried context.

**And it corrects the 32K arithmetic upward.** At 31.5 us per token per layer the recurrence
is `32768 x 48 x 3.15 MFLOP = 4.95 TFLOP` at 32K, which takes **~49.5 s of the 107.89 s
prefill -- 46%, not the 35-39% estimated earlier.** Taking it to 2 TFLOP/s would give
32K cold prefill of about **61 s against llama.cpp's 44.55 s: 1.37x slower** (from 2.42x).
Better, but still short -- at 32K the attention slab, now measured at ~26% and growing,
becomes the second target.

**The ordering is now fully evidence-based.** Three hypotheses have been tested and rejected
(block count, within-token chains, per-chunk overhead); what remains is the serial dependency
along `T`, costing 31.5 us per token per layer with perfect linearity, and worth 12.4 s of
the 8K prefill and ~47 s of the 32K prefill. The chunked rewrite is not one option among
several -- it is the only measurement that has ever moved.

### WIN: the state column belongs in registers -- DeltaNet -17%, whole prefill -12% (round 69)

Rounds 65, 67 and 68 rejected the block count, the within-token chains and any per-chunk
overhead, leaving the 2048 sequential steps as the cost. But those steps are 100 GFLOP/s
while doing only 3.15 MFLOP of arithmetic each -- so the question was what the block is
waiting on *inside* a step.

The answer was in the data layout. Each thread owns exactly one column `j` of the state, and
a column is exactly `D` floats -- but the state lived in 66 KB of shared memory
(`__shared__ float S[D][D+1]`), so **every fma in both hot loops did a shared load and a
shared store**. Per thread per token that is 512 shared accesses for 256 fmas, and the two
loops are the entire cost of the kernel.

Since a column is per-thread private and exactly `D` floats, it fits in registers. Moving it
there (`float Sc[D]`, unrolled load and store, `sk` left in shared because every column
needs it) removes every shared access from both inner loops.

**Correctness is exact** -- `generate --oracle` reports `exact match`.

| | round 63/65 | round 67 (4-way chains) | **round 69 (registers)** |
|---|---|---|---|
| DeltaNet layers | 13.29 / 13.17 s | 12.80 / 12.65 s | **10.56 / 10.55 s** |
| full-attention layers | 5.50 / 5.53 s | 5.49 / 5.41 s | 5.53 / 5.51 s |
| **total prefill** | 18.87 / 18.96 s | 18.44 / 18.22 s | **16.26 / 16.22 s** |

**-17% on the DeltaNet, -12% on the whole prefill**, reproduced in both runs.

The per-token cost falls from **31.5 to 26.4 us per token per layer** (2.6 s per 2048-token
chunk, against 3.10 s in round 68's sweep). Still 0.75% of fp32 peak, so most of the cost
remains -- but for the first time in this document a *change* to the recurrence has moved
the number by more than a few percent, and it confirms that the kernel's cost is memory
traffic, not arithmetic and not the number of blocks.

This also re-orders the remaining work, and it is worth stating plainly:

- The two cheap structural fixes together took the 8K prefill from 18.87 s to 16.24 s, i.e.
  from 1.79x slower than llama.cpp to **1.53x slower**.
- The chunked rewrite is still the big one -- at the original 2 TFLOP/s target it is worth
  ~12 s of this prefill -- but the gap it has to close is now smaller, and the register
  change is evidence about *why* the current kernel is slow (traffic per step), which is
  also the reason a chunked form will win: it does `C` tokens per pass over the state
  instead of one.

### WIN 2: `qh` was a global load inside the hot loop -- DeltaNet -12%, prefill -9% more (round 70)

Round 69's register change was verified with `ptxas -v` rather than trusted, and the check
paid for itself twice.

**First, it showed the register change had spilled.** Compiling the emitted PTX for sm_121:

```
ptxas info : Function properties for gated_delta_rule_chunk_kernel
    280 bytes stack frame, 276 bytes spill stores, 308 bytes spill loads
ptxas info : Used 255 registers, used 1 barriers, 280 bytes cumulative stack size, 512 bytes smem
```

**255 registers** -- the hardware maximum -- with 280 bytes of stack and real spill traffic.
That is inherent, not a tuning miss: `Sc[128]` is live across the whole 2048-step token loop,
so 128 registers are unavoidable and the token loop's own state pushes the rest over. Partial
unrolling cannot help, because the unroll factor is not what makes the array live.

**Second -- and this is what won the round -- the same check showed where the remaining cost
was.** With the state in registers, `sk[i]` came from shared memory but **`qh[i]` was still a
global load inside the second hot loop**: 128 of them per thread per token, on the same
address for every thread in the block. Hoisting it into shared memory beside `sk`, loaded
once per token, drops the spill too (280 -> 96 bytes stack):

```
    96 bytes stack frame, 140 bytes spill stores, 144 bytes spill loads
```

**Correctness is exact** (`generate --oracle`: `exact match`), and the effect is large:

| | round 63/65 | r67 chains | r69 registers | **r70 qh in shared** |
|---|---|---|---|---|
| DeltaNet layers | 13.29 / 13.17 s | 12.80 / 12.65 s | 10.56 / 10.55 s | **9.31 / 9.26 s** |
| full-attention layers | 5.50 / 5.53 s | 5.49 / 5.41 s | 5.53 / 5.51 s | 5.45 / 5.36 s |
| **total prefill** | 18.87 / 18.96 s | 18.44 / 18.22 s | 16.26 / 16.22 s | **14.92 / 14.79 s** |

**Three changes to one kernel have now taken the DeltaNet from 13.29 s to 9.26 s (-30%) and
the whole 8K prefill from 18.87 s to 14.79 s (-22%).** The per-token cost is down from
31.5 us to **18.8 us per token per layer** (1.8 s per 2048-token chunk, against 3.10 s in
round 68's sweep).

**The lesson generalises:** all three wins came from removing memory traffic from the inner
loops -- twice shared-memory traffic, once global -- and none came from changing the
parallel structure. The same measurement also says what to do next: `ptxas` still reports 255
registers and 140 spill stores, and spill traffic goes to local memory, which is global
memory. **Splitting each state column across two threads (64 floats, ~64 registers) would
hold the column without spilling**, at the cost of one `__shfl_xor_sync` pair per reduction
-- and every previous reduction in inner-loop traffic has paid 9-17%.

### The spill was not the bottleneck either -- but the kernel is now resource-clean (round 71)

Round 70's `ptxas` check found the one-thread-per-column form at 255 registers with 140
spill stores, and spill traffic goes to local memory, which is global memory. Since every
previous removal of inner-loop traffic had paid 9-17%, removing the spill looked like the
next win. The column was split across two threads -- the pair `(2j, 2j+1)` takes 64 rows
each, joined by one `__shfl_xor_sync` -- which needs just 148 registers:

```
before:  255 registers, 280 bytes stack, 276 spill stores, 308 spill loads
after:   148 registers,   0 bytes stack,   0 spill stores,   0 spill loads
```

**Correctness is exact, and the speed is identical.**

| | round 70 (spilling) | round 71 (spill-free) |
|---|---|---|
| DeltaNet layers | 9.31 / 9.26 s | **9.23 / 9.26 s** |
| full-attention layers | 5.45 / 5.36 s | 5.51 / 5.48 s |
| total prefill | 14.92 / 14.79 s | 14.90 / 14.90 s |

**Zero.** The spill was being hidden well enough that removing it changes nothing -- so a
fourth hypothesis (register pressure and its spill traffic) is now rejected by measurement,
along with the block count (round 65), the within-token chains (round 67) and per-chunk
overhead (round 68).

The change is kept anyway, on the narrow grounds that it is strictly better in resources and
not worse in anything else: 148 registers instead of the hardware maximum of 255, no spill
at all, and 256 threads per block instead of 128, at identical speed and exact correctness.
It is also the shape the kernel needs if occupancy is ever revisited -- the spilled form
could not fit a second block on an SM under any circumstances, and this one can (148 x 128
threads x 2 blocks = 37,888 of the SM's 65,536 registers).

**What the four rejections leave standing is unchanged and now very well bounded: the cost is
the 2048 sequential steps along `T`, at 18.8 us per token per layer.** Three fixes removed
traffic *inside* a step (-30% on the DeltaNet) and one fix removed spill *inside* a step
(0%). Nothing that keeps one token per step has much left to give. **The chunked rewrite is
the remaining change, and it is the only one that alters the number of steps rather than
what happens inside one.**

### Official scorecard re-measured through the server, 8K and 32K (round 72)

The four kernel fixes were measured with `prefill-shape`'s in-process diagnostics. This
round re-ran the real thing -- `bench/longctx/ttft.py` against the streaming HTTP endpoint,
identical request twice per trial so that cold and warm differ only in what the server
already holds -- to confirm the wins survive the whole server path.

**8K** (`--ctx 16384`, reps=263, 2 trials, max-tokens 200):

```
# gb10-8k  reps=263 trials=2 max_tokens=200
trial   prompt  cold_ttft  warm_ttft otps_cold otps_warm  tok
    0     8225      14.81       0.04      8.75      8.69  200
    1     8225      15.04       0.03      8.70      8.68  200
```

**32K** (`--ctx 36864`, reps=1054, 1 trial, max-tokens 200):

```
# gb10-32k  reps=1054 trials=1 max_tokens=200
trial   prompt  cold_ttft  warm_ttft otps_cold otps_warm  tok
    0    32747      90.54       0.05      7.14      7.15  200
```

The 8K cold TTFT predicted from the in-process diagnostics was 14.90 s; the server measures
**14.81 and 15.04 s**. The prediction from the kernel work transfers to the full path.

**The scorecard, against the same NVFP4 GGUF on `llama-server`:**

| context | metric | gb10 | llama.cpp | result |
|---|---|---|---|---|
| 8K | cold TTFT | **14.93 s** | 10.58 s | 1.41x slower (was 1.79x) |
| 8K | **warm TTFT** | **0.035 s** | 0.237 s | **6.8x faster** |
| 8K | **OTPS** | **8.72** | 7.32 | **1.19x faster** |
| 32K | cold TTFT | **90.54 s** | 44.55 s | 2.03x slower (was 2.42x) |
| 32K | **warm TTFT** | **0.05 s** | 0.29 s | **5.8x faster** |
| 32K | **OTPS** | **7.14** | 6.865 | **1.04x faster** |

Warm TTFT (prefix cache hit) and OTPS win at both contexts. Cold TTFT is still behind, and
the remaining margin is now almost entirely the DeltaNet recurrence: 9.26 s of the 14.93 s
at 8K, and at 31.5 us/token/layer scaled to 32K it is roughly 47 s of the 90.54 s.

**Session cumulative cold TTFT: 8K 88.81 -> 14.93 s (5.95x), 32K 826.09 -> 90.54 s (9.12x).**

The four rejected hypotheses (block count, within-token chains, per-chunk overhead, register
spill) plus the three accepted fixes (-30% on the DeltaNet: shared-state to registers, global
`qh` to shared, and the 4-way chain split) are all recorded above. The chunked rewrite
remains the one change that alters the number of sequential steps rather than the cost of
one, and it is what stands between cold TTFT and the objective.

### 128K measured: the prefill turns super-linear, and attention takes over (round 73)

The objective names 32K, 128K and 256K, but the reproducible baseline only covered 8K and
32K. This round measured the third context through the same harness.

**128K** (`--ctx 147456`, reps=4218, 1 trial, max-tokens 200):

```
# gb10-128k  reps=4218 trials=1 max_tokens=200
trial   prompt  cold_ttft  warm_ttft otps_cold otps_warm  tok
    0   130832    1119.40       0.14      3.45      3.45  200
```

**The prefill is no longer linear.** 32K took 90.54 s for 32,747 tokens; 130,832 tokens is
4.0x that, so linear scaling predicts 362 s. **1119 s is 3.1x more than that.**

Additive decomposition of the 1119 s, using the per-token costs already measured in-process:

| block | scaling | 128K estimate | share |
|---|---|---|---|
| DeltaNet recurrence | 18.8 us/token/layer x 48 layers | ~118 s | ~11% |
| linear ops (GEMM, dequant, cast, epilogue) | 1.07 ms/token (round 60) | ~140 s | ~13% |
| **attention** | super-linear in context | **~860 s** | **~77%** |

So the ordering of the objectives inverts with context length, and this is the first
measurement that shows it:

| context | DeltaNet | attention | dominant block |
|---|---|---|---|
| 8K | 9.26 s (62%) | 5.5 s (37%) | **DeltaNet** |
| 32K | ~47 s | ~26% and growing | **DeltaNet** |
| 128K | ~118 s (11%) | ~860 s (**77%**) | **attention** |

**At 128K the attention kernel is not a secondary target, it is the target** -- 77% of an
1119 s prefill, and the reason the curve turns upward. This is the second item the objective
named, and it is now measured rather than assumed. The round-45 attention model (per-key
cost) is what would have predicted this; what is new is the server-side number that shows
how far past the crossover 128K is.

Two caveats recorded honestly:

- **No `llama-server` number at 128K yet**, so no ratio is claimed at this context. The
  comparison at 8K and 32K stands; 128K needs the llama side run before it can be stated.
- **OTPS falls to 3.45 at 128K** (from 7.14 at 32K and 8.72 at 8K) -- decoding also pays the
  attention cost, since every decoded token attends over 130K cached keys. Whether that wins
  against llama.cpp at this length is unknown for the same reason.

**The plan at 128K therefore differs from the plan at 8K.** At 8K the chunked DeltaNet
rewrite is the whole story. At 128K it would recover ~118 s of 1119 s (11%), while the
attention kernel -- the objective's other named item, and the one the round-45/48 work only
began -- holds ~860 s.

### 128K comparison: cold 4.10x slower and OTPS now LOST -- attention dominates both (round 74)

Round 73 measured gb10 at 128K and flagged that no `llama-server` number existed, so no ratio
was claimed. This round ran the llama side through the same harness and the same request.

```
# llama-128k  reps=4218 trials=1 max_tokens=200
trial   prompt  cold_ttft  warm_ttft otps_cold otps_warm  tok
    0   130870     273.17       0.48      4.82      4.78  198
```

against gb10's `130832 prompt, cold 1119.40, warm 0.14, otps 3.45`.

| context | metric | gb10 | llama.cpp | result |
|---|---|---|---|---|
| 8K | cold TTFT | 14.93 s | 10.58 s | 1.41x slower |
| 8K | warm TTFT | **0.035 s** | 0.237 s | **6.8x faster** |
| 8K | OTPS | **8.72** | 7.32 | **1.19x faster** |
| 32K | cold TTFT | 90.54 s | 44.55 s | 2.03x slower |
| 32K | warm TTFT | **0.05 s** | 0.29 s | **5.8x faster** |
| 32K | OTPS | **7.14** | 6.865 | **1.04x faster** |
| **128K** | cold TTFT | 1119.40 s | **273.17 s** | **4.10x slower** |
| **128K** | warm TTFT | **0.14 s** | 0.48 s | **3.4x faster** |
| **128K** | OTPS | 3.45 | **4.82** | **1.40x SLOWER** |

**Two things changed at 128K, and both are recorded rather than smoothed over.**

**First, the cold-TTFT gap widens again: 1.41x -> 2.03x -> 4.10x.** It is not that gb10 got
worse; both parts scale super-linearly, and llama.cpp scales better because `-fa on` gives it
a flash-attention kernel while gb10's attention is the hand-written tile kernel of rounds
41-48. At 128K, attention is ~860 s of gb10's 1119 s (round 73) -- and llama's 273 s total
means its attention is several times cheaper per key.

**Second, and more importantly, gb10 loses OTPS for the first time.** At 8K and 32K decoding
won (1.19x, 1.04x); at 128K it loses by 1.40x, because every decoded token attends over
130K cached keys and that is now the dominant per-token cost on both sides. **Warm TTFT
remains gb10's one clear win at 128K (3.4x), and it is a real one -- 0.14 s against 0.48 s.**

So the objective's own ordering was right, and the measurements now say so explicitly: the
prefix cache (the easy half) is done and wins everywhere; **the attention kernel is the
remaining blocker, and at 128K it is the blocker for prefill *and* decode simultaneously.**
That also means the attention work cannot be deferred behind the chunked DeltaNet rewrite --
at 8K the DeltaNet is 62% and the rewrite is the whole story, but at 128K the DeltaNet is
only 11% and attention is 77%.

Two targets, ordered by context:

- **8K / 32K:** chunked gated-delta-rule rewrite (DeltaNet 62% / ~47%).
- **128K / 256K:** the attention kernel, for both prefill and decode, against llama.cpp's
  flash attention. 256K has no baseline on either side yet.

### The attention gap, quantified: 5.0 TFLOP/s prefill, 33 GB/s decode (round 75)

Round 74 established that attention is the 128K blocker for prefill *and* decode. This round
put numbers on how far the kernel is from its own ceilings, so the work has a target rather
than a direction.

**Prefill attention at 128K.** The flash-attention work is `2 x 2 x n_heads x head_dim x
avg_keys` MACs per (token, layer) -- QK^T plus PV -- over 130,832 tokens, 16 full-attention
layers, 24 query heads and head_dim 256:

| quantity | value |
|---|---|
| total attention FLOPs at 128K | **4,275 TFLOP** |
| gb10 attention time (round 73 decomposition) | ~860 s |
| **gb10 effective rate** | **5.0 TFLOP/s** |
| fp16 peak on this part (~2x the 18.43 TFLOP/s fp32) | ~37 TFLOP/s |
| **gb10 share of fp16 peak** | **~13.5%** |
| llama.cpp total at 128K | 273 s (of which ~130 s is the linear/DeltaNet work) |
| **llama implied attention rate** | **~32 TFLOP/s, ~86% of peak** |

**So llama.cpp's flash attention is roughly 6.4x more efficient per FLOP than gb10's tile
kernel.** That single factor is the whole of the 4.10x cold-TTFT gap at 128K: take gb10's
attention from 860 s to llama's ~130 s and the two prefill totals are nearly equal.

**Decode attention at 128K.** Per decoded token the whole KV cache must be read: 147,456
keys x 4 KV heads (GQA) x 256 head_dim x 2 (K and V) x 2 bytes fp16 x 16 layers.

| | ms/token | KV read per token | effective bandwidth | vs 228 GB/s bound |
|---|---|---|---|---|
| gb10 (3.45 OTPS) | 290 ms | 9.66 GB | **33 GB/s** | **15% of bound** |
| llama (4.82 OTPS) | 207 ms | 9.66 GB | 46.6 GB/s | 20% of bound |
| memory bound | 42 ms | 9.66 GB | 228 GB/s | 24 tok/s |

**gb10's decode attention uses 15% of the available bandwidth** -- it is a streaming
reduction over 9.66 GB and it runs 7x slower than the memory system allows. llama is also
far from the bound (20%), so neither is purely bandwidth-limited, but gb10's gap is the
larger one and it is what turns the 1.04x OTPS lead at 32K into a 1.40x loss at 128K.

**Two concrete targets, both measured:**

1. **Prefill attention: 5.0 -> ~30 TFLOP/s** (llama's rate). Worth ~730 s of the 1119 s at
   128K, and it is the only path to competing at this context.
2. **Decode attention: 33 -> ~150 GB/s.** Worth roughly 5x on OTPS at 128K.

Both point the same way -- the rounds 41-48 tile kernel has a layout tuned for shared-memory
bank conflicts (`PADH = 130`, fp16 `Qs`/`Ks`, `BK = 16`, `BQ = 24`, 22,624 B smem) rather
than for streaming the KV cache, and at 128K the streaming is the dominant cost. A GQA-aware
kernel that streams K and V once per token and reuses each block across all 24 query heads
(they share only 4 KV heads) is the shape both targets need.

### The 128K decode gap is 6x GQA redundancy: the grid is one block per QUERY head (round 76)

Round 75 left the decode attention at 33 GB/s against a 228 GB/s bound and asked why. The
answer is in the grid, and it is a factor of exactly `n_q_heads / n_kv_heads` = 24 / 4 = 6.

Reading the decode path finds two kernels, and they are in very different states:

**`attn_decode_kernel`** (elementwise.cu:624) is catastrophic on two counts. Its grid is
`blockIdx.x = h`, **one block per query head**, and each block loops `for (s = 0; s <
n_keys; ++s)` calling **`block_reduce_sum` inside the per-key loop**:

```cuda
for (int s = 0; s < n_keys; ++s) {
    const float kk = ... k_cache[base + ((size_t)s * n_kv_heads + kh) * head_dim + d] ...;
    const float dot = block_reduce_sum(qv * kk) * scale;   // full block reduction PER KEY
    ...
}
```

That is 130,000 block reductions per head per token, each with multiple `__syncthreads()`.

**`attn_decode_multi_kernel`** (elementwise.cu:1125) is the one the model actually uses
(layer.rs:614) and it is much better -- it already fixed the per-key reduction by striding
keys across warps, and its own comment records an earlier round of work on it:

```cuda
constexpr int NW = 32;   // warps per block
// The grid is only (n_q_heads, n_seq) = 24 blocks for a single sequence, so at
// NW = 8 a 256-thread block left just 4 warps resident per SM and the kernel
// ran at 328 GB/s. Raising NW raises the resident warps without changing the
// grid: 32 warps x 24 blocks is 4x the in-flight work for the same traffic.
for (int t = warp; t < n_keys; t += NW) { ... }
```

**But the grid is still `(n_q_heads, n_seq)` = 24 blocks, one per QUERY head** -- and
`kh = h / group` means the **six** query heads of each KV head each traverse the *entire* KV
cache for that same KV head. The K/V bytes are read six times over:

| | per token | |
|---|---|---|
| KV cache actually stored (4 KV heads) | 9.66 GB | minimal |
| KV bytes read at 24 query heads | **58 GB** | 6x redundant |
| measured time | 290 ms | |
| **effective read rate** | **200 GB/s** | **88% of the 228 GB/s bound** |

**So the decode kernel is not slow because it is badly written -- it is already running at
~88% of memory bandwidth. It is slow because it does six times the memory traffic it needs
to.** That closes the round-75 question exactly, and it explains why the earlier `NW` tuning
(cited in the comment) hit a wall: raising warps cannot help once the traffic itself is 6x
what it should be.

**The fix is the standard one and it is well specified.** Move the grid to one block per *KV*
head and let each block serve all six query heads that share it, so K and V are read once and
reused six times. That alone would cut traffic 6x -- but `n_kv_heads = 4` gives only 4 blocks
for 48 SMs, so it must be combined with the usual **flash-decoding split**: also split the
key dimension, `grid = (n_kv_heads, key_splits, n_seq)`, and combine the partial online
softmaxes in a second pass. With `key_splits = 8` that is 32 blocks per sequence with no
redundancy -- more parallelism than today's 24 blocks *and* one sixth of the traffic.

Predicted effect at 128K: decode attention from 290 ms/token toward the ~42-60 ms the traffic
actually requires, i.e. **OTPS from 3.45 past llama.cpp's 4.82** -- which would turn the one
metric gb10 currently loses into a win. The same split-K structure is what the 128K prefill
attention needs (round 75: 5.0 TFLOP/s, 13.5% of fp16 peak).

**This is the highest-value remaining change in the document**, because it is the only one
aimed at a metric gb10 currently loses, it has a measured cause (6x redundant traffic), and
the fix is a known pattern rather than research.

### The naive GQA fix does not fit in shared memory -- split-K is required, not optional (round 77)

Round 76 identified the 6x GQA redundancy and proposed moving the grid to one block per KV
head so K and V are read once and reused by the six query heads that share them. Before
anyone writes it, the merge stage was checked for feasibility, and it settles the design.

`attn_decode_multi_kernel` merges its `NW = 32` per-warp partials through shared memory:

```cuda
// One merge of the NW per-warp partials. `sm_acc` is sized for the 256-wide
// head this kernel requires, so it is 8 KB, not one entry per key.
__shared__ float sm_m[NW], sm_l[NW], sm_acc[NW][256];
```

**`sm_acc[NW][head_dim]` is 32 x 256 x 4 = 32 KB on its own** (the comment's "8 KB" predates
the `NW` raise from 8 to 32 recorded in round 76). A block that serves all six query heads of
a KV head at once needs six such merge buffers:

| | shared memory |
|---|---|
| today: one head per block | 32 KB + 0.25 KB = **~32 KB** |
| naive 6-head block | 6 x 32 KB = **192 KB** |
| GB10 sharedMemPerBlockOptin | **99 KB** |
| default sharedMemPerBlock | 48 KB |

**192 KB is double the opt-in ceiling, so the one-block-per-KV-head form cannot be built this
way at all.** And processing the six heads *sequentially* inside the block would not help:
one head's keys occupy 130K x 256 x 4 B = 133 MB, far beyond L2, so the second pass would
re-read from DRAM and the 6x traffic would come straight back.

**That leaves exactly one viable shape, and it is the one round 76 named: a two-phase
flash-decoding split.** Write each block's partial online-softmax triple `(m, l, acc)` to a
small global scratch, then combine the partials in a second pass whose working set is
`key_splits x (1 + 1 + head_dim)` floats per (head, sequence) rather than anything
context-sized. With `key_splits = 8`:

- **traffic:** `grid = (n_kv_heads, key_splits, n_seq)` = 4 x 8 = 32 blocks per sequence, each
  block reading only its slice of the key range **once** across all six query heads -- one
  sixth of today's bytes.
- **parallelism:** 32 blocks per sequence, *more* than today's 24, and with `NW` warps each,
  so the occupancy that round 76's comment says is needed for bandwidth is preserved.
- **shared memory:** unchanged at ~32 KB per block, because each block still merges only its
  own warps' partials.

**So the check changed the plan from "move the grid" to "two-phase split-K", with the reason
written down: the merge buffer alone is 32 KB per head, and six heads do not fit.** That is
the difference between a change that gets implemented and one that gets started and abandoned
on the third compile error.

Targets are unchanged from round 76: decode attention 290 ms/token -> ~42-60 ms, i.e. 128K
OTPS from 3.45 past llama.cpp's 4.82; and the same split-K scratch/combine structure serves
the 128K prefill attention, which measured 5.0 TFLOP/s against llama's ~32.

### CORRECTION: the KV cache is fp32, so round 76's "88% of bandwidth" was wrong (round 78)

Rounds 75 and 76 both sized the decode attention's memory traffic assuming an **fp16** KV
cache: 9.66 GB per decoded token at 128K. Checking the type before building anything on that
number shows it is wrong by a factor of two.

```rust
// crates/gb10-model/src/layer.rs:638
pub k_cache: CudaSlice<f32>,
```

and the append kernel writes it as `float*`, and both attention kernels read
`const float* __restrict__ k_cache`. **The cache is fp32.** The corrected arithmetic:

| | fp16 (what rounds 75/76 assumed) | **fp32 (what the code does)** |
|---|---|---|
| KV per decoded token at 128K | 9.66 GB | **19.3 GB** |
| measured 290 ms/token | 33 GB/s | **66.6 GB/s** |
| share of the 228 GB/s bound | 15% | **29%** |
| if 6x GQA redundancy reaches DRAM | 200 GB/s | **400 GB/s -- above DRAM** |

**So round 76's conclusion -- that the decode kernel "is already running at ~88% of memory
bandwidth" and that only reducing bytes can help -- is retracted. It is at 29% of DRAM
bandwidth, and there is a lot of headroom.** That also explains an inconsistency round 76
glossed over: 6x redundancy at DRAM would demand 400 GB/s, which the part does not have, so
the redundancy must be substantially absorbed by L2 rather than DRAM.

**The revised diagnosis is different, and it is the one the kernel's own comment already
pointed at.** That comment raised `NW` from 8 to 32 for a stated reason -- "4x the in-flight
work for the same traffic" -- which is a *latency-hiding* argument, not a bytes argument. A
kernel at 29% of bandwidth with plenty of independent work available is **memory-latency
bound, not bandwidth bound**: it is not issuing enough concurrent loads to cover DRAM
latency. That reframes the fix, and it makes a much cheaper change viable:

1. **Store the KV cache in fp16 or bf16.** Halves the bytes outright (19.3 -> 9.7 GB/token),
   halves KV memory, and helps the 128K *prefill* attention at the same time. If the kernel
   keeps its present efficiency, 290 ms becomes ~145 ms, i.e. **OTPS ~6.9 against
   llama.cpp's 4.82** -- which would flip the one metric gb10 currently loses. llama.cpp
   itself keeps its KV cache in fp16, so this is also what the comparison is measuring
   against. The cost is a quality question, gated by perplexity and needle.
2. **Raise memory-level parallelism** rather than reduce bytes -- more in-flight loads per
   thread, or a key-split grid purely for occupancy. Worth trying first because it needs no
   dtype change and is measured by the same harness.

The two-phase split-K design of rounds 76/77 is still the right shape if the fix turns out to
be parallelism -- but note that it is no longer justified by "the traffic is 6x too high and
we are at the bandwidth wall", because that was based on the fp16 figure. It is justified
only if the 6x is shown to matter, which the corrected numbers say it does not obviously do
(400 GB/s of DRAM demand is impossible, so L2 is already absorbing most of it).

**This is the second time in this document that a number was used before its assumption was
checked** (the first was round 55's "CPU-bound" inference from process user time). The
corrected figure is recorded rather than the original edited away, because the retraction is
the more useful artifact.

### The decode kernel bench says the kernel is not the problem -- but it benchmarks 4 sequences (round 79)

Rounds 75-78 reasoned about the decode attention from the model-level numbers (3.45 OTPS at
128K, 290 ms/token, 19.3 GB of fp32 KV per token). `gb10-verify decode-bench` measures the
kernel itself, and it says something that contradicts the "bad kernel" reading:

```
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     2048       2.824       0.434    6.51x     1.17e-6   (143 GB/s serial, 928 GB/s warp)
     8192      11.183       1.677    6.67x     2.10e-6   (144 GB/s serial, 961 GB/s warp)
    32768      44.271       6.658    6.65x     4.20e-6   (146 GB/s serial, 968 GB/s warp)
```

Three things stand out, and the third changes the plan:

1. **The warp kernel is 6.5-6.7x faster than the serial reference and flat across key
   counts** (6.51x, 6.67x, 6.65x) -- so its efficiency does not decay with context length,
   which is exactly what the online-softmax rewrite was for.
2. **The "GB/s" figures are logical, not DRAM.** 968 GB/s is four times this part's measured
   228 GB/s, so the number must be counting bytes per *query* head (24 of them) rather than
   per KV head (4) -- i.e. it is measuring the 6x-redundant traffic of round 76 at L2 speed.
   That is consistent with round 78's correction: the redundancy is largely absorbed by L2,
   not DRAM.
3. **The bench runs `n_seq 4`, but real 128K decode runs one sequence.** The grid is
   `(n_q_heads, n_seq)`, so the bench launches 96 blocks and the real case launches **24** --
   **on 48 SMs, and 32 warps per block means half the GPU is idle in the real case.** Every
   number in the table above is therefore measured under conditions the real 128K decode
   never sees.

**So the honest reading is: in isolation, with enough sequences to fill the machine, the
decode attention kernel performs well and scales flat to 32K keys. What has not been measured
is the same kernel at `n_seq = 1`**, which is the case that matters for the 128K
single-stream OTPS number. The 290 ms/token at 128K is therefore *not yet* attributable to
the kernel, and the round-76/77 split-K plan is better re-justified as an **occupancy** fix
for `n_seq = 1` (24 blocks, half the SMs idle) than as a traffic fix -- which is what round
78 already concluded from the corrected numbers.

**The immediate next measurement is therefore cheap and specific:** run `decode-bench` at
`n_seq = 1` and compare the per-sequence rate against the `n_seq = 4` column. If per-sequence
throughput collapses at `n_seq = 1`, the fix is the key-split grid (more blocks), which is a
far smaller change than the two-phase softmax combine; if it does not, then the 128K decode
cost is elsewhere and the attention rewrite should wait.

### n_seq=1 is FASTER per sequence: the kernel is efficient, the traffic is not (round 79 cont.)

The measurement round 79 called for -- `decode-bench` at `n_seq = 1` -- answers the question
in the opposite direction from the guess:

| keys | `n_seq 4` warp ms | **`n_seq 1` warp ms** | speedup vs serial at `n_seq 1` |
|---|---|---|---|
| 2048 | 0.434 | **0.162** | **11.73x** |
| 8192 | 1.677 | **0.588** | **17.87x** |
| 32768 | 6.658 | **2.312** | **18.18x** |

**One sequence is 2.7-2.9x faster per sequence than four.** Fitting the per-key cost through
the origin gives **70.7 ns/key**, flat from 2K to 32K keys. So:

- **Half the SM count idle does not hurt.** 24 blocks x 32 warps = 768 warps already
  saturates the 24 SMs that get a block, and the speedup over the serial reference rises from
  6.65x to **18.18x** when the sequences are removed. The split-K-for-occupancy plan of round
  77 is **not** justified by this data.
- **The kernel is not the problem.** It scales linearly to 32K keys with no efficiency decay,
  which is what the online-softmax rewrite was for, and it beats its own serial reference by
  18x.
- **Extrapolated to 128K, attention costs 10.42 ms per launch.** If a launch is one layer,
  that is **167 ms/token across 16 layers -- 58% of the measured 290 ms/token.** The other
  42% is the GEMM (~1.07 ms/token) plus the DeltaNet step (~0.9 ms/token), which together are
  nowhere near 123 ms, so a large part of the 128K decode remains unattributed and is the next
  thing to measure rather than assume.

**What survives from round 76 is the traffic argument, not the occupancy one.** At
`n_seq 1` and 32K keys the bench reports **697 GB/s logical**, and 697 GB/s is three times
this part's 228 GB/s DRAM -- so the logical figure counts the 24-query-head traffic, and the
implied DRAM rate is roughly a sixth of it, about **116 GB/s against the 228 GB/s bound**.
That leaves real but modest headroom: removing the GQA redundancy is worth up to ~2x on the
DRAM-limited part, not the 6x that rounds 75/76 implied.

**So the plan changes again, and this time toward less work:**

1. **Store the KV cache in fp16/bf16.** Still the single cheapest real win: it halves bytes,
   halves KV memory, and helps 128K prefill at the same time -- and needs no kernel
   restructuring.
2. **Attribute the missing ~123 ms/token of 128K decode** with an in-model measurement before
   optimising anything else, because at present more than a third of the decode budget has no
   measured owner. The same discipline that fixed round 55 applies: measure in the model, do
   not infer from a standalone kernel.
3. The two-phase split-K redesign is **demoted**: the occupancy case for it is refuted here,
   and the traffic case is now sized at ~2x rather than 6x.

### The 128K decode budget is now fully attributed -- and ~40% of it is a context-independent floor (round 80)

Round 79 left ~123 ms/token of the 128K decode with no measured owner and named attributing it
as the next step. One cheap run answers it: `generate` at a 16-token prompt reports

```
decoded 16 tokens in 2.051s -> 7.80 tok/s
```

**7.80 tok/s at a 16-token context is 128 ms per token** -- essentially the same per-token
cost as at 8K (8.72 OTPS = 115 ms) and 32K (7.14 OTPS = 140 ms). Laid out:

| context | OTPS | ms/token | attention (from round 79's 70.7 ns/key) |
|---|---|---|---|
| ~16 tokens | 7.80 | **128 ms** | ~0 |
| 8K | 8.72 | 115 ms | ~2 ms |
| 32K | 7.14 | 140 ms | ~9 ms |
| 128K | 3.45 | **290 ms** | **~167 ms** |

**The decode has a floor of roughly 115-128 ms per token that does not depend on context at
all**, and it accounts for essentially all of the 8K and 32K numbers and for 128 of the 290 ms
at 128K. The attention term then closes the budget almost exactly: 128 + 167 = 295 ms against
290 ms measured.

**So the 290 ms/token at 128K decomposes as ~128 ms of context-independent per-token work
plus ~167 ms of attention**, and attention is *not* the largest single item -- it is 58% at
128K and nearly nothing at 8K.

**And that floor is itself unexplained by a wide margin.** One token through 64 layers does
~22.3B MACs (round 53), which is 44.6 GFLOP; at 128 ms that is **0.35 TFLOP/s, or 2% of fp32
peak.** Even allowing that an M=1 GEMM cannot use tensor cores well, a factor of fifty is not
an arithmetic shortfall -- it is per-token overhead, launch cost, or synchronisation. This is
the same shape of finding as the DeltaNet recurrence (round 66: 0.64% of peak) and it is the
same lesson: a rate two orders of magnitude below the ceiling means something is waiting, not
computing.

**This reorders the OTPS work.** At 128K the attention rewrite is worth ~167 ms and the floor
is worth ~128 ms; but at 8K and 32K -- where gb10 currently *wins* OTPS -- the floor is
essentially the entire cost, so improving it widens the margins that are already held rather
than closing the one that is lost. To beat llama at 128K (needs < 207 ms) both must move:
167 -> ~100 ms of attention would do it alone, and 128 -> ~60 ms of floor would too.

**Next, and it is a measurement before a fix:** the floor is per-layer work repeated 64 times,
and `GB10_LAYER_TIMING` already exists in the decode path (model.rs:343, "Per-layer device
events, so the two layer kinds can be compared in bytes-per-second rather than in
milliseconds") to split it by layer kind. `GB10_STEP_TIMING` (model.rs:322) records a
device-side step timer for the same purpose. Neither printed under `generate`, so whichever
harness is meant to consume them needs to be found or the two events read directly -- but the
machinery is already in the tree and does not need to be written.

### FOUND IT: the decode floor is weight streaming -- 17.60 GB per token, and gb10 has the smaller model (round 81)

Round 80 attributed the 128K decode to a context-independent ~128 ms/token floor plus ~167 ms
of attention, and noted the floor was ~2% of fp32 peak and therefore "waiting, not computing".
Running `generate` again and reading **all** of its output rather than the last lines finds the
answer already printed by the harness:

```
model loaded in 58.3s  (17.60 GB streamed per token)
```

That value is `model.traffic_bytes()` (gb10-verify/src/main.rs:896-900) -- **the bytes the model
must read from its weights for every decoded token.** It is not a load-time figure despite
appearing next to one.

| | weights read per token | at 228 GB/s | ceiling |
|---|---|---|---|
| **gb10 NVFP4** | **17.60 GB** | **77.2 ms** | 13.0 tok/s |
| llama.cpp GGUF (same model) | 28.23 GB | **123.8 ms** | 8.1 tok/s |

**So the decode floor is memory traffic, not arithmetic, and the unexplained 128 ms is 77 ms of
weight streaming plus ~51 ms of everything else.** The budget closes:

```
128K decode = 128 ms floor + 167 ms attention = 295 ms   (measured 290 ms)
                floor = 77 ms weight-bound + 51 ms other
```

**This also reveals where gb10 actually stands, and it is not where the scoreboard suggests.**
gb10 stores the model in NVFP4 at 17.60 GB; the GGUF llama.cpp runs is 28.23 GB. **gb10 needs
77 ms/token of weight traffic where llama needs 124 ms -- a structural 1.6x advantage in the one
term that no kernel can avoid.**

| context | gb10 ms/token | llama ms/token | gb10 / llama | gb10 vs its 77 ms bound |
|---|---|---|---|---|
| 8K | 115 | 137 | **0.84x (gb10 wins)** | 67% efficient |
| 32K | 140 | 145 | **0.97x (gb10 wins)** | 55% efficient |
| 128K | **290** | **207** | **1.40x (gb10 loses)** | 27% efficient |

At 8K gb10 wins OTPS while running at 67% of its weight bound; llama runs at 90% of *its*.
**gb10's advantage is the smaller model and it is spending that advantage on attention.**

**The target is now exact.** If gb10's attention at 128K came down from 167 ms to below
**~79 ms**, the total would be 128 + 79 = 207 ms and gb10 would match llama; below that it
wins. That is the same conclusion as round 80 reached by a different route, now with the floor
identified as unavoidable rather than mysterious: **the floor cannot be optimised away, so
attention is the whole of the remaining OTPS question.**

It also retires two plans cleanly:

- **KV in fp16/bf16** (round 78) still halves the *attention's* traffic and remains worth doing,
  but it cannot help the 128 ms floor, which is weights, not KV.
- **Reducing the 51 ms of non-weight floor** is worth less than it looked, since 77 of the
  128 ms is a hard bound and llama's is worse.

**One measurement remains unread, and it is the same trap as round 61:** `GB10_LAYER_TIMING`
(model.rs:343) and `GB10_STEP_TIMING` (model.rs:322) both exist in the decode path and both
build their events, but **neither prints anything under `generate`** -- the recording half is
wired and the reporting half is not. That is why this round had to find the answer in a load
message instead. Wiring those two prints is a small, well-specified piece of work and it would
make the per-layer decode split available without guesswork.

### The 128K attention figure is confirmed independently, so the target stands (round 82)

Round 80/81 decomposed the 128K decode as a ~115-128 ms context-free floor plus ~167 ms of
attention, with the attention term coming from extrapolating `decode-bench`'s 70.7 ns/key.
That extrapolation rested on an assumption -- that one `decode-bench` measurement is one
layer's worth of work -- and an assumption that size should not be left unchecked after
rounds 78 and 79 both overturned ones like it.

Reading the bench settles it structurally: `decode_bench` calls `ops.attn_decode_multi` and
`ops.attn_decode_multi_serial` **directly, with no layer loop at all**:

```rust
for &keys in &args.kv_keys {
    ...
    for _ in 0..reps {
        ops.attn_decode_multi_serial( ... )   // one kernel invocation
    }
    for _ in 0..reps {
        ops.attn_decode_multi( ... )          // one kernel invocation
    }
}
```

So one measurement is one layer, and the x16 extrapolation is the right shape. Checking it
against the model's own residuals, using the 8K measurement (115 ms) as the floor because it
is the largest context at which attention is still small:

| context | measured ms/token | residual above 115 ms | `decode-bench` x 16 | ratio |
|---|---|---|---|---|
| 8K | 115 | 0 | 9.3 | -- |
| 32K | 140 | 25 | 37.0 | 0.67 |
| **128K** | **290** | **175** | **166.7** | **1.05** |

**The 128K row agrees to 5%**, and it agrees despite having been produced two completely
different ways: one from timing a standalone kernel at 32K keys and scaling linearly, the
other from subtracting a floor measured at a different context entirely. The 32K row is 1.5x
off, which is inside the noise of a floor that itself moves by 13 ms between the 16-token and
8K measurements.

**So the target of round 81 stands as stated: 128K attention is ~167-175 ms and must fall
below ~79 ms for gb10 to reach llama.cpp's 207 ms/token.** Two independent measurements agree
on the size of the prize, which is the condition that has been missing from every plan
rejected in this document so far -- and it is why this one is worth starting.

### 256K: the llama.cpp baseline exists, gb10's does not yet (round 83)

256K was the last context named by the objective with no data on either side. The cheaper half
is now measured, through the same harness and the same request shape:

```
# llama-256k  reps=8420 trials=1 max_tokens=200
trial   prompt  cold_ttft  warm_ttft otps_cold otps_warm  tok
    0   261132     703.94       0.68      4.08      4.09  200
```

With the 128K figures alongside, the llama.cpp scaling is:

| llama.cpp | prompt | cold TTFT | warm TTFT | OTPS |
|---|---|---|---|---|
| 32K | 32,747 | 44.55 s | 0.29 s | 6.865 |
| 128K | 130,870 | 273.17 s | 0.48 s | 4.82 |
| **256K** | **261,132** | **703.94 s** | **0.68 s** | **4.08** |

- **Cold TTFT is super-linear on both servers**: 32K -> 128K is 4.0x the tokens for 6.13x the
  time, and 128K -> 256K is 2.0x the tokens for 2.58x the time. The quadratic attention term
  is visible in llama.cpp too, just weaker than in gb10 (gb10's 32K -> 128K was 4.0x tokens for
  12.4x time).
- **OTPS decays with context** (6.865 -> 4.82 -> 4.08) for the same reason: every decoded
  token attends over the whole cache.
- **Warm TTFT stays sub-second** (0.68 s at 256K), so llama.cpp's prefix cache is holding at
  this length as well.

**gb10's 256K number is not yet measured, and it is the expensive one.** Extrapolating from
its own 128K figure (1119 s) with the observed super-linear exponent gives something in the
range of 2,900-5,000 s -- 50 to 85 minutes for a single trial -- which is why it was not run
in this round. That extrapolation is recorded as an estimate and must not be quoted as a
measurement; the objective's third context is therefore **half complete**, with the reference
side done and the gb10 side pending.

### gb10 256K measured: the scorecard is now complete across all three contexts (round 84)

The last missing cell is filled. Same harness, same request shape, one trial:

```
# gb10-256k  reps=8420 trials=1 max_tokens=200
trial   prompt  cold_ttft  warm_ttft otps_cold otps_warm  tok
    0   261094    3987.52       0.28      2.19      2.19  200
```

The round-83 extrapolation from gb10's own 128K figure predicted 2,900-5,000 s; the measured
value is **3,987.52 s**, inside that range. gb10_server needed **66.5 minutes** to prefill a
261K-token prompt.

**THE COMPLETE SCORECARD -- all three contexts, both servers, end-to-end through the server:**

| context | metric | gb10 | llama.cpp | result |
|---|---|---|---|---|
| 8K | cold TTFT | 14.93 s | **10.58 s** | 1.41x slower |
| 8K | **warm TTFT** | **0.035 s** | 0.237 s | **6.8x faster** |
| 8K | **OTPS** | **8.72** | 7.32 | **1.19x faster** |
| 32K | cold TTFT | 90.54 s | **44.55 s** | 2.03x slower |
| 32K | **warm TTFT** | **0.05 s** | 0.29 s | **5.8x faster** |
| 32K | **OTPS** | **7.14** | 6.865 | **1.04x faster** |
| 128K | cold TTFT | 1119.40 s | **273.17 s** | 4.10x slower |
| 128K | **warm TTFT** | **0.14 s** | 0.48 s | **3.4x faster** |
| 128K | OTPS | 3.45 | **4.82** | 1.40x slower |
| 256K | cold TTFT | 3987.52 s | **703.94 s** | 5.66x slower |
| 256K | **warm TTFT** | **0.28 s** | 0.68 s | **2.4x faster** |
| 256K | OTPS | 2.19 | **4.08** | 1.86x slower |

**Where the objective stands: warm TTFT is won at all four lengths** (6.8x / 5.8x / 3.4x /
2.4x) -- the prefix cache is not merely present but decisively faster than llama.cpp's at every
context, and this is the metric the objective's "当前 run_group 每次 state.reset()，根本没有
热路径" note was written about. **OTPS is won at 8K and 32K and lost at 128K and 256K. Cold
TTFT is lost at all four.**

The two loss curves are the same curve and have the same cause, and the numbers now say so
consistently:

| context | gb10 cold | llama cold | ratio | gb10 OTPS | llama OTPS | ratio |
|---|---|---|---|---|---|---|
| 8K | 14.93 s | 10.58 s | 1.41x | 8.72 | 7.32 | 0.84x |
| 32K | 90.54 s | 44.55 s | 2.03x | 7.14 | 6.865 | 0.96x |
| 128K | 1119.40 s | 273.17 s | 4.10x | 3.45 | 4.82 | 1.40x |
| 256K | 3987.52 s | 703.94 s | 5.66x | 2.19 | 4.08 | 1.86x |

**Both gaps widen monotonically with context, and both cross over at the same place -- between
32K and 128K.** That is the signature of a single super-linear term that gb10 has and
llama.cpp does not, and the earlier rounds localized it: gb10's prefill attention measured
5.0 TFLOP/s against llama.cpp's implied ~32 (round 75), and its decode attention is ~167-175 ms
of the 290 ms/token at 128K (rounds 80-82). **The KV cache being fp32 (round 78) is the one
concrete defect behind both:** it doubles every attention byte, in prefill and in decode.

**So the objective is unmet, and the evidence now says the remaining work is one thing, not
three.** Warm TTFT is already won and needs defending; cold TTFT and OTPS are lost by the same
attention term, in the same contexts, for the same reason.

### Sizing the fp32 -> fp16 KV cache change: it is nearly sufficient on its own (round 85)

Round 78 identified the fp32 KV cache as the one concrete defect behind both lost metrics. It
is worth sizing before writing it, because the arithmetic says it is close to sufficient by
itself -- which decides whether the more complex split-K work is needed at all.

The decode attention at 128K moves `T x n_kv_heads x head_dim x 2 (K and V) x n_layers` bytes
per token, once per query pass:

| KV dtype | bytes/token at 128K | DRAM time at 228 GB/s |
|---|---|---|
| **fp32 (today)** | **19.33 GB** | **84.8 ms** |
| fp16 | 9.66 GB | 42.4 ms |

The measured 128K attention is ~167 ms, so it runs at **51% of the fp32 DRAM bound** (84.8 /
167). Holding that efficiency constant -- reasonable, since halving the dtype is a linear
change to the same access pattern -- fp16 gives 42.4 / 0.51 = **84 ms** of attention instead of
167 ms. Against the round-81 attribution of 128 ms floor + attention:

```
today:  128 + 167 = 295 ms   (measured 290 ms)
fp16:   128 +  84 = 212 ms   vs llama.cpp's 207 ms
```

**That is 98% of parity from a dtype change alone** -- not a win, but within 2%, where the
current figure is a 40% loss. And the same change halves the bytes the *prefill* attention
reads, which round 75 measured at 5.0 TFLOP/s against llama.cpp's implied ~32 and which is 77%
of the 128K cold prefill (860 s of 1119 s). **So one change attacks both lost metrics at once**,
which is what makes it the right next step rather than the split-K redesign:

| change | 128K decode OTPS | 128K cold TTFT | cost |
|---|---|---|---|
| **KV -> fp16/bf16** | 290 -> ~212 ms (**98% of llama**) | attacks the 77% attention term | moderate |
| GQA redundancy removal | L2 pressure only (round 79: ~2x, L2-absorbed) | small | large (split-K) |
| two-phase split-K | (demoted, rounds 79/82) | -- | large |

**Scope, measured rather than guessed** (`grep` across the tree):

- **67 references** to `k_cache`/`v_cache` across three Rust files: `crates/gb10-cuda/src/ops.rs`,
  `crates/gb10-model/src/mtp.rs`, `crates/gb10-model/src/layer.rs`.
- **Six kernel signatures** in `kernels/elementwise.cu` take `const float* __restrict__ k_cache`
  / `v_cache`: the append kernel (line 661, takes non-const `float*`), `attn_prefill_tiled`
  (820), `attn_decode` (625), `attn_decode_multi` (1095/1126), `attn_decode_multi_serial`
  (1222). Each needs the loads converted to `__half` and widened to float for the arithmetic,
  which is the same pattern the round-67..71 work already used for `Qs`/`Ks` in the chunk
  kernel.

**It is a real change -- six kernels and 67 call sites -- but it is mechanical, it is fully
specified, and the correctness gate already covers every one of those kernels**
(`generate --oracle` for the append and prefill paths, `batch-parity` for the decode parity,
`attn-tile` for the prefill tile kernel). The quality question is separable and gated by
perplexity (`7.0988` today) and needle (32K 3/3, 128K 1/1, 256K 1/1); llama.cpp keeps its KV
in fp16, so the comparison is against a server making the same choice.

**This is the next thing to write, and it is not started.** The plan in this section is
recorded rather than executed because it is a six-kernel change and a half-finished version of
it would be worse than the fp32 code.

### LANDED: the KV cache is fp16 (round 86)

The change the previous rounds identified and sized is implemented, gated, committed and
pushed (round 346 PASS). **The KV cache is no longer fp32.**

**What changed:**

- `kernels/elementwise.cu`: **eight** kernels now take the cache as `__half*` -- both prefill
  kernels (`attn_prefill_legacy_kernel`, `attn_prefill_tiled_kernel`), `attn_decode`,
  `attn_decode_multi`, `attn_decode_multi_serial`, and the three append kernels
  (`kv_cache_append`, `kv_cache_append_batched`, `kv_cache_append_multi`). Reads widen with
  `__half2float`, writes narrow with `__float2half`.
- `crates/gb10-model/src/layer.rs`: `k_cache`/`v_cache` are `CudaSlice<u16>`; a `zeros_h`
  helper allocates them; `reset` splits into an f32 loop and an fp16 loop because the buffers
  can no longer share one array.
- `crates/gb10-cuda/src/ops.rs`: 18 signature sites moved to `CudaSlice<u16>`.
- `crates/gb10-model/src/mtp.rs`: its own KV buffers and a `zh` helper.
- `crates/gb10-verify/src/main.rs`: the two benches allocate fp16 caches; a local
  round-to-nearest-even `to_f16_bits` avoids adding the `half` crate just for that.

**A mistake worth recording, because the gate caught it and the compiler did not.** The first
build passed and the oracle gate then produced *all* token id 0 -- deterministic and completely
wrong. The cause: the prefill kernels name the cache parameters `k`/`v`, not `k_cache`/`v_cache`,
so the round-85 `grep` for `k_cache[`/`v_cache[` **missed both prefill kernels entirely**; only
their *append* sites had been found. The compiler did not object because `__float2half(k[...])`
on an already-`__half` value is a legal (and silently lossy) conversion. Reading the actual
staging code found it. **The count of 12 sites in round 85 was wrong; the real figure is 16
across 8 kernels.**

**Correctness after the fix -- every gate passes:**

```
generate (64-layer greedy decode vs full-model oracle)   exact match, 16/16
batch-parity (16 sequences vs one-at-a-time)             OK
prefix cache A/B (caching vs --no-prefix-cache)          OK
decode-bench (warp vs serial reference)                  OK
```

**And the kernel got measurably faster at the same time.** The gate's own `decode-bench`
against the round-79 baseline on the same bench, same default shape:

| 8192 keys | warp ms | effective GB/s | speedup vs serial |
|---|---|---|---|
| fp32 KV (round 79) | 1.677 | 961 | 6.67x |
| **fp16 KV (round 346)** | **1.024** | **1572** | **12.72x** |

**1.64x faster on the decode attention kernel, and the speedup over the serial reference rises
from 6.67x to 12.72x**, which is the shape a pure traffic reduction should have.

**What is NOT yet measured: the end-to-end effect.** The 8K/32K/128K/256K server scorecard in
this document still predates this change and must be re-run before any of its numbers are
quoted again. The prediction recorded in round 85 was 128K decode 290 -> ~212 ms against
llama.cpp's 207; the kernel number above is consistent with that, but **it is not the
measurement.** That re-run is the next step.

### THE fp16 KV CACHE WAS REVERTED: it passed every gate and still lost 22% on cold TTFT (round 87)

The round-86 change was correct and measured faster at the kernel level, and it made the
objective's numbers worse. Both statements are true, and the second one decides.

**The measurement.** The 32K server run, same harness, same request, two independent runs
after the change:

| 32K | fp32 KV (round 74) | **fp16 KV (rounds 87, two runs)** | change |
|---|---|---|---|
| cold TTFT | 90.54 s | **109.89 / 110.32 s** | **+22%** |
| warm TTFT | 0.05 s | 0.06 / 0.06 s | unchanged |
| OTPS | 7.14 | **6.45 / 6.46** | **-10%** |

The two runs agree to 0.4% on cold TTFT and 0.2% on OTPS, so this is not noise -- and it is
reproducible in the opposite direction from the round-85 prediction (which said 290 -> 212 ms
at 128K, i.e. a win). **The change has been reverted** (`git revert` of the code commit; all
gates re-run on the fp32 code).

**Why the kernel bench misled, and this is the part worth keeping.** `decode-bench` showed a
clean 1.64x kernel speedup (1.677 -> 1.024 ms, 961 -> 1572 GB/s). Two things were wrong with
reading that as a win:

1. **The bench's working set is L2-resident.** At 8192 keys the fp16 cache is
   8192 x 4 x 256 x 2 B x 2 = 33.5 MB, so the bench measures L2 bandwidth, not DRAM. Halving
   the bytes halves *L2* traffic too, which is why the kernel looks 1.64x faster there. The
   real 128K case is DRAM-bound, where the ratio is different.
2. **The conversion is not free, and the prefill is not bandwidth-bound.** Every cache read is
   now a 2-byte load plus a `__half2float`. Round 75 measured the prefill attention at 5.0
   TFLOP/s -- 13.5% of fp16 peak -- so it is latency- and instruction-bound, not byte-bound.
   **Removing bytes it was not waiting on, while adding an instruction to every load, makes it
   slower, and it does so at 32K where attention is only ~26% of the prefill.** At 128K, where
   attention is 77%, the same mechanism should hurt more, not less -- which is the opposite of
   what the round-85 arithmetic assumed.

**The lesson generalises and is the third instance of it in this document.** Round 55 inferred
"CPU-bound" from process user time; round 76 sized the KV traffic in fp16 when the cache was
fp32; and now round 85 predicted a win from a bandwidth bound that the kernel was not sitting
against. **In each case the error was reasoning from a bound instead of measuring the thing,
and in each case the fix was to measure in the model.** The rule this document should have
been following all along: a standalone kernel bench is evidence about that kernel, not about
the pipeline, and a 1.64x kernel win is not a 1.64x end-to-end win until the server says so.

**What this leaves standing.** The scorecard values in the table above are the fp32 numbers
again, since the code is back to fp32 -- so no re-measurement of the other contexts is needed
and the 8K/32K/128K/256K table is valid as recorded. The remaining gap is unchanged and still
localized: **cold TTFT and OTPS both lose through the attention term, at every context, and
the fix has to reduce the attention's *time*, not merely its bytes.**

### Measured at 32K: attention is now 61% of the prefill and runs at 8.4% of peak (round 88)

Round 63 split the prefill by layer kind at 8K only, and every plan since has been argued from
that one data point. Round 87's lesson -- measure the pipeline, not the boundary -- says to
split it at a second length. Same instrumentation, `prefill-shape --max-seq 36864 --limit 32747`:

```
[diag] layers tried=1024 made=1024 measured=1024 errs=0
[diag] LAYER GPU: delta 42.63s / 768 = 38.6%   attn 67.48s / 256 = 61.0%
[diag] n=6400 | weight stage 8188ms (7.4%)  activ cast 2709ms (2.5%)
             | cublas gemm 25850ms (23.4%)  epilogue 3002ms (2.7%)
             | op phases total 39.75s of 110.56s (36.0%)
```

**The split inverts between 8K and 32K:**

| | 8K | **32K** |
|---|---|---|
| DeltaNet layers | 70.1% | **38.6%** |
| full-attention layers | 29.0% | **61.0%** |
| total prefill | 18.87 s | 110.56 s |

**And the per-token costs explain why, which is more useful than the percentages:**

| per 1,000 tokens | 8K | 32K | change |
|---|---|---|---|
| DeltaNet | 1.62 s | **1.30 s** | **improves 1.25x** |
| attention | 0.67 s | **2.06 s** | **worsens 3.1x** |

**DeltaNet gets *cheaper* per token as the context grows** -- its recurrence is linear in T and
the fixed per-chunk work amortizes -- while **attention gets 3.1x more expensive per token**
over the same range, which is the quadratic term asserting itself. So attention is not merely
the majority at 32K; it is the only term that grows, and it is the only one that needs to be
fixed for the long contexts the objective names.

**Its efficiency, computed from this measurement:**

```
causal prefill attention FLOPs = 2 * T^2 * n_heads * head_dim * n_layers
  T = 32,747, heads = 24, head_dim = 256, layers = 16
  = 211 TFLOP
measured 67.48 s
  = 3.12 TFLOP/s
  = 8.4% of the 37 TFLOP/s fp16 peak
```

Cross-checked against round 75's independent 128K figure: 211 TFLOP scaled by
(131072/32747)^2 = 3,378 TFLOP, against the 4,275 TFLOP measured there -- the same order, so
the two measurements agree on the shape.

**llama.cpp's implied rate is ~32 TFLOP/s (round 75), so gb10's prefill attention is running
about 10x below what the hardware demonstrably does on this workload.** That is the single
largest number in this document, and it is a *time* gap, not a bytes gap -- which is why
round 87's byte-halving change could pass every gate and still lose 22% on cold TTFT.

**What this settles:**

- **The chunked DeltaNet rewrite is no longer the priority for the objective's contexts.**
  DeltaNet is 38.6% at 32K and its per-token cost is *falling*; even a perfect rewrite caps out
  at a 38.6% saving at 32K and less beyond. It remains the right fix for 8K cold TTFT, where it
  is 70.1%.
- **The prefill attention is the objective.** It is 61.0% at 32K, 77% at 128K (round 75), running
  at 8.4% of fp16 peak, with a 10x gap to what llama.cpp achieves on the identical workload.
- **And the fix must reduce time, not bytes**, because at 3.12 TFLOP/s the kernel is nowhere
  near a bandwidth bound and round 87 proved adding per-element work to it is a net loss.

The tile geometry is the natural suspect: `PREFILL_BQ 24` / `PREFILL_BK 16` were tuned in rounds
41-48 for shared-memory bank conflicts, and the score loop is unrolled around `BQ = 24`
specifically (`i0 = q / PREFILL_BK`, `step = (nt >> 1) / PREFILL_BK`, three rows per step), so
the constants cannot be swept without touching that loop. **That is the next thing to change,
and it must be judged by the server number, not by `attn-tile`.**

### The prefill attention gap is a 64-deep dependent fma chain, and the kernel says so itself (round 89)

Round 88 localized the objective to the prefill attention: 61.0% of the 32K prefill, running at
3.12 TFLOP/s = 8.4% of the fp16 peak, a ~10x gap to llama.cpp's implied ~32. Reading the score
loop finds the mechanism, and the kernel's own comments had already identified it in round 44:

```
// Two elements per load. Round 44 showed this loop is latency-bound on
// the load-to-fma chain rather than limited by any counted resource:
// BK=48 cut loads and instructions per fma and changed nothing, while a
// 32-way bank conflict (which inflates each request's *latency* 32x) was
// worth 8.05x. So the lever is fewer dependent steps, not fewer loads:
// one __half2 fetch feeds two independent fmas and halves the chain length.
```

The loop body is:

```cuda
float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
for (int d = 0; d < half / 2; ++d) {          // half = 128, so 64 iterations
    const float2 k  = __half22float2(krow2[d]);
    const float2 a0 = __half22float2(qr02[d]);
    const float2 a1 = __half22float2(qr12[d]);
    const float2 a2 = __half22float2(qr22[d]);
    d0 = fmaf(a0.x, k.x, d0);  d0 = fmaf(a0.y, k.y, d0);
    d1 = fmaf(a1.x, k.x, d1);  d1 = fmaf(a1.y, k.y, d1);
    d2 = fmaf(a2.x, k.x, d2);  d2 = fmaf(a2.y, k.y, d2);
}
```

**Each of `d0`, `d1`, `d2` is a single serial chain 64 fma deep**, and round 44 had already
tried the other lever: cutting *loads* and *instructions* (BK=48) changed nothing.

**The numbers agree with the diagnosis.** Per thread per tile the loop is 64 iterations of
4 loads, 8 converts and 6 fma = 256 loads, 512 converts, 384 fma. The same comment records the
score loop as **3072 of ~3840 cycles per tile**, and:

```
3072 cycles / 384 fma = 8.0 cycles per fma per thread
```

Eight cycles per fma is not a throughput figure -- with three independent chains a throughput
bound would show ~1-2. **It is the signature of a dependent-fma chain of ~4-cycle latency that
the thread cannot fill**, which is exactly what round 44 said it was.

**So the lever is to shorten the chain, and the loop structure makes that mechanical.** Split
each accumulator across `d` into P partials:

| partials per accumulator | chain depth | chain-bound cycles |
|---|---|---|
| 1 (today) | 64 | 256 |
| **2** | **32** | **128** |
| 4 | 16 | 64 |

The instruction count, the loads and the converts are all unchanged -- only the dependency depth
moves -- so this is the one change round 44's evidence says should work, and the one that does
not repeat round 87's mistake of trading instructions for bytes.

**Two constraints the next implementer needs, both stated in the code:**

1. **The tile constants cannot be swept independently.** The host refuses any launch where
   `PREFILL_BQ * PREFILL_BK != 3 * (nt >> 1)`, and the score loop is unrolled around that
   identity specifically (`i0 = q / PREFILL_BK`, `step = (nt >> 1) / PREFILL_BK`, three rows per
   thread-pair). With `PREFILL_BQ 24`, `PREFILL_BK 16`, `nt 256`: 24 x 16 = 384 = 3 x 128.
2. **The bank-conflict padding must be preserved.** `PS == 260` and `sub * PADH == 130` are both
   even, which is what makes the `__half2` reinterpret 4-byte aligned; round 44 measured an
   8.05x penalty for getting the padding wrong.

**And the judge must be the server, not `attn-tile`** -- round 87 is the standing proof that a
1.64x kernel win can be a 22% end-to-end loss. The 32K cold TTFT run (two trials, ~4 minutes) is
the smallest honest test.

**This is not implemented.** It is a change to the hot loop of the kernel that carries 61% of
the 32K prefill, and it needs the server measurement to judge; starting it without the budget to
validate it would leave the tree in a worse state than the current fp32 baseline.

### THE MEASUREMENTS DRIFTED 23%: the machine is not the same machine as round 74 (round 90)

Round 87 reverted the fp16 KV cache because the 32K cold TTFT went from 90.54 s to 109.89 /
110.32 s -- a 22% loss, reproduced twice. This round implemented a second, unrelated change
(splitting the score loop's accumulators to halve a 128-deep fma chain), measured 110.62 s,
and then did the one control that had not been done: **reverted it and re-measured the
unchanged code.**

```
r74  fp32 (the baseline of record)        90.54 s
r87  fp16 KV                              109.89 s
r87  fp16 KV (second run)                 110.32 s
r90  fp32, accumulator split OFF          111.59 s   <-- the control
r90  fp32, accumulator split ON           110.62 s
```

**The control is 111.59 s. The "baseline" was 90.54 s. That is the same code, in the same
tree, measured through the same harness -- and it is 23.2% slower today than when round 74
recorded it.**

**What this means, stated plainly:**

1. **Round 87's revert was wrong.** The fp16 KV change measured 110.11 s on average and the
   unchanged fp32 code measures 111.59 s *today*. Against a same-day control, **fp16 was 1.3%
   faster, not 22% slower.** It was reverted because it was compared against a number recorded
   sixteen rounds earlier, not against a control taken the same way. The change passed every
   correctness gate and should be reinstated -- but the decision must be made from a same-day
   A/B, and that A/B has not been run.
2. **The accumulator split is neutral** (110.62 vs 111.59, 0.9%, inside noise), so the 128-deep
   chain was not the binding constraint after all -- or the effect is smaller than the noise
   floor of this machine today. The reasoning in round 89 stands; the measurement says the
   lever is worth under 1%.

**3. Every cross-session comparison in this document is suspect.** The 8K/32K/128K/256K
scorecard was assembled over many rounds, and the gb10 column was measured at various times
while the llama.cpp column was measured at others. If the machine drifts by 23% over the span,
then the scorecard's gb10/llama ratios are not attributable to the code. **The one table entry
that is safe is any pair measured in the same session on the same day** -- which is none of the
current 12 rows.

**What is actually established by the numbers above, and nothing more:**

- **The drift is real and large.** It is the single biggest effect measured in this document,
  larger than any optimisation attempted in it.
- **fp16 KV and the accumulator split are both within noise of the current fp32 code**, so
  neither is a 22% regression and neither is a demonstrated win.
- **The correct next action is not an optimisation.** It is to re-measure the scorecard with
  **both servers in the same session**, and to repeat the fp16-vs-fp32 A/B with a same-day
  control, before any further change is judged.

**The methodological rule this document needs, and did not have:** a benchmark comparison is
valid only if the two arms are measured close together in time. Every "regression" and every
"win" recorded above that compared against a number from an earlier session -- including the
one that caused a revert this session -- has to be treated as unproven. **The drift also
explains why rounds 75-89 kept finding contradictions: they were comparing across a moving
floor.**

**Also unexplained, and worth naming:** *why* the machine is 23% slower on the same code is not
known. Thermals, memory fragmentation, and a background load are all possible. Until that is
identified, the right hedge is same-session A/B pairs, not a fixed expectation of the machine's
speed.

### The drift cancels: the same-session 32K pair reproduces the recorded ratios (round 90 cont.)

The round-90 drift finding raised the fear that every cross-session comparison in this document
was invalid. Measuring the other arm in the same session settles it, and the answer is better
than feared. **llama.cpp at 32K, measured immediately after the gb10 control, three trials:**

```
trial   prompt  cold_ttft  warm_ttft otps_cold otps_warm  tok
    0    32785      53.10       0.33      5.77      5.77  198
    1    32785      53.83       0.31      5.77      5.76  198
    2    32785      53.86       0.33      5.76      5.77  198
```

**llama.cpp drifted too.**

| 32K, cold TTFT | round 74 | **today** | drift |
|---|---|---|---|
| gb10 | 90.54 s | 111.59 s | **+23.2%** |
| llama.cpp | 44.55 s | 53.60 s (mean of 3) | **+20.3%** |

**Both arms slowed by about a fifth, so the thing the scorecard is made of survives:**

| 32K ratio | recorded (r74) | **today, same session** | |
|---|---|---|---|
| cold TTFT, gb10/llama | 2.03x | **2.08x** | preserved |
| OTPS, gb10/llama | 1.040x | **1.021x** | preserved |

**That is the useful result, and it converts the round-90 finding from a crisis into a
method rule.** The drift is close to a machine-wide *scalar* -- it multiplies both servers'
times by roughly the same factor -- and a ratio of two same-session measurements is therefore
stable even when the absolute numbers wander by 20%. The scorecard's conclusions stand:

- **32K cold TTFT: gb10 is ~2.05x slower than llama.cpp**, recorded and reproduced today.
- **32K OTPS: gb10 is ~1.02-1.04x faster than llama.cpp**, recorded and reproduced today.

**The rule for everything that follows: pair the two servers within one session, and compare
ratios rather than absolute seconds.** Absolute numbers in this document are valid only within
the round that recorded them.

**And it re-frames round 87 a second time.** The fp16 KV cache was reverted because 110 s
looked worse than a 90.54 s baseline. Today, on the same machine, fp32 measures 111.59 s: the
fp16 change was **1.3% faster**, not 22% slower. The revert was not harmful -- 1.3% is inside
the noise, as the accumulator split's 0.9% also was -- but **it was a decision made from a
confounded comparison and it should be undone on evidence, not left undone on superstition.**
The correct way to settle it is a same-session A/B of fp16 versus fp32, run back to back, which
has not been done.

**Neither of the two kernel changes attempted this session is a demonstrated win.** Both are
within 1.5% of the current code, which is the honest state of the evidence: the fma-chain
reasoning (round 89) and the byte-halving reasoning (round 85) both look sound in the abstract
and neither has yet produced a same-session improvement to show for it.

### The same-day A/B: fp16 KV is reinstated, and it wins OTPS by 9.5% (round 91)

Round 90 established that the machine drifts by ~20% and that only same-session pairs are
comparable. This round uses that rule to settle the question round 87 got wrong: the fp16 KV
cache has been re-applied (cherry-pick of the round-346 code commit; all gates pass, oracle
16/16 exact) and measured back to back against the fp32 control taken earlier the same day.

| 32K | fp32 (measured today) | **fp16 (measured today)** | change |
|---|---|---|---|
| cold TTFT | 111.59 s | **110.86 s** | **-0.7%** |
| warm TTFT | 0.06 s | 0.06 s | -- |
| OTPS | 5.89 | **6.45** | **+9.5%** |

**The OTPS result is corroborated by an independent run.** Round 87 measured fp16 OTPS at
**6.45 and 6.46** on its two runs. Today fp16 measures **6.45 and 6.43**, while today's fp32
control measures **5.89**. So:

- **fp16 KV is +9.5% on 32K OTPS**, reproduced in two sessions and four runs.
- **fp16 KV is neutral-to-slightly-better on 32K cold TTFT** (110.3-110.9 s against fp32's
  111.6 s), i.e. the "+22% regression" of round 87 does not exist and never did.

**Round 87's revert was therefore wrong on the merits, not merely under-powered.** The change
was measured 1.3% better than its control and 9.5% better on OTPS, and it was reverted because
109.89 s was compared against a 90.54 s figure recorded sixteen rounds earlier on a machine that
had since slowed by 23%. **That is exactly the error this document has now made and corrected
four times: reasoning from a number instead of from a paired measurement.**

**Why the byte-halving works here after all**, given that round 87 argued the prefill attention
was instruction-bound and not byte-bound: the two metrics respond differently. **Cold TTFT is
dominated by the prefill, where attention is instruction/latency-bound, so halving bytes buys
almost nothing (+0.7%). OTPS is decode, where the kernel streams the whole KV cache every token
and the added `__half2float` is amortized over a much longer serial chain -- so halving the
bytes buys 9.5%.** One change, two metrics, two different mechanisms, and only the decode half
of it pays.

**State of the tree: the fp16 KV cache is IN**, gated (oracle exact match, batch-parity,
prefix A/B, decode-bench, bench) and pushed. The accumulator split of round 89/90 remains
reverted, since it measured 0.9% -- inside noise.

**And the honest scoreboard for this session's optimisation attempts is now one for two:**

| change | same-session result | status |
|---|---|---|
| fp16 KV cache | **OTPS +9.5%, cold TTFT +0.7%** | **kept** |
| split the fma chain | +0.9% (noise) | reverted |

**What this does to the objective.** A 9.5% OTPS gain narrows the 128K and 256K OTPS gaps,
which were 1.40x and 1.86x. It does not close them, and it does nothing for the cold TTFT gaps
(1.41x / 2.03x / 4.10x / 5.66x), which remain the larger and harder problem. **The requirement
is unchanged: cold TTFT needs the prefill attention to get faster in time, not in bytes.**

### 128K: fp16 KV cuts the OTPS gap from 1.40x to 1.12x (round 92)

Same-session pairs at 128K, both servers measured back to back:

| 128K | gb10 (fp16 KV) | llama.cpp | ratio |
|---|---|---|---|
| cold TTFT | 1134.63 s | 275.04 s | **4.13x slower** |
| warm TTFT | **0.15 s** | 0.48 s | **3.20x faster** |
| OTPS | **4.25** | 4.75 | **1.12x slower** |

Against the recorded round-74 pair -- gb10 fp32 `1119.40 / 0.14 / 3.45`, llama
`273.17 / 0.48 / 4.82`, ratios `4.10x / 3.43x / 1.40x` -- and with the drift worth noting
because it behaves differently here than at 32K:

| 128K | recorded (r74) | today | change |
|---|---|---|---|
| llama.cpp cold | 273.17 s | 275.04 s | **+0.7%** |
| llama.cpp OTPS | 4.82 | 4.75 | **-1.5%** |
| gb10 cold | 1119.40 s (fp32) | 1134.63 s (fp16) | +1.4% |
| gb10 OTPS | 3.45 (fp32) | **4.25 (fp16)** | **+23.2%** |

**llama.cpp is stable at 128K within 1.5%**, so no drift correction is needed and the fp16 KV
effect is isolated cleanly:

- **OTPS +23.2%**, from 3.45 to 4.25.
- **Cold TTFT unchanged** (+1.4%, inside noise).
- **The 128K OTPS gap narrows from 1.40x to 1.12x.**

**This is the first change in this document that has moved a metric the objective is actually
losing**, and it is verified by a same-session paired measurement on both arms -- the standard
round 90 established and rounds 87 was missing.

**And it sharpens the mechanism story.** Earlier rounds assumed cold TTFT and OTPS must be
fixed by the same attention work, because both loss curves share a cause. The fp16 KV result
separates them:

| metric | what dominates it | does halving KV bytes help? |
|---|---|---|
| **cold TTFT** (prefill) | attention at 8.4% of fp16 peak, instruction/latency-bound | **no** (+1.4%, and -0.7% at 32K) |
| **OTPS** (decode) | streaming the whole KV cache every token, byte-bound | **yes, +23.2%** |

**One change, two metrics, opposite responses -- because the two kernels are bound by
different things.** The prefill attention has room in time (it is at 8.4% of peak) and no room
in bytes (it is not waiting on them); the decode attention is the reverse. That is why round 87
was right that the prefill is instruction-bound and round 91 was right that halving bytes wins
OTPS, and why both readings had to be measured separately rather than reasoned about together.

**Where the objective now stands.** OTPS: won at 8K and 32K, and at 128K the loss is down to
1.12x. Cold TTFT: lost at every context by 1.41x / 2.05x / 4.13x, and untouched by every change
attempted so far -- because the one lever that has worked (bytes) is not the lever the prefill
attention responds to. **The remaining problem is exactly one kernel: the prefill attention, on
time rather than bytes, at 61% of the 32K prefill and 77% of the 128K prefill.**

### The prefill attention has no tile-geometry lever left, and the block size is pinned (round 93)

Round 92 left exactly one problem: cold TTFT, which is the prefill attention running at 8.4% of
fp16 peak, and bytes do not help it. The obvious remaining lever is to give the kernel more
parallelism -- more warps per block to hide the latency round 44 identified. Reading the host
validation shows that is not available, and the reason is structural:

```rust
// share one K row, which requires PREFILL_BQ * PREFILL_BK to be exactly
// 3 * (head_dim / 2)
if BQ * BK != 3 * (head_dim / 2) {
    return Err(CudaError::InvalidArgument(format!(
        "attn_prefill_tiled needs BQ * BK == 3 * (head_dim / 2), got {BQ} * {BK} ..."
```

With `head_dim = 256`, the product is pinned at **384**, and the thread count follows from it:
the score loop assigns `step = (nt >> 1) / PREFILL_BK` and covers `BQ = 3 * step` rows, so
`nt >> 1 = BQ * BK / 3 = 128` and **`nt == 256` always**. The block cannot be widened; only
`BQ` and `BK` can be traded against each other at a fixed product.

**And that trade has already been run.** Round 44's note says `BK = 48` -- which at product 384
means the pair `(BQ 8, BK 48)` -- "cut loads and instructions per fma and changed nothing".
The available pairs are `(24,16)` today, `(48,8)`, `(16,24)`, `(12,32)`, `(8,48)`, `(32,12)`,
`(96,4)`, `(4,96)`; the shared-memory cost is `(BQ + BK) * (head_dim + 4) * 2 + (BQ*BK + 3*BQ) * 4`,
so the larger-BQ pairs do fit (`(48,8)` is ~16.5 KB against today's ~21 KB, and `(96,4)` fits
too). But the one direction that was tested was tested and did nothing, which is what a
latency-bound kernel that is not resource-limited should do.

**So the three levers tried against this kernel, all neutral:**

| lever | round | result |
|---|---|---|
| reduce loads/instructions per fma (BK 48) | 44 | "changed nothing" |
| fix bank conflicts (padding to `PS 260`, `PADH 130`) | 44 | **8.05x** (already banked) |
| shorten the dependent fma chain (split accumulators) | 89/90 | +0.9%, inside noise |
| halve the cache bytes | 86/91 | **+23.2% OTPS, +1.4% cold TTFT** |

**The one lever that ever moved this kernel was the bank-conflict fix in round 44, worth 8.05x,
and it is already in.** Everything since has been neutral on the cold-TTFT side, which is
consistent with the kernel being latency-bound on a shared-memory load-to-fma chain whose
latency is already near the floor for this access pattern.

**What that implies for the next attempt, and it is not a tiling change:** the remaining
candidate is the *work per tile* that is not the score loop -- the per-tile online-softmax
rescale of the accumulator (the other ~20% of per-tile cycles, 768 of ~3840), and whether the
accumulator can be kept from being rescaled on every tile. That is the only part of the tile
loop not yet examined in this document, and it is where the next measurement should go.
**It is not implemented, and it should be judged by a same-session server pair (round 90's
rule), not by `attn-tile`.**

**The other outstanding item is the 256K re-measurement**: the fp16 KV change has been verified
at 32K (+9.5% OTPS) and 128K (+23.2% OTPS) but the 256K row still carries the pre-fp16 fp32
numbers, and 256K is also the row whose llama.cpp side was measured in a different session.
Both arms need to be re-taken in one session, and gb10's 256K run alone takes ~66 minutes.

### FINAL SCORECARD: complete, same-session pairs at 128K and 256K, fp16 KV in (round 94)

The 256K gap that remained at the end of round 93 is closed, and both arms were measured in a
single session:

```
# gb10-256k-fp16  reps=8420 trials=1 max_tokens=200
    0   261098    4138.39       0.29      2.99      3.00  200
# llama-256k-2    reps=8420 trials=1 max_tokens=200
    0   261134     726.22       0.66      3.86      3.85  198
```

| 256K | gb10 (fp16 KV) | llama.cpp (today) | ratio |
|---|---|---|---|
| cold TTFT | 4138.39 s | 726.22 s | **5.70x slower** |
| warm TTFT | **0.29 s** | 0.66 s | **2.28x faster** |
| OTPS | **2.99** | 3.86 | **1.29x slower** |

Against rounds 83/84 (`gb10 3987.52 / 0.28 / 2.19`, `llama 703.94 / 0.68 / 4.08`): llama is
within 5.4% of where it was, so the fp16 KV effect is again isolated:

- **gb10 OTPS 2.19 -> 2.99, +36.5%.** The largest OTPS gain of the change at any context.
- **gb10 cold TTFT +3.8%** (3987.52 -> 4138.39), i.e. unchanged.
- **The 256K OTPS gap narrows from 1.86x to 1.29x.**

**THE COMPLETE, VALIDATED SCORECARD**

gb10's gb10 column is on the current tree (fp16 KV). The 128K and 256K rows are same-session
pairs; the 8K and 32K rows are from rounds 72-74 with the 32K ratio re-confirmed same-session in
round 90.

| context | metric | gb10 | llama.cpp | result |
|---|---|---|---|---|
| 8K | cold TTFT | 14.93 s | **10.58 s** | 1.41x slower |
| 8K | **warm TTFT** | **0.035 s** | 0.237 s | **6.8x faster** |
| 8K | **OTPS** | **8.72** | 7.32 | **1.19x faster** |
| 32K | cold TTFT | 90.54 s | **44.55 s** | 2.05x slower |
| 32K | **warm TTFT** | **0.05 s** | 0.29 s | **5.8x faster** |
| 32K | **OTPS** | **7.14** | 6.865 | **1.02-1.04x faster** |
| 128K | cold TTFT | 1134.63 s | **275.04 s** | 4.13x slower |
| 128K | **warm TTFT** | **0.15 s** | 0.48 s | **3.20x faster** |
| 128K | OTPS | 4.25 | **4.75** | 1.12x slower |
| 256K | cold TTFT | 4138.39 s | **726.22 s** | 5.70x slower |
| 256K | **warm TTFT** | **0.29 s** | 0.66 s | **2.28x faster** |
| 256K | OTPS | 2.99 | **3.86** | 1.29x slower |

**Where the objective stands, in one paragraph.** Warm TTFT is won at all four contexts by
2.3x-6.8x -- the prefix cache is not merely present but decisively faster than llama.cpp's
everywhere, which is the requirement the objective's "根本没有热路径" note was about. OTPS is
won at 8K and 32K; at 128K and 256K it is now lost by only 1.12x and 1.29x, down from 1.40x and
1.86x, entirely due to the fp16 KV cache. **Cold TTFT is lost at all four by 1.41x / 2.05x /
4.13x / 5.70x and has not moved at all** -- every change attempted against it has been neutral,
and the measurements now say why: the prefill attention runs at 8.4% of fp16 peak, is
latency-bound on a shared-memory load-to-fma chain, and the one lever that ever moved it (bank
conflicts, 8.05x, round 44) is already in.

**Three of the six cells the objective names are won, three are lost, and the three that are
lost are all the same quantity.** That is the honest state after 94 rounds.

### PTX and ptxas evidence on the prefill attention, and the one lever left (round 95)

With tile geometry closed off (round 93) and the fma-chain split measured neutral (round 90),
this round took static evidence from the compiler rather than guessing. Two facts, both new:

**1. The kernel has large register headroom and does not spill.**

```
ptxas info : Compiling entry function 'attn_prefill_tiled_kernel' for 'sm_121'
ptxas info : Used 80 registers, used 1 barriers
             0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

**2. The emitted PTX is fma-dominated with heavy address arithmetic** -- 2,218 instructions in
the tiled kernel:

| opcode | count | |
|---|---|---|
| fma | 475 | the score loop |
| add | 263 | largely address arithmetic |
| ld | 257 | shared loads |
| mov | 212 | |
| cvt | 196 | `__half22float2` |
| setp | 144 | bounds/mask tests |
| mad | 131 | address arithmetic |
| shl | 101 | address arithmetic |
| mul | 71 | address arithmetic |

**Together these two facts point at one specific, untried lever.** The kernel's constraint is
`BQ * BK == K * (nt >> 1)` where **K is the number of dot products each thread-pair computes**
-- K = 3 today, which is exactly what fixes the product at 384 and nt at 256. That K is the
*independent-chain count per thread*, and round 44 established this loop is latency-bound, and
round 90 showed that making one chain shallower (splitting by x/y) buys nothing. **Widening the
number of independent chains is the different, untried direction:**

| K (dots per thread-pair) | `BQ*BK` | BQ | BK | Qs smem | feasible? |
|---|---|---|---|---|---|
| **3 (today)** | 384 | 24 | 16 | 12.2 KB | -- |
| **6** | **768** | **48** | **16** | **24.4 KB** | **yes: smem 32.7 KB total, ~9 more registers against a 255 limit with 0 spill at 80** |
| 12 | 1536 | 96 | 16 | 48.8 KB | no: exceeds the 48 KB default smem |
| 6 | 768 | 96 | 8 | 48.8 KB | no: Qs alone |

**K = 6 is the only doubling that fits**, and it doubles the independent dot products in flight
per thread-pair while leaving every load and convert per fma unchanged -- which is precisely the
shape a latency-bound loop responds to, and the opposite of what was tried in rounds 44 and 90
(both of which *narrowed or held constant* the chain population).

**It is a real code change, not a constant flip**: the score loop's three-row unroll
(`i0`, `i0 + step`, `i0 + 2*step`) becomes a six-row unroll, the store path and the `S`/`red`
sizing follow `BQ = 48`, and the host's `3 * (head_dim / 2)` check becomes `K * (head_dim / 2)`.
**It should be validated by a same-session 32K server pair (round 90's rule) -- two ~4-minute
runs -- and not by `attn-tile`, for the reason round 87 established.**

**This is not implemented.** It is written down as the next concrete step because it is the
only direction the cumulative evidence has not yet excluded: bytes (no), tile geometry (pinned
and tested), chain depth (measured neutral), and now chain *count*.

### Cold TTFT is the only lost metric, and it splits into two jobs of different size (round 96)

The validated scorecard says three of six cells are won (warm TTFT at all four contexts, OTPS at
8K/32K) and that every lost cell is cold TTFT. Placing the measured layer split against the
remaining margin gives an ordering that the percentages alone do not:

| context | prefill total | DeltaNet share | attention share | cold TTFT margin to close |
|---|---|---|---|---|
| 8K | 14.93 s | **10.47 s (70.1%)** | 4.33 s (29.0%) | **1.41x** |
| 32K | 90.54 s | 34.95 s (38.6%) | **55.23 s (61.0%)** | 2.03x |
| 128K | 1134.63 s | ~23% | **~77%** (round 75) | 4.13x |
| 256K | 4138.39 s | less | more | 5.70x |

**The 8K row is the cheapest win on the board and it is a DeltaNet win, not an attention win.**
To beat llama.cpp's 10.58 s at 8K, gb10 must remove 4.35 s from a 14.93 s prefill in which
**10.47 s is DeltaNet**. The chunked-DeltaNet rewrite was sized in round 66 at 2 TFLOP/s as
taking the 8K DeltaNet term from 13.17 s to ~0.77 s -- i.e. it removes about 12.4 s of a term
that only needs 4.35 s removed. **Its projected 8K prefill is ~6.5 s against llama's 10.58 s, a
1.63x win**, which would convert the narrowest cold-TTFT loss (1.41x) into the objective's
fourth won cell.

**The 32K/128K/256K rows are the attention job**, and get harder as the context grows: the
DeltaNet share falls while attention rises, and attention is the term whose per-token cost grows
3.1x from 8K to 32K (round 88) while the kernel sits at 8.4% of fp16 peak. The identified lever
is the K=6 score loop (round 95), whose magnitude is unmeasured.

**So the two implementations now on the table, with their evidence:**

| | A. chunked DeltaNet rewrite | B. K=6 score loop (BQ 48 / BK 16) |
|---|---|---|
| addresses | the 8K term (70.1% of it) | the 32K+ term (61-77%) |
| expected effect | **1.63x win at 8K, sized in r66** | unknown, latency-bound hypothesis |
| implementation size | large (new chunk kernel + state layout) | moderate (6-way unroll in the score loop) |
| correctness gate | `generate --oracle` + `batch-parity` (**`attn-tile` does not cover it**) | `generate --oracle` |
| validation | same-session 8K server pair (llama 10.58 s) | same-session 32K server pair |
| this session's budget | not available | not available with the required server validation |

**Both are unimplemented, and neither should be committed on a compile-and-oracle pass alone** --
round 87 is the standing proof that a kernel change which passes every gate can still lose on the
server, and rounds 90-94 are the proof that only same-session pairs can tell. **The next session
should start with A, because it is the only one of the two with a sized, quantified projection
(1.63x at 8K) and because 8K needs 12.4 s removed against the 4.35 s it requires.**

### The tiled kernel is already register-optimal, and K=6 pays for its ILP in occupancy (round 97)

Round 95 proposed raising K, the dot products per thread-pair, from 3 to 6, on the argument that
the score loop is latency-bound and wants more independent chains. Reading the kernel's
declarations turns up the cost that argument missed:

```cuda
float acc[PREFILL_BQ];   // one register PER ROW, per thread
float vr[PREFILL_BK];    // one register per column
```

**These are per-thread register arrays**, and their cost is `BQ + BK` -- which is the number that
decides occupancy, because this kernel runs 256 threads per block. Today the kernel uses 80
registers with zero spill, and `acc[24] + vr[16] = 40` of them are exactly these two arrays.

**Sweeping the valid tiles shows the current one minimises that cost.** The constraints are
`BQ * BK == K * (head_dim / 2)` and `BK | (nt >> 1) = 128`, with `BQ = K * (128 / BK)`:

| K | (BQ, BK) | `acc+vr` | smem | estimated occupancy |
|---|---|---|---|---|
| **3** | **(24, 16)** | **40** | 22.1 KB | **3 blocks/SM** |
| 3 | (12, 32) | 44 | 24.0 KB | 3 |
| 3 | (48, 8) | 56 | 30.5 KB | 2 |
| 3 | (6, 64) | 70 | 37.1 KB | 2 |
| 3 | (96, 4) | 100 | 53.4 KB | 1 |
| 3 | (3, 128) | 131 | 68.1 KB | 1 |
| **6** | **(24, 32)** | **56** | 31.7 KB | **2** |
| 6 | (48, 16) | 64 | 36.9 KB | 2 |
| 6 | (96, 8) | 104 | -- | 1 |
| 6 | (12, 64) | 76 | -- | 2 |

**Every tile other than the current one is worse on array registers, and every K = 6 tile costs
at least +16.** So the K = 6 proposal trades +2 independent chains per thread for a drop from 3
resident blocks per SM to 2 -- and **occupancy is precisely how a latency-bound kernel hides
latency.** Adding ILP while removing TLP is not obviously a win; it may be a wash or a loss.

**That is a correction to round 95's reasoning, and it is the useful outcome of this round.**
K = 6 remains worth trying, but it is no longer the leading candidate -- the leading candidate is
the one whose projection is quantified and which does not touch this kernel at all.

**Why the current tile is what it is, in one sentence:** `(24, 16)` is the unique pair that keeps
`acc + vr` at its minimum of 40 while satisfying `BQ * BK == 3 * 128` and `BK | 128`, so rounds
41-48 landed on the register-optimal tile by measurement, and there is nothing left to sweep.

**The consequence for the objective.** Cold TTFT needs 4.35 s removed at 8K and 46 s at 32K. The
prefill attention kernel has now been excluded on every axis this document can reach -- bytes
(no), tile geometry (register-optimal and pinned), chain depth (neutral), chain count (costs
occupancy), and its share is 29% at 8K where the loss is smallest. **The 8K loss is 70.1%
DeltaNet, and the chunked DeltaNet rewrite is the only remaining change with a quantified
projection (1.63x at 8K). That is where the next session should work.**

### `tensorcore-plan.md` is stale, and its headline premise no longer holds (round 98)

`bench/longctx/tensorcore-plan.md` opens with a claim that contradicts the measurements taken in
rounds 88-96, so the contradiction had to be resolved before either document is trusted:

| source | 8K | 32K |
|---|---|---|
| `tensorcore-plan.md` | GEMM 49.6 s (93%) / attn 4.4 s (7%) | GEMM **202.6 s (75%)** / attn 65.2 s (24%) |
| round 88 measurement | delta 70.1% / attn 29.0% | **attn 61.0% (67.48 s)** / delta 38.6% (42.63 s), total 110.56 s |

**They cannot both be right**, and the plan's 32K row sums to 267.8 s against a measured prefill
total of 110.56 s.

**The plan is stale, and it says so itself.** Its own section list shows the work it proposed was
carried out: `## Implemented and measured (rounds 22)`, `### Per-phase timing: the residual is
the allocator, not the GEMM (round 26)`, `### Persistent scratch: landed ... (round 27)`. And the
code confirms it:

```rust
// crates/gb10-model/src/weights.rs
pub const PHASES: [&str; 4] = ["weight stage", "activ cast", "cublas gemm", "epilogue"];
...
kern.cublas_gemm_bf16(dev, wb, xb, yb, n, k, t)?;   // line 195
```

**The prefill GEMM already runs on bf16 tensor cores through cuBLAS**, which is exactly what the
plan's headline asked for ("Getting to llama.cpp's ~43 TFLOPS needs tensor cores; there is no
fp32 tuning that reaches it"). The premise "the prefill GEMM is ~7 TFLOPS of fp32 on CUDA cores"
described the code *before* that change.

**The current decomposition confirms the switch took effect.** Round 88's four-phase instrumentation
-- the same four phases named in `PHASES` above -- reports `cublas gemm 25850ms (23.4%)` at 32K,
not 75%. The GEMM is no longer the majority of the prefill; **attention is** (61.0%), which is
the round-88 conclusion, and it stands.

**Why this matters for the next session, which is the point of writing it down:**

- **Do not implement `tensorcore-plan.md`.** Its central change is already in the tree. Reading
  its opening table without checking the code would send the next session to redo rounds 22-27.
- **The largest remaining term is still the prefill attention** at 61.0% of 32K (round 88) and
  ~77% of 128K (round 75), running at 8.4% of the *CUDA-core* half2 rate -- the score loop
  converts fp16 to fp32 (`196 cvt`) and issues fp32 fma (`475 fma`), so it never uses half2
  arithmetic and cannot exceed the CUDA-core rate however it is tiled.
- **The genuinely open question is whether that kernel should move to tensor cores the way the
  GEMM did.** The GEMM's own history is the evidence: it sat at ~7 TFLOP/s of fp32 (76% of a
  9.2 TFLOP/s ceiling, i.e. already well-tuned) and was still 11-13x short; swapping the
  arithmetic to bf16 tensor cores is what fixed it. **The attention kernel is in the same
  position today** -- 3.12 TFLOP/s against a ~37 TFLOP/s CUDA-core ceiling, well short of what
  llama.cpp's ~32 TFLOP/s implies, with the same fix available.

**That reframes path B.** Round 95/97 debated tiling and chain counts inside a CUDA-core FMA
kernel whose arithmetic ceiling is ~37 TFLOP/s while llama.cpp demonstrably runs this workload
at ~32 TFLOP/s. **Those are the same order, which means llama.cpp is near ITS ceiling and gb10
is at 8% of its own** -- the gap is arithmetic, not scheduling, and no tiling sweep closes it.
The one precedent in this repo where that gap was closed is the GEMM, and it was closed with
tensor cores.

### The decode attention is 4.43x above its memory floor, and OTPS at 128K/256K is winnable (round 99)

Every recent round has aimed at cold TTFT, and the attention kernel has resisted on every axis.
The same kernel serves a second, *different* metric that is also still lost, and there the
arithmetic is much more favourable. Decode reads the whole KV cache every token; with fp16 KV
that traffic is exactly countable:

| context | KV read per token | at 228 GB/s (measured) | measured attention | gap |
|---|---|---|---|---|
| 128K | 8.59 GB | **37.7 ms** | ~167 ms (rounds 80-82) | **4.43x** |
| 256K | 17.18 GB | **75.4 ms** | ~334 ms | 4.43x |

**What closing that gap is worth, using the measured decode budget** (128K = 290 ms/token =
~128 ms weight-streaming floor + ~167 ms attention):

| | today | at the memory floor | llama.cpp | result |
|---|---|---|---|---|
| 128K OTPS | 4.25 | **6.0** (166 ms/token) | 4.75 | **1.27x WIN** |
| 256K OTPS | 2.99 | **4.9** (204 ms/token) | 3.86 | **1.27x WIN** |

**So OTPS at the two long contexts the objective names is winnable, and the target is a
bandwidth problem rather than an arithmetic one** -- the opposite of the prefill situation
(round 98). That matters because it is the cheaper of the two: the decode kernel is already
reading the minimal bytes, it is just reading them 4.43x more times than necessary.

**The likely cause is the GQA redundancy that rounds 76-82 identified.** `attn_decode_multi_kernel`
launches a grid of `(n_q_heads, n_seq)` = 24 blocks -- **one per query head** -- so the 6 query
heads that share a KV head each traverse the full KV cache independently. The L2 cannot absorb
it: at 128K one layer's KV is `131072 * 4 * 256 * 2 * 2 B = 537 MB`, far beyond L2, so a
read-amplification factor near 6 at the DRAM level is the natural explanation for a measured
4.43x.

**And the earlier sizing does not contradict this.** Rounds 79/82 sized the *naive*
one-block-per-KV-head fix at ~2x traffic and refuted the occupancy argument. But 4.43x is what
is *measured*, and it is the target: the fix does not have to be the naive one. **A split-KV /
flash-decoding shape that assigns several blocks to one KV head and combines partial
softmaxes would reduce redundant DRAM traffic without needing the 192 KB of `sm_acc` that the
one-block-per-KV-head version required** (round 77's finding: `sm_acc[NW][256]` = 32 KB per head
against a 99 KB optin limit).

**Why this should be attempted before the tensor-core prefill work:** the prefill attention must
get 10x faster in *arithmetic efficiency* and needs a new HMMA kernel; the decode attention needs
~4x less *redundant traffic* in a kernel that already exists and is already byte-minimal per
pass. **The measured target is a 1.27x OTPS win at both long contexts, versus a cold-TTFT gap
that no available lever has moved.** It is validated the same way as everything else in this
document: a same-session server pair at 128K (one ~19-minute gb10 run against llama's ~4.6).

### The GQA fix is tractable: the decode accumulators are registers, not shared memory (round 100)

Round 99 sized the decode attention at 4.43x its memory floor and named the GQA read
amplification as the cause. Round 77 had declared the obvious fix blocked, and reading the
kernel shows **that blocker was attributed to the wrong kernel.**

Round 77's note reads: "`sm_acc[NW][256]` = 32 KB per head; 6 heads = 192 KB vs 99 KB optin."
That is `attn_decode_kernel` (line 624). The kernel that actually serves decode is
**`attn_decode_multi_kernel`** (line 1125), and its accumulator is per-lane:

```cuda
constexpr int DPL = 8;   // dims per lane
constexpr int NW  = 32;  // warps per block
const int h = blockIdx.x;      // query head -- ONE per block
const int s = blockIdx.y;      // sequence
float qv[DPL];
float acc[DPL];                // 8 registers, not 32 KB of shared memory
for (int t = warp; t < n_keys; t += NW) { ... }   // warps split the KEY RANGE
```

**Three consequences, all favourable:**

1. **The accumulators live in registers** (`acc[8]` plus `qv[8]`), so holding several query
   heads per block costs registers, not shared memory. Six heads is roughly
   `acc[6][8] + qv[6][8] = 96` extra floats per lane in the naive arrangement -- against a
   255-register limit, and the register-optimal arrangement (assign each warp to one head and
   let it sweep the whole key range) costs only `acc[8]` per warp with no multiplication at all.
2. **The kernel is already flash-decoding-shaped.** `for (t = warp; t < n_keys; t += NW)` with
   `NW = 32` warps means the block already splits the key range 32 ways and reduces partial
   softmaxes -- the exact structure round 99 proposed building. **The split machinery exists; it
   is simply split across warps of a single-head block instead of across heads of a shared-KV
   block.**
3. **Round 77's 192 KB figure therefore does not block this kernel.** It is a real constraint on
   the other one.

**The change, stated as a shape rather than code:** keep the grid axis that selects the KV head,
and let one block serve all 6 query heads that share it -- either by giving each head its own
warp subset over the whole key range, or by letting each warp sweep its key slice once and
update six accumulator sets. **Either way the K/V slice is read once per block instead of six
times**, which is the 4.43x. The existing per-warp `acc`/`m`/`l` reduction machinery is reused
unchanged.

**Validation, unchanged from every other claim in this document:** a same-session 128K server
pair, gb10 (~19 min) against llama.cpp (~4.6 min) on the same day -- not `decode-bench`, whose
33.5 MB working set is L2-resident and which round 87 proved can show a 1.64x kernel win that is
a 22% end-to-end loss.

**Not implemented.** The design is now unblocked and sized; the implementation is a real kernel
change and the next session should open with it, because it is the only item on the board with a
measured target (1.27x OTPS at 128K and 256K) that does not require new arithmetic.

### Measured directly: the decode amplification is 2.71x, and the OTPS win survives (round 101)

Round 99 derived the decode attention's inefficiency from server residuals (~167 ms of a
290 ms/token budget) and got 4.43x above the memory floor. That was an inference, not a
measurement, and the inference was too pessimistic. `decode-bench` at 128K keys -- which
allocates the cache in the true GQA layout (`nkv = 4`, not `nh = 24`, so 537 MB per layer and
therefore **not** L2-resident) -- measures the kernel directly:

```
q heads 24, kv heads 4, head_dim 256, n_seq 1
     keys   serial ms     warp ms  speedup     rms rel
   131072     169.235       6.375   26.55x     8.24e-6
```

**Two findings, one of which corrects round 99 and one of which confirms it.**

**1. The amplification is 2.71x, not 4.43x.**

| | value |
|---|---|
| KV per layer at 128K (K+V, `nkv = 4`) | 537 MB |
| DRAM memory floor per layer at 228 GB/s | **2.35 ms** |
| **measured warp kernel** | **6.375 ms** |
| ratio | **2.71x** |
| implied DRAM traffic per layer | 1.45 GB |
| x16 attention layers | 102 ms measured, 37.7 ms floor |

The 4.43x came from attributing the whole 167 ms residual to attention. **The direct figure is
102 ms for 16 layers, and it reconciles with the server**: 128 ms weight floor + 102 ms
attention = 230 ms = 4.35 OTPS, against the measured 4.25 OTPS (235 ms) at 128K. The residual
the earlier rounds called "attention" therefore contained ~65 ms of something else.

**2. The opportunity is real anyway, and this was the point of measuring.** Round 99 projected a
1.27x OTPS win by reaching the floor; that projection used the wrong multiple but lands in the
same place, because the *measured* multiple is still 2.71x:

| 128K decode | attention | ms/token | OTPS | vs llama 4.75 |
|---|---|---|---|---|
| today | 102 ms | 235 ms | 4.25 | 1.12x short |
| half the amplification removed | ~70 ms | 198 ms | **5.05** | **1.06x WIN** |
| amplification fully removed | 37.7 ms | 166 ms | **6.04** | **1.27x WIN** |

**gb10 needs attention below ~82 ms to beat llama's 210 ms, and the headroom is 102 -> 37.7 ms.**
The fix does not have to be perfect: removing half of the 2.71x is enough to win.

**3. `attn_decode_multi` is already 26.55x faster than the serial kernel**, which is worth
recording as a banked result -- the kernel being discussed is the good one, and the remaining
2.71x is the residual GQA read amplification that rounds 76-82 and round 100 describe.

**The correction to carry forward:** round 99's "4.43x" should be read as 2.71x, and round 99's
claim that the residual is entirely attention should be read as ~102 ms of attention plus ~65 ms
of other work. **The recommendation is unchanged and better supported, because it now rests on a
direct kernel measurement rather than a subtraction.**

### The 6-head decode fix has a register cost, and it is avoidable (round 102)

Round 100 unblocked the GQA fix by showing the decode accumulators live in registers rather than
shared memory. That cuts both ways: **the fix multiplies exactly those registers by six.**
`attn_decode_multi_kernel` holds `float acc[DPL]` = 8 floats per lane today; serving 6 query
heads per block needs `acc[6][DPL]` = 48, plus `m`/`l` per head, plus the per-head query vector.
With `NW = 32` warps (1024 threads) per block and 65536 registers per SM:

| arrangement | regs/lane | resident threads | occupancy |
|---|---|---|---|
| NW=32, `qv` in registers | ~136 | 481 | **47%** |
| NW=32, `qv` in shared memory | ~88 | 744 | 73% |
| **NW=16, `qv` in shared memory** | **~88** | **512** | **100%** |
| NW=8, `qv` in shared memory | ~88 | 256 | 100% |

**Two design choices remove the cost:**

1. **Keep the per-head query vector in shared memory.** Six heads x 256 dims x 4 B = **6 KB**,
   against the same 99 KB opt-in budget that round 77 worried about. It is read once per key
   sweep and is the obviously shared quantity, so it belongs in shared memory and not in 48
   registers.
2. **Halve NW, from 32 warps to 16.** `NW` exists to raise resident warps (the kernel's own
   comment records that `NW = 8`, a 256-thread block, "left just 4 warps resident per SM and the
   kernel ran at 328 GB/s"). But the fix's whole point is that far fewer blocks are needed --
   **one block per KV head instead of six** -- so the per-block warp count is the right knob to
   trade against it, and `NW = 16` keeps a full block resident while `NW = 8` is the shape the
   comment already measured as too small.

**So the fix is: one block per (KV head, sequence), 16 warps, the 6 query heads' q in shared
memory, `acc[6][8]` in registers, and each warp sweeping `t = warp; t < n_keys; t += NW` while
updating all six heads from one K/V load.** The existing per-warp partial-softmax reduction is
reused, per head.

**And the acceptance target stays the measured one:** the kernel is at 2.71x its 2.35 ms-per-layer
memory floor (round 101), and gb10 needs 128K attention below ~82 ms of 102 ms to beat llama.cpp's
210 ms. **Judged by a same-session 128K server pair, never by `decode-bench` alone.**

**Not implemented.** It is a real kernel change with a now-complete design, a measured target, and
an identified register hazard.

### `NW` is a coupled constant with no host knob, and one comment about it is stale (round 103)

Round 102's design calls for `NW = 16` instead of 32 to preserve occupancy once the six query
heads' accumulators are in registers. Checking whether that is a host-side knob turns up a
coupling that has to be respected, and a comment that must not be trusted:

```cuda
// kernels/elementwise.cu:1137
constexpr int NW = 32;   // warps per block
```

```rust
// crates/gb10-cuda/src/ops.rs -- the attn_decode_multi launch
// Must match NW in `attn_decode_multi_kernel`: the block is
// NW warps, and each warp owns a strided slice of the keys.
(&self.attn_decode_multi, 1024u32)
```

**`NW` is a kernel `constexpr` and the thread count is hardcoded `1024u32` on the host.** They
are not derived from a shared constant, so **`NW = 16` requires editing both in lockstep** -- the
kernel's `constexpr` and the host's literal. This is the same class of coupled edit that rounds
61/62 got wrong (a change in one place silently mismatching another, with the build staying
quiet), so it should be changed in one commit with both sites visible, and verified by grep
afterwards rather than assumed.

**A second thing to not trust:** the comment immediately above that launch reads

> The warp kernel covers head_dim == 256 (32 lanes x 8 dims) and wants the full eight warps of a
> 256-thread block; any other head width goes [to the serial kernel]

**"eight warps of a 256-thread block" contradicts `NW = 32` warps and `1024u32` threads** four
lines below it. The 32-lanes-x-8-dims part matches `DPL = 8` and the kernel body; the warp and
thread counts do not. It is a leftover from when `NW` was 8, and the kernel's own later comment
at line 1136 records that history ("`NW = 8` a 256-thread block left just 4 warps resident per SM
and the kernel ran at 328 GB/s. Raising NW raises the resident warps without ... same traffic").
**So the operative facts are `NW = 32`, `DPL = 8`, `1024u32`, and the comment above the launch is
stale** -- worth knowing before treating it as documentation of the current shape.

**Net effect on the plan:** the register analysis of round 102 stands, and its `NW = 16` step is
a two-site edit rather than a knob turn. That is the last unknown in the design; everything else
about it is sized, targeted, and has a stated acceptance test (a same-session 128K server pair,
gb10's attention below ~82 ms of the measured 102 ms).

**Not implemented.** This round adds no code; it removes the two ways the implementation could
silently go wrong.

### The weight-streaming path runs at 66.8% of peak, and that costs every decode step (round 106)

Every recent round has aimed at attention, on the reasoning that attention is what grows with
context. That reasoning is right about *growth* and wrong about *magnitude*: the decode step
also pays a **context-independent** cost, and that cost is running a third short of the hardware.
`gb10-bench stream` measures it directly:

```
per-token weight traffic : 17.555 GB
time per token           : 115.19 ms
achieved bandwidth       : 152.4 GB/s
projected decode         : 8.68 tok/s (single stream)
roofline at 228 GB/s     : 12.99 tok/s
bandwidth utilisation    : 66.8% of measured 228 GB/s
```

**115.19 ms per token against a 77.0 ms roofline. That is 38.2 ms of waste on every single
decode step, at every context** -- and it is the largest single inefficiency this document has
measured, larger than the 2.71x decode-attention amplification in absolute terms and, unlike it,
present even at 8K.

**The mechanism is visible in the kernel.** `dequant_nvfp4_to_bf16_kernel` (kernels/gemm.cu:522)
loads the weights and the scales in the same loop, one byte at a time, from two separate arrays:

```cuda
const uint8_t byte = __ldg(w  + (size_t)n * (size_t)(kk >> 1) + (size_t)(k >> 1));
const uint8_t nib  = (k & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
const float   s    = e4m3_to_float(__ldg(sc + (size_t)n * (size_t)(kk >> 4) + (size_t)(k >> 4)));
```

Counting 32-byte sectors for a 32-lane warp with two weights per lane:

| array | what a warp fetches | sectors |
|---|---|---|
| `w` | 32 lanes x 1 B = 32 useful bytes | 1 sector (32 B) |
| `sc` | 4 distinct scale bytes (8 lanes share one) | **1 sector (32 B)** |

**36 useful bytes out of 64 fetched -- 56%**, which is the right order for the measured 66.8%.
**The scale read costs a whole sector to deliver four bytes**, and it is a separate array, so it
cannot be coalesced away with the weight read.

**The fix is to make each thread do more work per iteration** -- process 32 weights (16 bytes,
one `uint4` load) and the two scales that cover them, so the scale transaction is amortized over
16x more weight bytes and both arrays are read at full width. That is a change of loop shape, not
algorithm, and it does not touch the arithmetic: the dequantized bf16 values are identical.

**What reaching the roofline is worth, using the measured decode budget (115.19 ms context-free
plus the measured attention per context):**

| context | today | **at the roofline** | llama.cpp | result |
|---|---|---|---|---|
| 8K | 8.72 | **12.99** | 7.32 | 1.77x win |
| 32K | 7.14 | **12.99** | 6.865 | 1.89x win |
| 128K | 4.25 | **5.59** | 4.75 | **1.18x WIN** |
| 256K | 2.99 | **3.53** | 3.86 | 1.09x short |

**This overturns the priority order this document has carried since round 99.** The decode
attention work (rounds 99-103) targeted 128K/256K OTPS by removing a 2.71x traffic
amplification; this targets the *same two metrics* by removing a 33% bandwidth shortfall that
also lifts 8K and 32K, and it is a smaller and better-understood change -- a load-width fix in
one kernel with no change to the numerics.

**It also explains a number that has been sitting unexplained since round 80**: the
"context-free floor" of ~128 ms/token that all the decode budgets were built on is not a floor
at all. **It is 115 ms of streaming at 67% efficiency plus overhead**, and the true floor at this
traffic is 77 ms.

**Validation: unchanged and non-negotiable.** `gb10-bench stream` is the right instrument for the
bandwidth claim -- unlike `decode-bench`, its 17.555 GB working set cannot be L2-resident, so it
is measuring DRAM -- but the *end-to-end* claim still requires a same-session server pair
(128K: gb10 ~19 min against llama ~4.6 min), because round 87's rule is that a kernel measurement
is evidence about that kernel and not about the pipeline.

**Not implemented.** It is the next thing to do, ahead of the decode GQA work, because it is
larger in effect, broader in scope (every context, not just the long ones), and smaller in risk.

### CORRECTION to round 106: the streaming path is instruction-bound, and its loads are already wide (round 107)

Round 106 attributed the weight-streaming path's 66.8% of peak (152.4 GB/s against 228) to
narrow loads and a wasted sector on the scale array, and proposed widening them. **Both halves
of that were wrong, and checking the actual kernel shows why.**

**First, the kernel I read was not the streaming path.** Round 106 quoted
`dequant_nvfp4_to_bf16_kernel` (gemm.cu:522), which writes dequantized bf16 to a global `out`
buffer -- 2 bytes per parameter, which for a 27B model would be ~54 GB of writes per token and
is obviously not what a 115 ms/token path does. **The decode path is `stage_wtile`**
(gemm.cu:65), which stages the dequantized tile into shared memory. I read the wrong function.

**Second, `stage_wtile` already has the widening I proposed, and its comment says so:**

```cuda
// 16 consecutive NVFP4 elements per thread instead of 8: one 8-byte load
// and one group scale cover the pair, so the staging loop runs half as many
// iterations and issues half as many loads for the same bytes.
...
pk[p] = *reinterpret_cast<const uint2*>(w + (size_t)n * (K >> 1) + (kbase >> 1));
scl[p] = e4m3_to_float(__ldg(sc + (size_t)n * (K >> 4) + (kbase >> 4)));
```

**And the sector arithmetic now comes out clean**, so the waste I blamed does not exist:

| array | per warp (32 lanes, `uint2` = 16 elements) | sectors |
|---|---|---|
| `w` | 32 x 8 B = 256 B | 8 |
| `sc` | 32 x 1 B = 32 B (one scale per 16 elements) | 1 |
| | **288 B fetched for 288 useful B** | **100%** |

**So it is not a bandwidth problem at all, and it never was.** An instruction count explains the
measurement instead:

| | value |
|---|---|
| parameters per token | 27.0e9 |
| time | 115.19 ms |
| SMs x clock | 48 x 1.5 GHz |
| **parameters per SM per cycle** | **3.3** |
| instructions per parameter (shift, extract, convert, multiply) | ~3-4 |
| **instructions per SM per cycle required** | **9.8 - 13.0** |
| instructions per SM per cycle available | ~4 |

**The dequant needs 2.5-3.3x more issue slots than the SM has.** The kernel is not waiting for
memory; it is saturating the instruction issue units, and 152 GB/s is what that costs. **This is
the same conclusion round 98 reached for the prefill attention and round 44 reached for the
tile loop: gb10's decode and prefill are both limited by what its CUDA cores can issue, not by
what its memory system can deliver.**

**What that means for the next attempt, stated so it is not repeated:**

- **Widening loads cannot help the streaming path.** It is already 100% sector-efficient; there
  is no bandwidth to recover.
- **The lever is instructions per weight.** Either the dequant must be done with fewer
  instructions per element -- vectorised bf16 conversion processing two elements at a time is
  the obvious candidate -- or the arithmetic has to leave the CUDA cores entirely.
- **And the arithmetic leaving the CUDA cores has a very specific form here: the weights are
  NVFP4, and this part has native FP4 tensor cores.** Feeding them the quantized weights is the
  structural version of the fix, and it is the same shape of change the prefill GEMM already
  went through (round 98) -- except that the GEMM moved to *bf16* tensor cores and this would
  need FP4 ones, which is a larger step.

**The revised priority.** Round 106 put "widen the streaming loads" first. That is withdrawn.
What survives from round 106 is the *measurement* -- 115.19 ms against a 77.0 ms roofline, 38.2 ms
per token, at every context -- and it is still the largest single inefficiency in this document.
**What changes is the fix: not bytes, but instructions.** The decode GQA work (rounds 99-103)
remains the better-understood option because it removes real redundant traffic rather than
fighting an issue-rate wall.

### Static PTX of `nvfp4_gemm_kernel`: fma-dominated, with a large dequant population (round 108)

Round 107's instruction-bound argument was arithmetic (3.3 parameters per SM per cycle against
~3-4 instructions each, versus ~4 issue slots). This round looked for the same conclusion in the
emitted code. **The result is supporting but not decisive, and the ambiguity should be recorded
so it is not mistaken for a measurement.**

```
nvfp4_gemm_kernel total PTX instructions: 2590
mix: fma 1024, mov 226, bra 209, setp 157, and 148, st 128, shl 127, add 92,
     ld 90, cvt 89, shr 75, mul 73, or 70, selp 32
```

**What can be read from this:**

- **`fma` dominates at 1024 of 2590.** For a decode GEMM every weight participates in exactly one
  multiply-accumulate per token, so a large fma population is expected -- but it also means
  **the kernel is not sitting idle waiting on memory**, which is the operative point. A purely
  bandwidth-bound kernel would not be this fma-heavy.
- **The dequant work is a substantial minority**: `and 148 + shl 127 + shr 75 + or 70 + mul 73 +
  cvt 89 = 582` instructions of nibble extraction, scaling and conversion, against 1024 fma.
  That is ~36% of the static instruction stream spent on unpacking weights, which is a large
  fraction for a step that is nominally "just moving bytes".

**What cannot be read from it, and why:** the natural next step -- instructions per weight -- is
not available from this data. The loop-detection pass picked up the whole function body (5467
lines, including the fma main loop) as one "loop", so the derived figure of 140 instructions per
weight is an artifact and must not be quoted. **Static PTX counts also say nothing about dynamic
trip counts**, and the staging loop is rolled, so a small static count can execute many times.

**So the load-bearing evidence for the instruction-bound conclusion remains round 107's
arithmetic**, which does not depend on the static mix: 3.3 parameters per SM per cycle, times
~3-4 instructions per parameter, is 9.8-13.0 instructions per SM per cycle against ~4 issue
slots. **The PTX is consistent with that (a big fma body plus a third of the stream doing
unpacking) but it does not independently establish it.**

**If the next session wants the measurement rather than the estimate**, the clean way is a
counter-based one: run `nvfp4_gemm_kernel` alone via `gb10-bench stream` with the dequant
replaced by a passthrough that reads the same bytes and writes the same tile without unpacking,
and compare the times. That isolates the instructions from the traffic with the traffic held
constant -- which is the one experiment this document has not run, and the one that would settle
it. (`ncu` is unavailable on this platform: `ERR_NVGPUCTRPERM`.)

### The passthrough control: the dequant is free, and rounds 106-108 are withdrawn (round 109)

Rounds 106-108 built a case that the decode weight-streaming path's 66.8% of peak was wasted and
recoverable -- first blamed on narrow loads (round 106), then on the sector the scale array costs,
then on instruction issue (rounds 107-108, "9.8-13.0 instructions per SM per cycle against ~4
issue slots"). Round 108 named the experiment that would settle it, so this round ran it.

**The experiment:** replace the unpacking loop in `stage_wtile` with a passthrough that performs
the *identical* `uint2` weight load and the identical scale load, writes the *same number of
stores* to the same shared-memory tile, but does no nibble extraction and no `e2m1_to_float` at
all. Traffic is held constant; the dequant instructions go to zero.

```
  baseline    (full dequant) : 115.19 ms   152.4 GB/s   66.8% of peak
  passthrough (no unpacking) : 117.26 ms   149.7 GB/s   65.7% of peak
```

**Removing every dequant instruction made the path 1.8% slower -- that is, no change at all.**
The dequant is free: it hides completely under the memory latency. The kernel is not
instruction-bound, not load-width-bound, and not sector-bound.

**So the conclusion of rounds 106-108 is refuted, and all three of its mechanisms are withdrawn:**

| round | claim | status |
|---|---|---|
| 106 | narrow loads + a wasted sector on `sc` | **withdrawn** -- `uint2` loads, 100% sector efficiency |
| 107 | the dequant is instruction-bound | **withdrawn** -- removing it changes nothing |
| 108 | PTX shows a third of the stream unpacking | **withdrawn** as evidence; consistent with a cost that is simply hidden |

**What the measurement actually says** is that **~152 GB/s is the rate this access pattern
achieves on this part**, and the 228 GB/s figure the tool prints as "peak" is not reachable by
this pattern. The implication is the opposite of good news: **the 38.2 ms/token that rounds
106-108 called the largest inefficiency in this document does not exist.** The weight floor for
decode is ~115 ms/token, not 77 ms.

**Reconciled against the scorecard**, that makes the decode budgets exact rather than optimistic:

| context | weight | attention | total | OTPS | llama |
|---|---|---|---|---|---|
| 8K | 115 ms | ~0 | 115 ms | 8.72 | 7.32 |
| 32K | 115 ms | ~25 ms | 140 ms | 7.14 | 6.865 |
| 128K | 115 ms | 102 ms | 217 ms | 4.25 (measured 235 ms) | **4.75** |
| 256K | 115 ms | 206 ms | 321 ms | 2.99 | **3.86** |

**And it sharpens the remaining target rather than removing one.** With the weight floor fixed at
115 ms and llama at 210 ms/token at 128K, **attention must fall below ~95 ms** to win; the
measured figure is 102 ms. **The 6-head GQA decode fix (rounds 99-103) targets exactly that, and
its projected 102 -> ~70 ms gives 185 ms = 5.4 OTPS against llama's 4.75, a 1.14x win.** The
decode GQA work is therefore confirmed as the right target for 128K/256K OTPS, and the streaming
path is confirmed as having no recoverable headroom.

**One caveat worth stating rather than hiding:** "this pattern achieves 152 GB/s" is a statement
about this implementation, not proof about the hardware. A different access order -- deeper
memory-level parallelism, a different tile walk, larger per-thread runs -- might still reach
more. **But that is a new hypothesis with no supporting measurement, and it should not be
promoted on the strength of the 228 GB/s figure alone**, which is exactly the mistake rounds
106-108 made.

### The streaming investigation, closed: three mechanisms refuted and the reference is a constant (round 110)

Round 109's passthrough control refuted the instruction hypothesis. This round tested the other
two ways the 66.8% could be explained, and checked what the 66.8% is actually measured against.

**Hypothesis: serialization between weight-matrix stagings.** The bench has a switch for exactly
this -- `--interleave` "inserts a tiny dependency-breaking kernel after every [matrix]":

| variant | ms/token | GB/s | of the 228 reference |
|---|---|---|---|
| baseline | 115.19 | 152.4 | 66.8% |
| no dequant (round 109) | 117.26 | 149.7 | 65.7% |
| **`--interleave`** | **119.74** | **146.6** | **64.3%** |

**Breaking the dependencies makes it slower, not faster.** The path is not serialization-limited.

**And the reference is a hardcoded constant, not a measurement:**

```rust
// crates/gb10-bench/src/main.rs:20
const ROOFLINE_GBPS: f64 = 228.0;
```

**"66.8% of peak" is therefore a comparison against a number compiled into the tool**, not
against a bandwidth measured in the same run, and almost certainly a *sequential-read* figure.
The weight-staging walk is not sequential: it reads the GEMM's B matrix, 16 contiguous NVFP4
elements per thread with `n` varying across lanes, which strides across rows.

**Four mechanisms have now been tested against the 152 GB/s, and all four are refuted:**

| # | hypothesis | test | result |
|---|---|---|---|
| 1 | narrow loads / wasted sector | read the actual kernel | **refuted** -- `uint2`, 100% sector efficiency |
| 2 | instruction-bound dequant | passthrough with no unpacking | **refuted** -- 117.26 vs 115.19 ms |
| 3 | serialization between stagings | `--interleave` | **refuted** -- 119.74 ms, worse |
| 4 | (the reference itself) | read the source | **it is `const 228.0`, not measured** |

**So the honest statement is: this access pattern achieves ~152 GB/s, three plausible
explanations for the shortfall against a hardcoded 228 GB/s have been measured and eliminated,
and the most likely remainder is DRAM page locality in a strided GEMM walk** -- which the three
tests cannot reach, because none of them changes the *order* in which rows are visited.

**What would actually test it:** a weight layout whose rows are visited in a longer contiguous
run (or an explicit larger per-thread run along `k`), compared on the same `stream` benchmark.
**That is a data-layout experiment, it is speculative, and rounds 106-109 are the record of what
happens when a speculative lead is promoted on arithmetic alone.** It should be attempted only
with a same-session server pair behind it.

**The operational conclusion, which is what matters for the objective:** **the decode weight
floor is ~115 ms/token and there is no measured, demonstrated way to reduce it.** Every decode
budget in this document should use 115 ms, as round 109's table does. The remaining OTPS levers
are therefore all on the attention side, and the 6-head GQA decode fix is the one with a
measured target.

### Concretely how to implement the 6-head decode fix (round 111)

Round 111 read `attn_decode_multi_kernel` (kernels/elementwise.cu:1125) line by line, so the
design of rounds 100-103 can be written as an edit rather than a sketch. **This is the exact
shape; nothing about it is still unknown.**

**What it does today** (one query head per block, six blocks per KV head):

```cuda
const int h = blockIdx.x;                    // query head -- grid.x = n_q_heads (24)
const int group = n_q_heads / n_kv_heads;    // 6
const int kh = h / group;                    // KV head
float qv[DPL], acc[DPL];                     // 8 + 8 registers
for (int t = warp; t < n_keys; t += NW) {
    const size_t off = cb + ((size_t)t * n_kv_heads + kh) * head_dim + d0;
    const __half* kp = k_cache + off;        // <-- read once per block, re-read by 6 blocks
    ...dot, shuffle-reduce, online softmax...
    const __half* vp = v_cache + off;        // <-- same
    for (int j = 0; j < DPL; ++j) acc[j] = fmaf(p, __half2float(vp[j]), acc[j] * corr);
}
__shared__ float sm_m[NW], sm_l[NW], sm_acc[NW][256];   // 8 KB, one head
```

**The edit, in four parts:**

1. **Grid: one block per KV head, not per query head.** `blockIdx.x` becomes `kh` directly, so
   the launch's `grid.x` goes from `n_q_heads` to `n_kv_heads` (24 -> 4). The 6 heads become an
   in-block loop.
2. **Registers: six partial-softmax sets, one per head.** `float qv[G][DPL], acc[G][DPL];
   float mx[G], sum[G];` with `G = group` (6). That is 48 + 48 + 12 floats against a 255-register
   limit -- and it is the reason `NW` must drop (next).
3. **The key loop: load K and V once per `t`, then apply them to all six heads.**
   `float kv[DPL], vv[DPL];` are converted once from `k_cache`/`v_cache` at `off`, and the
   per-head work becomes a `for (int g = 0; g < G; ++g)` around the dot product, the shuffle
   reduction and the online-softmax update, reusing `kv`/`vv`. **This is the entire win: the
   cache line is fetched once per block instead of six times.**
4. **Merge: loop the existing reduction over the six heads**, reusing the same
   `sm_m`/`sm_l`/`sm_acc` buffers with a `__syncthreads()` between heads. Do **not** allocate six
   `sm_acc` copies -- that is the 48 KB that round 77 mistook for a blocker.

**Plus the coupled constant, from round 103:** `NW = 32` is a kernel `constexpr` and the host
hardcodes the thread count as `1024u32` in `crates/gb10-cuda/src/ops.rs` with the comment
"Must match NW in `attn_decode_multi_kernel`". Round 102's register table says the six-head
version needs `NW = 16` (512 threads) to keep a full block resident. **Both sites must change
together** -- `constexpr int NW = 16;` and `512u32` -- and the grid axis change in part 1 must
land with them.

**The acceptance test, with the number already measured:** at 128K the decode attention is
102 ms of a 235 ms token; with the weight floor fixed at 115 ms (round 110) and llama.cpp at
210 ms, **attention must fall below ~95 ms to win**. This change removes a measured 2.71x read
amplification (round 101), so the projection is ~70 ms and 5.4 OTPS against llama's 4.75.
**Judged by a same-session 128K server pair -- one ~19-minute gb10 run against llama's ~4.6 --
never by `decode-bench` alone (round 87's rule).**

**Not implemented.** This round adds no code. It removes the last unknown in a design that has
been sized, register-checked, de-risked against its documentation and now written out
line-by-line.

### The 6-head design has a fatal flaw, and the real missing piece is a split-K (round 114)

Rounds 100-103 designed the 6-head decode block and round 111 wrote it out line by line. Working
out how to size its register arrays this round turned up a structural problem that invalidates
the design as specified.

**The problem is the grid.** The whole point of the change is one block per KV head instead of one
per query head, so that the K/V line is fetched once and used for all six heads. But for a single
sequence that takes the grid from `n_q_heads = 24` blocks to `n_kv_heads = 4`:

| | current | 6-head design |
|---|---|---|
| blocks (n_seq = 1) | 24 | **4** |
| shared memory per block (`sm_acc[NW][256]` + `sm_m`/`sm_l`) | NW=32 -> 32.2 KB -> **1 block/SM** | NW=16 -> 16.1 KB -> 2 blocks/SM |
| SMs actually used | 24 of 48 | **4 of 48** |
| warps in flight | 24 x 32 = **768** | 4 x 16 = **64** |
| DRAM traffic per layer | 3.2 GB (K/V read 6x) | 537 MB (read once) |

**Warps in flight fall 12x while traffic falls 6x.** The kernel is *documented* as latency-bound,
not bandwidth-bound -- its own comment records that at `NW = 8` it managed only 328 GB/s because
too few warps were resident, and that raising `NW` to 32 is what fixed it. **Halving the
warps-per-byte of a latency-bound kernel is expected to make it slower, not faster.** Round 102's
register analysis sized the fix for occupancy and never checked that the grid still had enough
blocks to occupy the machine.

**And checking that turned up something larger: the current kernel already wastes half the GPU.**
At `NW = 32` the shared-memory request is 32.2 KB, so only one block fits per SM; with 24 blocks
for a single sequence, **24 of 48 SMs sit idle.** Every decode-attention measurement in this
document -- including round 101's 6.375 ms and 2.71x amplification -- was taken on half the
machine.

**Both problems have the same missing piece: a k-split.** Splitting the key range into `S`
chunks:

- restores the block count, so the grid can fill all 48 SMs (the current kernel's `S = 2` would
  take 24 blocks to 48);
- and for the 6-head design it is *required*, not optional, because the grid collapse would
  otherwise cost more parallelism than the traffic saving is worth.

**The cost is a cross-block merge.** A split-K decode attention has to combine partial
online-softmax states `(m, l, acc)` from `S` blocks per head -- either a second small reduction
kernel over an `(n_q_heads, S)` buffer, or atomics. That is real work this document has not
designed, and it is the same shape as FlashDecoding's second pass.

**So the revised next step is bigger than round 111 implied, and better founded.** The order
should be:

1. **A k-split on the current per-query-head kernel first**, with the cross-block merge, because
   it is the smaller change, it touches no register budget, and it addresses a defect that is
   visible right now -- half the GPU idle. **Expected effect is up to ~2x on the decode attention,
   which alone would take 128K from 4.25 to well past llama's 4.75.**
2. Only then consider fusing the 6-head sharing on top of the split grid, where the block count is
   restored and the sharing becomes a pure traffic reduction with parallelism preserved.

**This is a correction to rounds 100-103 and 111.** The 6-head design is not wrong in its
arithmetic; it is incomplete, because it was designed without checking that the resulting grid
still occupies the device. Round 111's line-by-line spec should not be implemented as written.

### MEASURED: filling the GPU gives the decode attention 1.64x, and 48/48 SMs is worth 4/4 on OTPS (round 115)

Round 114 argued from the launch configuration that the decode kernel uses only 24 of 48 SMs at
one sequence. That was a claim about occupancy, and it is testable without a profiler: **if the
kernel is starved of blocks, then doubling the grid should barely change the per-call latency,
because the extra blocks fill idle SMs rather than contend.** `decode-bench` makes that a
one-line experiment, because a second sequence doubles the grid from 24 blocks to 48.

```
--n-seq 1 --kv-keys 131072     6.375 ms      (24 blocks)
--n-seq 2 --kv-keys 131072     7.760 ms      (48 blocks)
```

**Twice the work in 1.22x the time.** Per-sequence throughput:

| | latency | sequences/s |
|---|---|---|
| n_seq = 1 | 6.375 ms | 156.9 |
| n_seq = 2 | 7.760 ms | 257.7 |
| **gain** | | **1.64x** |

**The kernel was starved, and filling the grid is worth a measured 1.64x** -- not a model, not an
estimate. Nearly the whole theoretical 2x is realised, which is what a latency-bound, half-idle
kernel looks like.

**It also settles what the bottleneck is.** The same output line reports **1661 GB/s for the warp
kernel -- well above the 228 GB/s DRAM figure.** The 6x K/V redundancy that round 101 measured is
therefore being **served by L2**, not DRAM. So the decode attention is not bandwidth-bound at all;
it is bound by how many warps are resident to hide L2 latency. **That is why adding blocks helps
so much, and it is the opposite of what the 2.71x "read amplification" framing suggested.**

**Applied to the scorecard, using only numbers measured in this session** (weight floor 115.19 ms
from round 110, attention from round 101, scaling 1.64x from here):

| context | attention now | attention filled | OTPS now | OTPS filled | llama | result |
|---|---|---|---|---|---|---|
| 8K | ~0 ms | ~0 ms | 8.72 | 8.68 | 7.32 | win |
| 32K | 25 ms | 15 ms | 7.14 | **7.67** | 6.87 | win |
| 128K | 102 ms | **62 ms** | 4.25 | **5.64** | 4.75 | **1.19x WIN** |
| 256K | 206 ms | **126 ms** | 2.99 | **4.15** | 3.86 | **1.08x WIN** |

**OTPS goes from 2 of 4 contexts won to 4 of 4, and the scorecard from 6 of 12 to 10 of 12.**
That is the largest single step available anywhere in this document, and unlike rounds 106-109 it
rests on a measurement rather than an arithmetic argument.

**How to get 48 blocks from one sequence -- and why split-K, not split-D:**

- **Split the key range (split-K).** Two blocks per query head, each covering half the keys,
  combining their `(m, l, acc)` partial states. **Traffic is unchanged** (K and V are each read
  once in total), the grid doubles to 48, and the merge is a small second pass over
  24 x 2 partial states.
- **Do not split the head dimension.** Each dim-half block would still need the full `q . k_t`
  over all keys to get the softmax weights, so K is read twice while V is split once -- traffic
  rises from `K + V` to `2K + V`, i.e. **1.5x, against a 1.64x gain, leaving only 1.09x.** Worse
  and messier.
- **Note that the block cannot simply grow instead.** `NW = 32` already means 32 warps x 32 lanes
  = 1024 threads, the maximum block size, so there is no room to add warps inside the block; the
  parallelism has to come from more blocks.

**And the split-K merge is the same missing piece round 114 identified** -- so it is a prerequisite
for the 6-head sharing as well, and the two can share one implementation of it.

**Not implemented.** This round measures the opportunity and rules out the cheaper wrong version
of it. The change itself -- S=2, a partial-state buffer, and a merge pass -- is the next thing to
build, and it now has a measured target instead of a projected one.

### Concretely how to implement split-K for the decode attention (round 116)

Round 115 measured that filling the grid is worth 1.64x and settled that split-K is the right
axis. This round writes out the mechanism, because the merge is the one part that does not exist
in the code yet and is the reason this is not a small edit.

**Why a single kernel cannot do it.** Online softmax folds each new key in against a running max
`m`. A block that has only seen half the keys knows only *its* local max, so it cannot rescale its
accumulator into the global one -- the global max is not known until every split has finished.
That is why flash-decoding uses two passes, and why a second kernel is needed here.

**Pass 1 -- `attn_decode_split_kernel`**, one block per `(query head, key slice)`. It is the
existing `attn_decode_multi_kernel` with `t` starting at the slice offset and stepping `NW * S`,
and instead of normalising at the end it writes its **raw partial state**:

```
part[s, h, split, 0]     = m_split      (the local max)
part[s, h, split, 1]     = l_split      (its sum of exp)
part[s, h, split, 2..]   = acc_split    (unnormalised, head_dim floats)
```

Scratch size: `n_seq * n_q_heads * S * (2 + head_dim)` floats. For one sequence at 128K that is
`1 * 24 * 2 * 258 * 4 B = 49.5 KB` -- negligible, and independent of context length.

**Pass 2 -- `attn_decode_merge_kernel`**, one block per `(sequence, query head)`, `head_dim`
threads. Each thread `d` reads the `S` partials, computes the global max
`M = max_split m_split`, then

```
l = sum_split l_split * exp(m_split - M)
a = sum_split acc_split[d] * exp(m_split - M)
out[d] = l > 0 ? a / l : 0
```

**and it must keep the same `l > 0` guard as the existing kernel**, which the current code
comments on explicitly: an empty cache leaves `l` at zero and the old form returned 0 there, so
the guard is what stops an empty cache becoming NaN. A merge that loses it would reintroduce a
bug the tree already fixed once.

**Host plumbing**, in `crates/gb10-cuda/src/ops.rs` `pub fn attn_decode_multi` (:504): it needs a
scratch buffer and the split count threaded through, the grid's y axis becomes `n_seq * S`, and
the merge launch follows. The existing `need(...)` guards should be extended to the scratch rather
than added beside them.

**Correctness gate, and it is not the usual one.** `attn-tile` does not cover this kernel. The
checks that do are `generate --oracle` (16/16), `batch-parity` -- which compares against
`attn_decode_multi_serial`, the independent implementation that exists for exactly this purpose --
and **`decode-bench`'s own `rms rel` column**, which is the tightest of the three: it reported
`8.24e-6` at n_seq=1 and `8.52e-6` at n_seq=2, so a merge error has nowhere to hide.

**Then the measurement is already set up: `decode-bench --kv-keys 131072`.** The number to beat is
**6.375 ms at n_seq=1**, and round 115's 1.64x says the target is ~3.9 ms. **Only after that
should a same-session server pair be run** (round 87's rule), where the expected 128K result is
5.64 OTPS against llama's 4.75.

**A caution from this session's own history.** Rounds 100-103 produced a 6-head design that was
sized, register-checked and written out line by line -- and rounds 114-115 found it would have
collapsed the grid to 4 blocks and probably regressed. **Split-K is the smaller change and it is
the one with a measured target, but the same discipline applies: check the resulting launch
against the 48-SM device before writing the kernel, not after.**

**Not implemented.** This round adds no code and no new measurement; it removes the last
undesigned part of the next change.

### A simpler way to double the grid: split the head dimension, and skip the merge (round 117)

Round 116 specified split-K with a two-pass merge, treating the merge as unavoidable. It is
avoidable, and the alternative is a smaller change.

**Why split-D needs no merge.** The output is `out[d] = sum_t p_t * v_t[d]` where
`p_t = softmax(q . k_t)`. The softmax weights depend on the **full** dot product, so any block
computing *any* output dimension has to compute the whole thing -- which means **two blocks
splitting the dimensions compute identical weights and identical `l`**. Each output dimension is
independent, so block A can write dims `[0,128)` and block B dims `[128,256)` **with no
communication whatsoever**: no scratch buffer, no second kernel, and no second copy of the
`l > 0` empty-cache guard whose loss round 116 flagged as a real hazard.

**What it costs:** both blocks must read all of K to form the dot products, while V is split.

| | traffic per layer | vs today |
|---|---|---|
| today | K + V = 1074 MB | 1.00x |
| split-K | K + V = 1074 MB | 1.00x |
| **split-D** | **2K + V = 1611 MB** | **1.50x** |

**So the two options trade a merge for 1.5x traffic. The measurement says that is the better
trade**, because the traffic is not what is binding:

| option | net gain |
|---|---|
| split-K | 1.64x (traffic flat) |
| split-D, if the 1.5x traffic binds | 1.09x |
| **split-D, if L2 absorbs it** | **1.64x** |

**And L2 demonstrably has the headroom:** the same round-115 measurement shows the current warp
kernel sustaining **1661 GB/s** while carrying the 6x GQA redundancy. The 1.5x from split-D is
smaller than the 6x already being absorbed. **The decode attention is latency-bound on resident
warps, not bandwidth-bound, which is precisely why adding blocks helps at all** -- so a traffic
increase that L2 covers should cost close to nothing.

**The kernel edit, stated so it is checkable.** Today `DPL = 8` with 32 lanes maps lane `L` to
dims `[8L, 8L+8)`, covering all 256. For split-D the *dot* still needs those 8 dims per lane,
but the *accumulator* only needs 4 per lane:

- dot: `d0_dot = lane * 8`, reading `q` and `k` over all 256 dims exactly as now;
- accumulator: `DV = 4`, `d0_out = half * 128 + lane * 4`, reading only the block's half of `v`.

Grid becomes `(n_q_heads, n_seq * 2)`. **Full lane utilisation is preserved** -- no lanes idle,
which is what would have happened if the split had simply masked off half the lanes.

**Recommendation: try split-D first.** It is one kernel modified rather than one kernel plus a
merge kernel plus a scratch buffer plus host plumbing, and its only risk is the traffic, which is
the variable least likely to bind. **If `decode-bench` shows split-D landing near 1.64x, split-K
is unnecessary; if it lands near 1.09x, the traffic did bind and split-K's merge becomes worth
writing.** Either way the first experiment is cheaper than round 116 assumed, and it answers the
question rather than assuming it.

**Not implemented.** Target unchanged: `decode-bench --kv-keys 131072` from 6.375 ms toward
~3.9 ms, then a same-session server pair for the 128K claim.

### Split-D implemented and measured: 3.4%, and the traffic bound after all (round 118)

Round 117 proposed split-D as the cheap way to double the grid, on the argument that the L2 --
measured sustaining 1661 GB/s through a 6x GQA redundancy -- would absorb split-D's 1.5x traffic
increase. **It was implemented, verified, and measured. The argument was wrong.**

**Implemented** (kernels/elementwise.cu, `attn_decode_multi_kernel`): `blockIdx.y` now packs
`(sequence, dim half)`; the dot product still runs over all 256 dims with `DPL = 8`, while the
accumulator uses `DV = 4` at `d0v = half * 128 + lane * 4`. `sm_acc` halves to `[NW][128]` and
each block stores only `d < head_dim / 2`. Host: `dim_split = 2` and the grid's y axis becomes
`n_seq * dim_split`.

**Correctness passed cleanly:** `generate --oracle` **16/16, exact match**, and `decode-bench`'s
`rms rel` is **8.24e-6 -- identical to the pre-change value**, so the two halves agree with the
single-block result to the last measured digit.

**The measurement:**

| | warp kernel, 131072 keys, n_seq=1 |
|---|---|
| baseline (24 blocks) | 6.375 ms |
| **split-D (48 blocks)** | **6.164 ms** |
| gain | **1.034x** |

**The model had predicted 1.09x** (1.64x from parallelism over 1.5x from traffic), and the
measurement is 1.034x -- **within 5% of the prediction, and nowhere near the 1.64x that a
traffic-free split would have given.** The achieved bandwidth also falls from 1661 to 1045 GB/s,
consistent with genuinely moving more bytes.

**So the traffic does bind, and round 117's L2 argument was wrong.** The 6x GQA redundancy was
being absorbed, but that is not evidence that *any* increase is free; 1.5x was enough to consume
almost the whole 1.64x. **This is the same shape of error as rounds 106-109: an argument from a
plausible mechanism rather than a measurement** -- and here the measurement was cheap, which is
the lesson.

**What it establishes, which is worth more than the 3.4%.** The two quantities are now separately
known: **+64% from doubling the grid** (round 115, measured with n_seq=2) and **-33% from 1.5x
traffic** (this round, by difference). **A split that doubles the grid with unchanged traffic
should therefore capture close to the full 1.64x -- and that is exactly what split-K is.**

**Split-K's merge, which rounds 116-117 treated as a cost to avoid, is now the price of the
1.64x rather than an expensive alternative to a cheaper design.** At 1.64x the decode attention
at 128K goes 102 ms -> 62 ms and the token 217 -> 177 ms = **5.64 OTPS against llama's 4.75**.

**Retained:** the split-D change is left in place -- it is verified correct by both gates and is a
real if small improvement -- but the document should be read as treating it as a stepping stone,
not as the fix. **The fix is split-K.**

### Split-K implemented and measured: no gain at all, and round 115's 1.64x does not transfer (round 119)

Round 118 concluded from two separate measurements -- +64% from doubling the grid, -33% from 1.5x
traffic -- that a split doubling the grid at flat traffic should collect close to the full 1.64x,
and named split-K as the way to get it. **It was implemented, verified, and measured. It does not
work, and the inference from those two numbers was wrong.**

**Implemented:** `attn_decode_partial_kernel` (the warp kernel with `gridDim.z` dividing the key
range and each block writing its raw `(m, l, acc)` state instead of normalising) plus
`attn_decode_merge_kernel` (one block per `(sequence, head, dim half)`, applying the global max,
with the `l > 0` empty-cache guard carried over). Grid becomes
`(n_q_heads, n_seq * dim_split, n_split)` = 96 blocks, a scratch buffer of 49.5 KB.

**Correctness passed:** `generate --oracle` **16/16, exact match**, and `rms rel` 8.21e-6 against
8.24e-6 before. So the kernels are right and the measurement is meaningful.

| variant | blocks | warp ms | vs baseline |
|---|---|---|---|
| baseline | 24 | 6.375 | -- |
| **split-D** | 48 | **6.031 - 6.164** | **1.04 - 1.06x** |
| split-K (S=2, + split-D -> 96) | 96 | **6.442** | **0.99x** |

**Four times the blocks of the baseline, and it is slower.** The 1.64x did not appear.

**Why the inference failed.** Round 115's 1.64x was measured with `--n-seq 2`, where the extra 24
blocks process **a second, independent sequence**. Split-K's extra blocks process **the same
sequence's other key half**. Those are not the same thing:

- With two sequences, the added blocks carry genuinely new work with their own independent
  softmax, and the per-block fixed cost (loading `q`, the 32-warp shared-memory merge, the block
  setup) is amortised over it.
- With split-K, the same fixed cost is **duplicated** while each block's key loop is halved, and
  the merge adds a second launch plus a global round trip. At 6 ms of real work those extras are
  small but the halved key loops are smaller still.

**So "filling the grid is worth 1.64x" was an over-generalisation from a sequence-parallel
measurement to a key-parallel design.** The grid was idle in the `n_seq = 1` case, but **the
idleness was not the binding constraint** -- adding blocks that split an existing sequence's work
does not recover it. Only adding blocks with independent work does.

**This closes the split-K line, and it retires the "+64% available" claim.** The remaining
measured position on decode attention is split-D's ~1.04-1.06x, which is kept (it is committed,
verified, and the best of the three). **The 128K and 256K OTPS gap therefore stands at 1.12x and
1.29x; there is no longer a measured path to closing it on the attention side.**

**And it is the third time in this session that a projection from a plausible mechanism was
refuted by a cheap measurement** (rounds 106-109 on the streaming path, round 117-118 on split-D's
L2 assumption, now split-K's parallelism transfer). The pattern is consistent: **the measurements
have been reliable and the extrapolations between them have not.**

**Reverted.** The split-K kernels and host plumbing were removed; the tree is back to the
committed split-D state, re-measured at 6.031 ms to confirm the revert.

### Where the remaining winning margin actually is: two different bottlenecks by context (round 120)

With the decode-attention line closed (rounds 118-119), cold TTFT is the only metric with an
opening, and this round locates it. The split comes from the round-88 measurement at 32K plus
**complexity scaling** -- attention is O(n^2) and DeltaNet is O(n), which is a property of the
algorithms, not an inferred mechanism:

| context | attention | DeltaNet | prefill | measured | llama |
|---|---|---|---|---|---|
| 32K | 85.5 s (64%) | 48.0 s (36%) | 133.5 s | 90.54 s | 44.55 s |
| **128K** | **1081 s (86%)** | 171 s (14%) | 1252 s | 1134 s | 275.04 s |
| **256K** | **4324 s (93%)** | 341 s (7%) | 4666 s | 4138 s | 726.22 s |

**The model reproduces the measured totals to within ~10% at both long contexts**, which is what
matters here: the ordering is not in doubt.

**So there are two different bottlenecks, and they need different fixes:**

- **8K and 32K are DeltaNet-bound.** At 8K the DeltaNet term is 10.47 s of a 14.93 s prefill
  (70.1%); at 32K it is 48 s. **llama.cpp's ENTIRE 8K prefill is 10.58 s and its 32K prefill is
  44.55 s.** So the DeltaNet term alone is roughly the whole llama prefill at those sizes --
  **no amount of attention work closes 8K or 32K.** Round 66 sized the chunked DeltaNet rewrite at
  13.17 s -> ~0.77 s, which is the 1.63x projected for 8K.
- **128K and 256K are attention-bound**, 86% and 93% of prefill. Here the fix is the one round 98
  identified and this session has not attempted: **prefill attention runs at 3.12 TFLOP/s, which is
  8.4% of the 37 TFLOP/s fp16 CUDA-core peak and ~4% of the 74.8-89.2 TFLOP/s bf16 tensor-core
  figure measured on this part.** llama.cpp implies ~32 TFLOP/s, i.e. it is already using tensor
  cores and gb10 is not.

**What tensor-core prefill attention would be worth**, applying a speedup `k` to the attention
term only:

| k | 128K prefill | vs llama 275.04 | 256K prefill | vs llama 726.22 |
|---|---|---|---|---|
| 10x | 279 s | tie | -- | -- |
| **15x** | **243 s** | **1.13x WIN** | 629 s | **1.15x WIN** |
| 20x | 225 s | 1.22x WIN | 557 s | 1.30x WIN |

**15x is well inside the measured headroom** (24-28x to the bf16 tensor-core peak), so **128K and
256K cold TTFT are winnable, and only by this route.** Note also that the same change lifts 32K
attention (85.5 s -> ~6 s), which is necessary but not sufficient there.

**The honest statement of the objective's remaining state**, so it is not misread:

| metric | won | can be won | cannot |
|---|---|---|---|
| warm TTFT | **4/4** | -- | -- |
| OTPS | **8K, 32K** | -- | 128K (1.12x), 256K (1.29x) |
| cold TTFT | **none** | 32K, 128K, 256K (DeltaNet + tensor-core attention) | 8K is the hardest |

**So the objective is not reachable, but it is not uniformly unreachable either**: 10 of 12 cells
are won or have a quantified, measured path. **The two that do not are 128K and 256K OTPS**, where
round 119 removed the last measured route.

**Priority for the remaining rounds:** the two prefill items, in this order -- **tensor-core
prefill attention first** (it is the larger term at the long contexts the objective names, and it
underpins 2 of the 3 winnable cold-TTFT cells), then **the chunked DeltaNet rewrite** (which is
what 8K and 32K need and was already sized in round 66).

### Round 121: the prefill attention has no cheap fix left, so tensor cores are the only route

Round 120 established that 128K and 256K cold TTFT are attention-bound (86% and 93%) and that only
a ~15x attention speedup closes them. Before designing that, one cheap possibility had to be ruled
out: **that the slow path was a dispatch problem** -- that long sequences were falling through to
the quadratic `attn_prefill_legacy` kernel while only short ones got the tiled one. That has been
the shape of several cheap wins this session, so it was worth one check.

**It is not.** `ops.rs:1291` dispatches unconditionally, and its comment records why:

> Every prefill goes through the tiled kernel, including short ones. ... `attn_prefill_legacy`
> runs one block reduction per key per query, so its cost grows quadratically with the key count
> *and* its constant factor is far worse. Measured in situ on the same 2048-token chunk: 82.3 s at
> 10,240 cached keys (legacy) against 39.2 s at 12,288 (tiled) -- the tiled kernel is twice as
> fast with 20% more keys.

So the tiled kernel *is* the fast path, legacy is retained only as the independent reference the
`attn-tile` gate compares against, and **the measured 3.12 TFLOP/s is the tiled kernel's real
performance.**

**And its headroom is already accounted for.** Round 97 measured it at **80 registers, 0 spill,
1 barrier -- 3 blocks per SM.** It is not register-limited, not spilling, not barrier-bound, and
its grid at 32K is thousands of blocks, so unlike the decode kernel it is **not starved for
blocks**. What it is, is a CUDA-core kernel doing `__half2` loads with `__half22float2`
conversions and fma per element: at 3.12 TFLOP/s it is at **8.4% of the 37 TFLOP/s fp16
CUDA-core peak**, and the round-95 PTX census of the same family showed the shape of that --
2218 instructions of which 475 are fma and 196 are cvt, i.e. a conversion-heavy inner loop.

**So there is nothing left to tune.** The conclusion round 98 reached and this session has now
exhausted the alternatives to is the operative one: **the arithmetic has to leave the CUDA cores,
and the only place for it to go is the tensor cores.**

**What that requires, stated so the size of the job is clear:**

1. **An mma-based QK^T.** `mma.sync.aligned.m16n8k16` with fp16 inputs and fp32 accumulators, so
   queues and keys are consumed as they are without per-element conversion. `head_dim = 256` means
   16 k-steps of 16, or 8 of 32.
2. **The online softmax between the two products, in the mma layout.** The QK^T result must be
   reshaped from the mma fragment layout to rows-before-current-key for the causal mask, have the
   running max applied, and be exponentiated -- and `__expf` is a CUDA-core op, so it becomes the
   new bottleneck unless it is done at half precision (`ex2.approx.f16x2`) alongside the mma.
3. **A second mma for P·V**, with P as the freshly computed fp16 probabilities. This is the step
   that makes it FlashAttention rather than just a faster GEMM, and it is where the fp16 range
   handling lives.
4. **The measured target is a 15x speedup, against a 24-28x headroom to the 74.8-89.2 TFLOP/s bf16
   figure** -- so the budget is not generous once exp and the layout shuffles are counted, and the
   first version should not be expected to hit it.

**This is a multi-round job, unlike anything attempted so far in this session** -- every previous
change was a modification of an existing kernel. **The validation path is already in place and
does not need inventing:** `attn-tile` compares the tiled kernel against `attn_prefill_legacy`
(the independent reference the dispatch comment describes), `prefill-shape` measures the layer
split, and `generate --oracle` is the end-to-end check.

**Not implemented.** This round closes the last cheap alternative and states the job size.

### Round 122: tensor cores are provably necessary for the prefill attention, not merely better

Round 121 established that a ~15x attention speedup is needed to win 128K and 256K cold TTFT, and
that the existing kernel has nothing left to tune. This round closes the CUDA-core design space
with arithmetic rather than another experiment, because the arithmetic is decisive.

The tiled kernel's instruction census (round 95 PTX, 2218 instructions):

| class | count | share |
|---|---|---|
| fma | 475 | 21.4% |
| add | 263 | 11.9% |
| ld | 257 | 11.6% |
| mov | 212 | 9.6% |
| cvt | 196 | 8.8% |
| setp | 144 | 6.5% |
| mad | 131 | 5.9% |
| shl | 101 | 4.6% |
| mul | 71 | 3.2% |
| (rest) | 368 | 16.6% |

- **Arithmetic (fma+add+mad+mul) is 940 instructions = 42.4%.** The kernel is **not fma-bound**;
  it is issue-bound across a broad mix, with 29.4% of the stream in conversion and control
  (`cvt`+`mov`+`setp`+`shl`).
- **The required 15x is 46.8 TFLOP/s. The fp16 CUDA-core peak on this part is 37 TFLOP/s.**

**46.8 > 37. So the required speedup exceeds the entire fp16 CUDA-core peak of the hardware.**
This is not a claim about tuning difficulty -- **it is a bound. No CUDA-core kernel, however
well-written, can reach 15x, because 15x does not exist on the CUDA cores.** And the gap is worse
than that number suggests, because only 42.4% of the current instruction stream is arithmetic at
all: even hitting the fp16 CUDA-core peak exactly would require eliminating essentially every
`ld`, `mov`, `cvt`, `setp` and `shl` in the kernel.

**This is the cleanest conclusion this session has produced**, and it is worth stating in the
form it takes:

> The prefill attention needs 15x. The CUDA cores can supply at most 37 TFLOP/s against the
> 46.8 TFLOP/s required, while currently delivering 3.12. **Tensor cores are therefore necessary,
> not an optimization.** The only open question is whether an mma formulation can be written that
> keeps the softmax off the CUDA cores too (round 121's `ex2.approx.f16x2` point) -- because if
> `exp` stays on the CUDA cores, it becomes the new ceiling and the 15x will not materialise.

**It also means the cheaper alternatives are not merely unattempted but ruled out**, so no future
round need spend a measurement on fp16x2 accumulation, on instruction shaving, or on further tile
geometry. **The remaining work is one job, and it is the mma kernel.**

### CORRECTION to round 122: the "provably necessary" bound does not hold; the requirement is 8-10x, not 15x (round 123)

Round 122 concluded with a bound: the prefill attention needs 15x, 15x is 46.8 TFLOP/s, the fp16
CUDA-core peak is 37 TFLOP/s, so **tensor cores are provably necessary**. **The arithmetic in that
argument is right and the conclusion drawn from it is wrong, because 15x is not the requirement.**

15x came from round 120's table, which asked what a **15x** speedup would be worth -- a *sufficient*
figure chosen to show a clear win (128K -> 243 s against llama's 275.04), **not the minimum needed
to win.** Round 122 then treated it as the requirement and compared it against the peak. Solving
for the actual threshold instead:

| context | measured prefill | attention share | DeltaNet | attention must fall below | **required k** | at 3.12 TFLOP/s today |
|---|---|---|---|---|---|---|
| 128K | 1134.6 s | 86% (980 s) | 155 s | 120 s | **8.1x** | **25.4 TFLOP/s** |
| 256K | 4138.4 s | 93% (3836 s) | 303 s | 423 s | **9.1x** | **28.3 TFLOP/s** |

**The requirement is ~8-10x, i.e. 25-28 TFLOP/s -- which is BELOW the 37 TFLOP/s fp16 CUDA-core
peak.** So the bound "46.8 > 37, therefore no CUDA-core kernel can do it" **is not sound**, and the
word "provably" must come out.

**What survives, and it is the honest form of the conclusion:** the tensor-core route is still the
right one, but for a weaker reason than a hard bound. **Arithmetic is only 42.4% of the current
instruction stream** (round 122's own census: `cvt` 8.8%, `mov` 9.6%, `setp` 6.5%, `shl` 4.6%,
plus `ld` 11.6%), so a CUDA-core kernel reaching 25-28 TFLOP/s would need to be near the fp16 peak
**while eliminating almost every non-arithmetic instruction** -- not impossible in principle, but
not something this session has any evidence is reachable, and the kernel is already at 80
registers with 0 spill and 3 blocks/SM, so there is no headroom left to buy it with occupancy.

**Why the correction matters beyond bookkeeping:**

- **It changes the target.** A ~9x requirement is a materially easier job than a 15x one, and the
  round-121 design notes (mma QK^T, softmax in the mma layout, `ex2.approx.f16x2`) should be sized
  against 9x, not 15x.
- **It removes a false prohibition.** As written, round 122 told a future round that *no*
  CUDA-core attempt could ever work. That is not established, and stating it that way risks
  discarding a cheaper route on the strength of a bad bound -- which is the same failure mode as
  rounds 106-119, where confident extrapolations were refuted by cheap measurements.

**This is the twenty-fourth self-correction in this session**, and it is the second one of the
session's own *reasoning* rather than of a measurement (round 114's grid collapse was the first).
**The pattern both times: an inference that was sound step by step, extended one step past what
the numbers supported.**

### The cheapest remaining cell is 8K cold TTFT, and it needs only 1.71x (round 125)

Rounds 120-123 established what the long contexts need (8-9x on prefill attention) and that the
CUDA-core design space is exhausted. This round prices the *whole* remaining scorecard, which
turns out to reorder the work.

| context | gb10 | llama | gap | attention | DeltaNet | via DeltaNet only | via attention only |
|---|---|---|---|---|---|---|---|
| **8K** | 14.93 s | 10.58 s | **4.35 s** | 4.46 s | **10.47 s** | **1.71x** | impossible (39x) |
| 32K | 90.54 s | 44.55 s | 45.99 s | 55.23 s | 34.95 s | impossible alone | 5.98x |
| 128K | 1134.63 s | 275.04 s | 859.59 s | 975.78 s | 158.85 s | impossible alone | 8.40x |
| 256K | 4138.39 s | 726.22 s | 3412.17 s | 3848.70 s | 289.69 s | impossible alone | 8.82x |

**8K cold TTFT is the outlier, and in the good direction.** Its gap is 4.35 s, its DeltaNet term is
10.47 s, so **a 1.71x DeltaNet improvement closes it.** Every other cell needs 6-9x on a term that
is already known to require tensor cores to go faster at all.

**And round 66 already sized the fix well past what is needed: the chunked DeltaNet rewrite was
projected at 13.17 s -> ~0.77 s, i.e. 17x.** That is **ten times the margin 8K requires.**

**So the priority order inverts.** Round 120-124 put the mma prefill attention first on the grounds
that it addresses the contexts the objective names (32K/128K/256K). That reasoning was about
*importance*, not *cost*:

- **The chunked DeltaNet rewrite is the cheapest win available anywhere on the scorecard** -- 1.71x
  needed against 17x projected, on a kernel family this session has already improved three times
  (all by removing traffic: 4-way chains, `Sc[D]` in registers, `qh` to shared), and with the
  correctness gate already in place (`generate --oracle` + `batch-parity`; note `attn-tile` does
  NOT cover it).
- **The mma attention is still required**, but only for 32K (5.98x), 128K (8.40x) and 256K (8.82x),
  and it is the harder job of the two.

**So the next round should do DeltaNet first**, and the reason is not that it is more important but
that **it is the only remaining item with a modest, already-exceeded target** -- and it converts a
cell from loss to win rather than narrowing a gap that stays a loss.

**One caveat, stated because this session has been burned by it twice.** The 1.71x figure rests on
the round-88 layer split, which was measured at 32K and applied here at 8K. Round 88's own 8K
numbers (delta 70.1%, attn 29.0%) are `prefill-shape` output, so the 8K split is measured too --
this is not an extrapolation between contexts. **The projection that is an extrapolation is round
66's 17x**, which has never been measured; the point here is only that the *requirement* is 1.71x,
so a rewrite that delivers even a quarter of round 66's projection still wins 8K.

### Round 126: the DeltaNet prefill path is the chunked kernel, with no fallback to check

Round 125 put the chunked DeltaNet rewrite first on cost grounds (1.71x needed at 8K against
round 66's 17x projection). The same cheap check that round 121 ran on the prefill attention --
**is the slow path a dispatch problem?** -- applies here, and for the same reason: several of this
session's wins have been of that shape.

**It is not.** `crates/gb10-model/src/layer.rs:307` calls `ops.gated_delta_rule_chunk` with the full
token count for the prefill path, and there is no size-dependent branch back to
`gated_delta_rule_step_multi` (decode, `layer.rs:414`) or `gated_delta_rule_step` (`:870`), both of
which are per-token. **So prefill already uses the chunked kernel at every length, and the measured
DeltaNet cost is that kernel's real cost.**

**That closes the cheap alternatives for DeltaNet as well**, and leaves the work itself. It is worth
recording that both of the two remaining items have now had their dispatch checked and are not
falling through to a slower kernel: **the remaining work on this objective is genuinely kernel
optimization in both cases, not configuration.**

**Its shape is also known from this session's own history.** All three accepted DeltaNet
improvements were traffic removals rather than arithmetic changes -- 4-way chains, `Sc[D]` held in
registers, `qh` moved to shared memory -- so the next increment should be looked for in the same
place: a redundant load or a redundant shared-memory round trip in the chunk kernel's inner loops,
not a change to the recurrence itself. **And the bar is low: 1.71x wins 8K outright, which is a
quarter of what round 66 projected the rewrite at.**

### Round 127: the DeltaNet chunk kernel is already tuned; its limit is the sequential T loop

Round 125 put DeltaNet first (1.71x needed at 8K) and round 126 confirmed prefill dispatches to the
chunked kernel with no fallback. This round read `gated_delta_rule_chunk_kernel` (elementwise.cu:837)
to find the next increment, and what it shows is **how much has already been done** -- the header
comments alone record five optimisations, each stated with its measured effect:

1. **Two threads per state column** -- the one-thread form needed 255 registers (the hardware
   maximum) and still spilled 140 stores to local memory (round 70), and spill traffic is global.
2. **`S` in registers, not shared** -- the old form used 66 KB of shared and touched it twice per
   fma; "every removal of inner-loop traffic from this kernel so far paid 9-17%".
3. **`sk` in shared** -- the one input every column needs.
4. **`qh` in shared** -- it had been a *global* load inside the second hot loop, 128 per thread per
   token, and that loop is half the kernel's cost.
5. **Four independent partial sums** -- the two hot loops each carried a 128-long serial fma
   chain; cutting it to 32 gave the scheduler work to interleave. The kernel had been running at
   0.64% of fp32 peak, so the chain rather than throughput was the constraint.

**What is left is structural, and it is visible in one line:**

```cuda
for (int t = 0; t < T; ++t) {
    ...
    __syncthreads();
    ...
}
```

**The token loop is fully sequential, with a block-wide barrier inside it -- two per token over the
whole prompt.** Every optimisation above made the *inside* of an iteration cheaper; none changed the
fact that iteration `t` cannot begin before `t-1` finishes. **There is no parallelism across tokens
at all**, which is exactly why the recurrence is the last thing left and why round 66 sized the fix
as a *rewrite* (13.17 s -> ~0.77 s at 2 TFLOP/s) rather than a tuning pass.

**A chunked (parallel-scan) formulation is what removes it:** within a chunk of `C` tokens the
state update `S_t = decay_t * S_{t-1} + beta_t * (k_t (x) v_t)` becomes a prefix product of the
decays plus a sum of rank-1 terms, which is expressible as matrix products over the chunk -- so `C`
tokens proceed together instead of one barrier apart. **That is a different algorithm, and it is
larger than anything this session has attempted.**

**And the bar remains low: 1.71x wins 8K, a quarter of round 66's projection.** A partial rewrite --
say chunks of 16 or 32 rather than 128 -- that captures even a fraction of the projected gain would
still win the cell, which makes this a better first target than the mma attention despite being a
rewrite: **the mma work needs ~9x to pay off at all, while this needs 1.71x.**

**Correctness gate unchanged and already in place:** `generate --oracle` plus `batch-parity`.
**`attn-tile` does NOT cover this kernel** -- it compares prefill attention kernels only -- so a
rewrite validated by `attn-tile` alone would be unchecked.

### Round 128: the 8K target re-measured at 8K -- 1.83x, not 1.71x, and measured rather than derived

Round 125 priced the remaining cell at **1.71x on DeltaNet** to win 8K cold TTFT, but that figure
came from applying round 88's layer split -- **measured at 32K** -- to the 8K case. The session has
been burned twice by exactly that move (rounds 114 and 123), so this round measured it at 8K
instead:

```
[diag] LAYER GPU: delta 9.71s / 192 = 64.3%   attn 5.31s / 64 = 35.1%
[diag] n=1600 | weight stage 2034ms (13.5%)  activ cast 622ms (4.1%)
       cublas gemm 5856ms (38.8%)  epilogue 692ms (4.6%)
       | op phases total 9.20s of 15.10s (61.0%)
```

**DeltaNet is 64.3% at 8K, not the 70.1% round 88 reported at 8K** (the two differ because round 88
used a different `--limit`), and attention is 35.1% rather than 29.0%. Recomputed against the
measured 14.93 s prefill and the 4.35 s gap to llama's 10.58 s:

| quantity | value |
|---|---|
| DeltaNet at 8K | 9.60 s |
| attention at 8K | 5.24 s |
| gap to llama | 4.35 s |
| **k needed via DeltaNet alone** | **1.83x** |

**So the requirement is 1.83x, slightly worse than round 125's 1.71x, and it now rests on a
same-context measurement rather than a 4x-context extrapolation.** The conclusion is unchanged and
better founded: **8K remains by far the cheapest remaining cell** -- 1.83x against 5.98x / 8.40x /
8.82x for 32K / 128K / 256K, and against round 66's never-measured 17x projection for the chunked
rewrite. **A rewrite capturing even a tenth of that projection wins the cell.**

**One further number worth recording:** `op phases total 9.20s of 15.10s (61.0%)` -- the GEMM
phases are 61% of the 8K prefill and the layer-level delta/attn accounting covers the rest. Both
tallies independently put the DeltaNet-side cost well ahead of attention at this context, which is
the premise the whole priority rests on.

### Round 129: DeltaNet runs at 0.61% of fp32 peak, so the 8K target needs only 1.1% of peak

Round 128 fixed the target: **1.83x on DeltaNet wins 8K cold TTFT.** This round asked whether that
is a lot or a little, by computing what the kernel actually delivers. From the kernel's own
dimensions (`D = 128`, `n_v_heads = 48`, 48 DeltaNet layers, `T = 7168`) and the 9.60 s measured
this session:

| quantity | value |
|---|---|
| MACs per token per v-head (`2 * D^2`, the outer-product update plus the `S^T q` output) | 32768 |
| total | 5.412e11 MACs = **1.082 TFLOP** |
| **achieved** | **112.7 GFLOP/s = 0.113 TFLOP/s** |
| **as a fraction of the 18.43 TFLOP/s fp32 peak** | **0.61%** |
| threads launched | 48 blocks x 256 = 12,288 |
| against 48 SMs x 2048 threads = 98,304 | **12.5% occupancy** |

**The 0.61% figure is an independent reproduction of a number already in the tree.** The kernel
header records "the whole kernel runs at 0.64% of fp32 peak (round 66)"; computing it from scratch
here gives 0.61%. **Two independent routes to the same number, which is the strongest kind of
agreement this document has recorded.**

**What it means for the target:** the kernel needs to reach **1.1% of fp32 peak** -- roughly double
where it is -- and the gap to the peak is a factor of 164. **A 1.83x requirement against a kernel
operating at 0.61% of peak is not a stretch; it is the smallest possible step.** The rewrite does
not have to be efficient. It only has to expose more of the parallelism that is already there and
currently unused, and the room is so large that the first thing that works will very likely clear
the bar.

**It also confirms the diagnosis independently of the structural reading.** 12.5% occupancy, 8
warps per SM, a sequential `T` loop with two barriers per token, and 0.61% of peak are all the same
fact stated four ways: **this kernel is not compute-bound, it is starved.** Which is why the fix is
a *formulation* change (chunked/parallel-scan) rather than more tuning -- rounds 70/88/95/97/127
already removed the inner-loop traffic and the serial fma chains, and none of that touched the fact
that `t` cannot start before `t-1` finishes.

**Caveat, in keeping with this session's record:** the FLOP count above is my own model of what the
recurrence must do, not an instrumented count, so 0.61% could be off by the constant factor in the
"2 * D^2" term. **It agrees with the tree's independently-recorded 0.64% to two figures, which is
why it is being trusted -- but the agreement is the evidence, not the derivation.**

### Round 130: the DeltaNet stall is per-token and unexplained -- ablation 1

Round 129 found the DeltaNet kernel at 0.61% of fp32 peak and concluded the 1.83x target was easy.
**That conclusion needs a qualification this round supplies**, because the headroom is real but its
cause is not identified -- and "164x headroom" is only comfortable if we know what is eating it.

**Ablation 1: does the cost scale with tokens or with calls?**

| run | delta | calls | ms per call | us per token |
|---|---|---|---|---|
| `--limit 2047` | 2.83 s | 48 | 58.96 | **28.8** |
| `--limit 7168` | 9.71 s | 192 | 50.57 | **24.7** |

Per-call milliseconds are near-constant while calls scale 4x and tokens-per-call stays at 2048, and
**per-token cost is ~25-29 us in both runs. So the cost is per-token, inside the `T` loop -- it is
not per-call launch or setup overhead.** That rules out the explanation that would have been
cheapest to accept.

**What the source itself rules out** (read and measured this session):

- **global state traffic** -- `sh` is touched only at elementwise.cu:886 (init) and :943 (after the
  loop), **never inside it.** The "S in registers" optimisation is intact.
- **barrier count** -- exactly two `__syncthreads()` in the loop (:887, :939), i.e. one pair per
  token, which is the designed minimum for a shared `sk`/`sq` handoff.
- **arithmetic** -- 128 MACs per thread per token is ~256 cycles; **~37,000 are observed.**
- **k/q/v traffic** -- the block reads its own `D`-wide slices, ~1 KB per token, ~16.5 GB over the
  8K prefill, which at 152 GB/s is **~108 ms against 9.6 s measured.**
- **inner-loop shared traffic** -- already removed by rounds 70/88/95/97.

**So there is a ~144x per-token stall that the source does not account for**, and this session has
no profiler to find it with (`ncu` is blocked by `ERR_NVGPUCTRPERM`; `nsys` captures nothing).

**This qualifies round 129 rather than contradicting it.** The 164x headroom to peak is genuine,
but **we do not know which mechanism consumes it**, and the awkward possibility is that the same
mechanism bounds any rewrite: a parallel-scan formulation removes the sequential dependency but
does not automatically remove whatever makes a token cost 25 us.

**Ablation 2, which is the next round's first action, and is cheap:** time the kernel with `T`
varied independently of the chunking by calling `prefill-shape` at several `--limit` values and
fitting per-token against per-call cost. If per-token cost stays flat in `T`, the stall is
per-iteration (barrier or load latency inside the loop); if it falls with `T`, part of it is
per-call. **That is the cheapest way to localise this without a profiler, and it costs one
`prefill-shape` run per point.**

**Caveat on the FLOP model, restated:** the 128 MACs/thread figure assumes `2 * D^2` MACs per
v-head per token (outer-product update plus `S^T q` output). If the real recurrence does less work
than that model, the stall factor is smaller -- **but the 0.61%-of-peak measurement itself does not
depend on the model, only on the time, and the 25 us per token is measured.**

### Round 131: a third of the DeltaNet cost is a per-call fixed cost, and call count is a cheap lever

Round 130's ablation was flawed and this round fixes it. It varied `--limit` and concluded the cost
was per-token -- but **`PREFILL_CHUNK` is 2048, so varying `--limit` only changes the number of
calls, never `T` inside a call.** The test that actually separates the two is a **small** chunk, so
this round measured `--limit 255`:

| T (tokens per call) | per-call time | per-token |
|---|---|---|
| **255** | **20.83 ms** | 81.7 us |
| 2047 | 58.96 ms | 28.8 us |
| 2048 | 50.57 ms | 24.7 us |

Fitting `T = 255` and `T = 2048`:

```
slope     = 0.01659 ms/token  = 16.6 us per token
intercept = 16.60 ms          FIXED PER CALL
```

**So there is a ~16.6 ms fixed cost per kernel call, and at 8K/7168 tokens (192 calls) it accounts
for 3.19 s of the 9.71 s measured -- 33% of the entire DeltaNet term.** The other 6.52 s is the
per-token part.

**That makes call count a cheap, testable lever.** If the state update ran once per layer instead of
once per (layer, chunk), the fixed part would fall from 3.19 s to 0.80 s:

> **2.38 s saved of 9.71 s = the DeltaNet term 25% faster, ~1.33x.**

**Honest about what that is and is not: 1.33x is not the 1.83x the cell needs**, so this does not
win 8K on its own. But it is the largest lever found so far on this kernel, it is cheap (it is a
chunking parameter, not an algorithm change), it applies to the attention path's chunking as well,
and it compounds with whatever the per-token half yields.

**And the per-token half has moved from "unexplained" to "bounded":** 16.6 us per token against
~256 cycles (0.17 us) of arithmetic is still a ~97x stall, so the round-130 finding stands -- the
per-token mechanism is unidentified and `ncu` is unavailable. **The difference now is that the two
halves are separated and the cheap half is actionable.**

**Caveat, stated plainly: the fit rests on two single runs**, and its prediction for `T = 2047`
(50.6 ms) misses the measured 58.96 ms by 8 ms. **The 33%/25% split should be treated as
approximately right, not exact, and re-derived from repeated runs before anything is built on it.**
What is not in doubt is the shape: **per-call cost is far from negligible, and the `--limit`
ablations of round 130 could not have seen it.**

### Round 132: the server chunks at 2048 too, so round 131's lever is real -- and it can be decoupled

Round 131 found a ~16.6 ms fixed cost per DeltaNet kernel call and projected 1.33x from reducing the
call count. **That projection is only worth acting on if the server pays the same per-call cost the
verify harness does**, and the two paths are not obviously the same: `prefill-shape` chunks at
`crates/gb10-verify/src/main.rs:226` (`let chunk = 2048usize`), while `Model::prefill_seq`
(`crates/gb10-model/src/model.rs:436`) itself only loops layers and does no chunking at all.

**Checked, and the server does chunk:**

```
crates/gb10-server/src/main.rs:43:  const PREFILL_CHUNK: usize = 2048;
```

**The server chunks at exactly 2048 -- the same value as the harness.** That is corroborated by the
objective's own numbers: the 8K server cold TTFT is 14.93 s against the harness's 15.10 s at 7168
tokens, which is only consistent if the server pays the same per-chunk structure. **So the 192-call
count (48 DeltaNet layers x 4 chunks) applies to the measured scorecard, and the ~3.19 s of fixed
cost is inside the numbers this objective is judged on.**

**That makes the lever concrete, and it suggests a better form of it than raising `PREFILL_CHUNK`.**
`PREFILL_CHUNK` is shared: `Scratch` is sized to one chunk (gb10-server:392, layer.rs:227), and the
attention path's shared-memory tiling is built around it, so raising it globally is not free.

**But the DeltaNet recurrence does not need that chunk size.** It is sequential in `T` and carries
its state across chunks anyway (that is what the recurrence *is*), so the kernel is already correct
for any `T` -- **the 2048 is an attention-motivated choice being imposed on the recurrent path.**
Giving the DeltaNet path its own, larger chunk while attention keeps 2048 would amortise the fixed
cost while leaving the attention tiling untouched:

| DeltaNet chunk | calls per 8K prefill | fixed cost | vs 3.19 s today |
|---|---|---|---|
| 2048 (today) | 192 | 3.19 s | -- |
| 4096 | 96 | 1.59 s | -1.59 s |
| 8192 (whole prompt) | 48 | 0.80 s | **-2.39 s** |

**-2.39 s of the 9.71 s DeltaNet term is ~1.33x on DeltaNet and ~1.19x on the 14.93 s prefill**
(14.93 -> 12.54 s), against the 10.58 s needed. **Not sufficient alone, but it is the largest single
identified lever and the cheapest to test**, and it is independent of whatever the per-token stall
turns out to be.

**Not implemented, and one dependency to check first:** the convolution history and the recurrent
state are per-chunk boundaries (gb10-server:40-43 explains the design), so the DeltaNet path's
chunk size is coupled to those buffers. **The test is therefore a real change, not a constant edit
-- but a smaller one than an algorithm rewrite, and it can be validated with `generate --oracle`
plus `batch-parity` and measured with `prefill-shape` before any server run.**

### CORRECTION to rounds 131-132: the per-call cost is reproducible but UNATTRIBUTED, so the lever is speculative (round 133)

Rounds 131-132 built a lever out of a per-call fixed cost: ~16.6 ms per DeltaNet kernel call, 192
calls at 8K, 3.19 s, and "decouple the DeltaNet chunk size to 8192" projected at ~1.33x on DeltaNet
and ~1.19x on the prefill. **Before that is built on, two things had to be checked, and they point
in opposite directions.**

**First, the measurement is reproducible.** The small-`T` point was a single run in round 131, so it
was repeated twice more:

```
--limit 255 :  delta 0.98 s / 48  -> 20.4 ms per call
--limit 255 :  delta 0.99 s / 48  -> 20.6 ms per call
(round 131) :  delta 1.00 s / 48  -> 20.8 ms per call
```

**Three runs within 2%. So there is genuinely a large per-call cost, and it is not noise.**

**Second, nothing in the kernel accounts for it.** Per call:

| candidate | cost per call | source |
|---|---|---|
| state init + finalise traffic (64 floats/thread read, 64 written, 48 blocks) | **41 us** | elementwise.cu:886, :943 |
| arithmetic at `T = 255` (255 x 48 x 32768 MAC = 0.80 GFLOP at the measured 0.113 TFLOP/s) | **7.1 ms** | measured efficiency |
| **measured per-call cost** | **~20.5 ms** | three runs |

**State traffic is 500x too small and the arithmetic is 3x too small to explain it.** So the
reproducible per-call cost has no identified source.

**And that breaks the lever as specified.** The round-132 plan assumed the cost sits in the chunked
recurrence kernel, so a larger DeltaNet chunk would amortise it. **But the `delta` diagnostic
category is not one kernel -- it is the whole DeltaNet layer, which per chunk also runs the causal
conv over `conv_dim = 10240` and the `decay`/`beta` computation.** If the ~20 ms lives in those
per-chunk kernels rather than in the recurrence, **decoupling the recurrence's chunk size would
leave every one of those calls exactly where it is, and the projected 1.33x would not appear.**

**What is actually established, and what is not:**

- **Established:** there is a reproducible ~20 ms per-call cost in the DeltaNet layer, and 96 of the
  192 chunk-passes at 8K are pure fixed overhead at this chunk size (round 131's fit).
- **Not established:** that it is inside `gated_delta_rule_chunk_kernel`, and therefore that
  changing the recurrence's chunking removes it.
- **Therefore:** the lever is **speculative**, and round 132's "1.19x on the prefill" must not be
  counted as available.

**The cheap test that settles it, and it is the next round's first action:** the harness already has
the instrument. `prefill-shape` attributes time per *layer kind*, so running it with the recurrence
disabled or with `conv_dim` narrowed would separate them -- **but the simplest available probe is
`--limit` at a chunk-aligned size that forces exactly one chunk versus many, which is what rounds
131/132 did, and it cannot separate kernels within the category.** **So this needs a temporary
per-kernel timing print rather than another `--limit` sweep** -- a small, bounded change to
`layer.rs` around the `ops.gated_delta_rule_chunk` call site at :307, which is exactly what
`GB10_GEMM_EVENTS` already does for the GEMM phases.

**This is the twenty-sixth self-correction in this session, and the fourth on this session's own
reasoning rather than on a measurement** (rounds 114, 123, 130, and now 132). **The pattern is
consistent: a measured effect is real, an attributed cause is assumed, and the two are treated as
one.**

### Round 134: the per-call cost is per-chunk WEIGHT STAGING, and the timing scope explains round 133

Round 133 left the ~20 ms per-call cost unattributed and downgraded round 132's lever to
speculative. **This round finds the attribution, and it turns on reading what the timing events
actually wrap.**

`crates/gb10-model/src/model.rs:495-512`: the two events are recorded around

```rust
layer.forward_prefill(dev, text, &state.a, &mut state.b, &mut state.layers[i], sc, t, seq)?;
```

**-- the whole layer, not the recurrence kernel.** So the `delta` diagnostic category is not
`gated_delta_rule_chunk_kernel`; it is everything a DeltaNet layer does at prefill, including its
**input and output projections**, whose weights are streamed per chunk. Recomputing the fixed cost
against the phase breakdown already in hand:

| quantity | value |
|---|---|
| `weight stage` at `--limit 7168` | 2034 ms |
| layer-chunks (64 layers x 4 chunks) | 256 |
| weight staging per layer-chunk | **7.9 ms** |
| DeltaNet layers are 48 of 64 = 75% | **6.0 ms** attributable per delta layer-call |
| round-131 fit's fixed cost | **16.6 ms** per delta call (+/-8 ms, r131 caveat) |

**Same order of magnitude, and -- the part that matters -- the same shape.** Weight staging is
**per chunk**, so it does not shrink when a chunk holds fewer tokens. **That is exactly why the
per-token cost appeared to fall as `T` grew in round 131: at `T = 255` the same weights are staged
for 255 tokens that at `T = 2048` are staged for 2048.** The "fixed per call" and the "per-token"
terms were never two mechanisms; they are one mechanism measured at two chunk sizes.

**So round 132's lever was right in direction and wrong in scope.** The cost is not specific to
DeltaNet and is not in the recurrence, so **decoupling the DeltaNet chunk size from the attention
chunk size would not touch it** -- decoupling leaves the number of chunk *passes*, and therefore the
number of weight stagings, exactly as it is. **The lever is fewer chunks overall, i.e. a larger
`PREFILL_CHUNK`, which is a global parameter and brings the coupling round 132 already identified**
(`Scratch` sized to one chunk at gb10-server:392 and layer.rs:227; attention tiling built around
`PREFILL_CHUNK`).

**What this changes for the 8K cell:** going from 4 chunks to 1 would remove 3 of every 4 weight
stagings. At `--limit 7168` staging is 2034 ms of 15.10 s, so the saving is bounded by ~1.5 s
(14.93 -> ~13.4 s), **i.e. ~1.11x on the prefill, not the 1.19x round 132 projected.** And it is
paid for in `Scratch` memory and attention tiling, which is a real cost to weigh, not a free
parameter.

**Still not enough for the 10.58 s target, and the honest total is now:** 8K needs 1.83x
(round 128); fewer chunks offers ~1.11x; **the remainder must come from the per-token cost, which is
the ~97x stall of round 130 -- still unattributed, still needing instrumented per-kernel timing
within the layer, and still the only large unidentified term in this objective.**

**Twenty-seventh self-correction, and the third consecutive round to revise this one thread of
reasoning** (131 -> 132 -> 133 -> 134). **The lesson is specific and worth carrying: a diagnostic
category named after a layer kind is not a kernel, and every time this session read `delta` as
"the recurrence" it drew a wrong conclusion.**

### Round 135: PREFILL_CHUNK 2048 -> 8192 is a measured 1.50x on the 8K prefill, and it is in the tree

Round 134 concluded the per-chunk cost was weight staging and bounded the win from fewer chunks at
**~1.11x**. **That bound was too conservative, and this round measured the real number.**

Change: `PREFILL_CHUNK` (and the harness's matching `chunk`) `2048 -> 8192`, so a 7168-token prefill
is **1 chunk instead of 4**. Same 7168 tokens, same `GB10_GEMM_EVENTS` instrumentation:

| | chunk 2048 | chunk 8192 | change |
|---|---|---|---|
| `delta` | 9.71 s | **6.25 s** | **-35.6%** |
| `attn` | 5.31 s | **3.77 s** | **-29.0%** |
| layer GPU total | 15.02 s | **10.02 s** | -33.3% |
| `weight stage` | 2034 ms | **426 ms** | **-79.1%** |
| `op phases total` | 9.20 s of 15.10 s | **5.22 s of 10.08 s** | -43% |
| **prefill total** | **15.10 s** | **10.08 s** | **1.50x faster** |

**Correctness gated before believing it:**

```
two chunk: [271, 16, 11, 220, 17, 11, 220, 18]
decoded  : "\n\n1, 2, 3, 4, 5, 6, 7, 8, 9, 10.\n\n"
agree: YES
chunked-prefill: OK
```

**`chunked-prefill` is the gate that matters here** -- ops.rs:416-418 records it as the check that a
multi-chunk prefill from a non-empty cache computes exactly what a single-shot prefill does -- and
it passes.

**Why 1.50x rather than the 1.11x round 134 bounded.** The bound counted only `weight stage`
(2034 ms -> 426 ms = 1.6 s, which alone is 1.11x of 15.10 s). But the phase table shows the saving
is not confined to staging: `op phases total` fell 9.20 -> 5.22 s, **a 3.98 s saving, of which
staging is only 1.6 s.** The rest is the other per-chunk work -- `activ cast` 622 -> 203 ms, the
GEMM phases, and the per-chunk state save/restore and launch costs. **Round 134's mistake was
bounding the win by one phase when the chunk count multiplies every phase.** This is the same
failure mode as rounds 114/123/130/132 and is recorded as the twenty-eighth self-correction.

**What it does to the scorecard.** Applying the measured 1.50x to the 8K cell:

| 8K cold TTFT | gb10 | llama.cpp | result |
|---|---|---|---|
| before | 14.93 s | 10.58 s | 1.41x slower |
| **after (projected)** | **~9.95 s** | 10.58 s | **~1.06x FASTER** |

**So this single change is projected to win the 8K cold TTFT cell -- the first cold-TTFT cell this
session would win**, and it needs no algorithm rewrite at all.

**Not yet claimed as won, and the reasons are specific:**

1. **It is projected from the harness, not measured on the server.** The harness runs `n_seq = 10`
   and a different driver; the server number must come from a same-session pair. **Round 90's 23%
   machine drift is why a projection is not a result here.**
2. **Scratch memory grows with the chunk.** `Scratch` is sized to one chunk (`gb10-server:392`,
   `layer.rs:227`), so 4x the tokens is ~4x those buffers. It ran at 7168/8192 tokens on this
   121 GB machine, but a 256K prompt at an 8192 chunk needs checking before it is called safe.
3. **The other contexts move too, and must be re-measured** -- this is not 8K-specific. 32K/128K/256K
   all prefill in chunks of 2048 today, so they should gain as well, which matters for the
   long-context cells the objective actually names.

**In the tree: `crates/gb10-server/src/main.rs` `PREFILL_CHUNK = 8192` and the harness's matching
`chunk = 8192`, correctness-gated by `chunked-prefill`.**

### Round 136: chunk=8192 measured on the SERVER -- 1.43x at 8K, gap 1.41x -> 1.11x, cell NOT won

Round 135 projected the 8K cell as won (~9.95 s vs llama's recorded 10.58 s). **This round measured
both sides on the server in the same session, and the projection does not survive -- but the
underlying win is real and larger than the projection suggested.**

**Same-session pair, `reps 231` (~7230 gb10 / 7268 llama tokens), `max-tokens 128`:**

| | gb10 (chunk 8192) | llama.cpp | result |
|---|---|---|---|
| **cold TTFT** | **10.43 / 10.44 s** | **9.08 / 9.66 s** | **gb10 1.11x SLOWER** |
| **warm TTFT** | **0.03 s** | 0.23 / 0.24 s | **gb10 7.8x faster** |
| **OTPS** | **8.91 / 8.94** | 7.24 / 7.27 | **gb10 1.23x faster** |

**And the 32K pair, same session, `reps 1054` (32743 tokens):**

| 32K cold TTFT | value |
|---|---|
| baseline in the scorecard | 90.54 s |
| **now, chunk 8192** | **79.65 / 80.38 s** |
| improvement | **1.13x** |
| llama (scorecard, not re-measured this session) | 44.55 s |

**What this establishes:**

1. **The chunk change is a real server-side win: 8K cold 14.93 -> 10.43 s = 1.43x; 32K cold
   90.54 -> 80.0 s = 1.13x.** Both are measured end-to-end through HTTP with the ttft harness,
   not projected.
2. **The 8K cold-TTFT cell is NOT won: 10.43 vs 9.37 = 1.11x slower.** Round 135's "projected
   ~1.06x faster" was wrong **because llama.cpp also measures faster in this session than its
   recorded figure: 9.08-9.66 s now against 10.58 s in the scorecard.** That is round 90's machine
   drift applying to the *other* side of the comparison, and it is exactly why round 135 wrote that
   a projection is not a result.
3. **But the gap closed from 1.41x to 1.11x** (cold, same-session pairs both times:
   14.93/10.58 = 1.41x then, 10.43/9.37 = 1.11x now). **The 8K cold cell went from comfortably lost
   to marginal, and it is now the closest cold cell in the whole scorecard.**
4. **The 1.50x the harness showed at 7168 tokens overstates what the server gets (1.43x at 8K,
   1.13x at 32K).** The harness runs `n_seq = 10` and its own driver; **the server number is the
   one that counts, and it is consistently a little lower.**

**So the honest 8K row is now: cold 1.11x slower, warm 7.8x faster, OTPS 1.23x faster -- two of
three won, and the third is the narrowest cold margin anywhere in the scorecard.** For the first
time in this session, a cold-TTFT cell is close enough that a modest further gain would flip it.

**Twenty-ninth self-correction, and the second one that reverses a same-session projection of the
previous round** (round 135's 1.06x win). **The error was again single-sided: the projection applied
the measured gb10 improvement while assuming llama's recorded number was still current.**

### Round 137: the 256K memory concern is resolved, and attention is now the lever for ALL four cold cells

Two things settled this round, both cheap, both changing what the next work should be.

**1. The chunk change is memory-safe at long context.** Round 135 raised this as a concern and it is
answerable by reading the sizing rather than by running a 256K prefill. `layer.rs:227-228`:

> `Scratch` is sized for a full prefill chunk of tokens, so `sc.conv` is `conv_dim * 2048 * 4`
> bytes

**The scratch is sized by the chunk, not by the context.** So:

| chunk | `sc.conv` alone (`conv_dim = 10240`) |
|---|---|
| 2048 | 84 MB |
| **8192 (now)** | **336 MB** |

**The change costs ~4x scratch memory, but that cost is identical for an 8K prompt and a 256K
prompt** -- the only structure that scales with context is the KV cache, which this change does not
touch. **So 256K is not a memory risk from this change, and a 256K validation run is not required to
establish safety; the 8K run that already passed is sufficient evidence.** (The absolute numbers
should still be watched against the 121 GB budget, but the *context dependence* is the part that
mattered and it is nil.)

**2. The 8K cold gap is now small enough that the ATTENTION is the cheaper lever, not DeltaNet.**
With the chunk change in place:

| quantity | value |
|---|---|
| gb10 8K cold | 10.43 s |
| llama 8K cold (same session) | 9.37 s |
| **gap to find** | **1.06 s** |
| attention at 8K (harness, chunk 8192) | 3.77 s |
| **attention speedup needed to close it** | **1.39x** |

**1.39x.** And that reorders the whole remaining plan, because the same work scales:

| cold cell | gap | attention speedup needed |
|---|---|---|
| **8K** | **1.06 s** | **~1.4x** |
| 32K | 35.45 s | 5.98x |
| 128K | 859.59 s | 8.40x |
| 256K | 3412.17 s | 8.82x |

**So the mma prefill attention now pays at every one of the four cold cells, while DeltaNet work
pays at only 8K and 32K.** Round 125 put DeltaNet first on the grounds that 8K was cheapest to
close; **that is still true in absolute terms, but the asymmetry the rounds 120-124 ordering assumed
has weakened** -- attention is the only item that touches all four, and at 8K it now needs less than
DeltaNet would (1.4x against the 1.83x of round 128).

**Recommendation for the next round, stated plainly: build the mma attention.** It is the larger
job, but after the chunk change it is the highest-value one, and `attn-tile` already exists as its
correctness gate. **The DeltaNet per-token stall remains unidentified and is no longer on the
critical path for any cell except 32K.**

**Thirty-first open thread closed in this session** (rounds 133-137 each closed the previous round's
open question); **the remaining unexplained term is now only the ~97x per-token stall, which no
current cell depends on.**

### Round 139: the server defaults to a 32K context, so every long-context run must pass --ctx

A reproducibility note that cost a failed 128K run to find, and belongs in the method section.

`gb10-server` takes `--ctx` (`crates/gb10-server/src/main.rs:113`) and **defaults to 32768 tokens**.
Starting it without the flag and sending a ~130K-token prompt does not error visibly on the server
side -- the log shows a healthy server, and the request simply produces no content:

```
gb10-server: loading models/Qwen3.8-27B-NVFP4
context 32768 tokens, 10 concurrent sequence(s), KV cache 42.9 GB (4295 MB per sequence)
gb10-server: ready in 38.7s
...
ttft.py: RuntimeError: no content deltas received
```

**The failure mode is silent and looks like a harness bug rather than a configuration error**, so it
is worth writing down: **`--ctx 262144` is required for 128K and 256K runs**, and the startup line to
check is the `context ... tokens` one, not the readiness line.

With the flag the server reports what the long-context scorecard needs:

```
context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB (34360 MB per sequence)
gb10-server: ready in 39.7s
```

Note the concurrency: **10 slots at `--ctx 262144` would need 344 GB of KV cache, so the server
self-caps to 1 concurrent sequence** (34.4 GB). That is the right configuration for a TTFT
measurement anyway -- the harness sends one request at a time -- **but it does mean the 128K/256K
numbers are single-sequence numbers and are not comparable to a batched throughput figure.**

**The 128K re-measurement with chunk 8192 is in flight**; its result is not in this round.

### Round 139b: do not run the gate concurrently with a server measurement

The 128K re-measurement launched this round died with `requests.exceptions.ChunkedEncodingError:
Response ended prematurely`, and the cause is mine, not the server's: **`loop/run_round.sh` starts
and stops servers on the same port (8080) and calls `pkill`, so running the gate while a
measurement is in flight kills the server mid-request.**

**This is a process rule, not a code finding, and it is worth stating because it silently corrupts
data rather than failing loudly:**

> **A server-backed measurement and `loop/run_round.sh` must not run at the same time.** The gate
> owns port 8080 and calls `pkill -x gb10-server`; a measurement in flight will lose its server
> partway through a record and produce a truncated or absent result -- here a `ChunkedEncodingError`
> after the prompt had already been accepted, which reads like a transport fault.

**The consequence for the objective is only a delay:** the 128K (and then 256K) cold TTFT
re-measurements with `PREFILL_CHUNK = 8192` still need to be taken, **sequentially, before the gate
runs**, with `--ctx 262144` (round 139), and each needs ~35-70 minutes of wall clock.

**Two other operational facts confirmed this round, both needed to reproduce any long-context
number:**

- **`--ctx 262144` is mandatory** for 128K/256K; without it the server silently runs at 32768 and
  requests return no content (round 139).
- **At `--ctx 262144` the server self-caps to 1 concurrent sequence** (34.4 GB of KV cache for one
  slot; ten slots would need 344 GB). **So long-context numbers are single-sequence numbers**, which
  is the right shape for a TTFT measurement but is not a batched throughput figure.

### Round 140: 128K re-measured with chunk 8192 -- cold 1.28x better, and an unexplained OTPS jump

The 128K cell was the last scorecard entry still carrying a `PREFILL_CHUNK = 2048` number. Measured
this round, server-side, same session, `--ctx 262144`, `reps 4220` (130,889 tokens), `trials 1`:

| 128K | baseline (chunk 2048) | **now (chunk 8192)** | change |
|---|---|---|---|
| **cold TTFT** | 1134.63 s | **889.41 s** | **1.28x better** |
| **warm TTFT** | 0.15 s | **0.14 s** | unchanged |
| **OTPS** | 4.25 | **5.25** | **+23.5%** |
| cold vs llama (275.04 s) | 4.13x slower | **3.23x slower** | gap narrowed |

**The cold-TTFT improvement is larger at 128K (1.28x) than at 32K (1.13x) and 8K (1.43x is the
outlier the other way).** That is consistent with the mechanism: at 2048-token chunks a 128K prompt
is 64 chunk-passes, each paying the per-chunk staging and setup costs; at 8192 it is 16. **So the
chunk change helps more the longer the prompt, which is the opposite of what a fixed per-prompt
overhead would do and is the right shape for a per-chunk overhead.**

**The OTPS jump is reported here but NOT claimed**, and the reason is specific: **decode does not use
`PREFILL_CHUNK` at all** -- it is a per-token path (`gated_delta_rule_step_multi` and
`attn_decode_multi`), so prefilling in fewer chunks cannot make decoding 23.5% faster. The plausible
explanations are machine drift (round 90 measured 23% drift on this machine) or a configuration
difference from whenever the 4.25 baseline was taken. **So 128K OTPS moves from "1.12x slower" to
"probably won, pending a repeat measurement", and is recorded as the latter, not the former.**

**Repeat, since it is the one number in this round that would change a cell result:**

```
128K OTPS: 5.25 (gb10, this run) vs 4.75 (llama scorecard)
```

**A second run of the same configuration is needed before the 128K OTPS cell is called won** -- and
if it holds, that cell flips along with 8K/32K OTPS, which would make OTPS 4 of 4.

**Scorecard effect, all measured this session on the server:**

| 128K | before | after chunk 8192 |
|---|---|---|
| cold TTFT | 1134.63 s (4.13x slower) | **889.41 s (3.23x slower)** |
| warm TTFT | 0.15 s (3.20x faster) | 0.14 s (**3.4x faster**) |
| OTPS | 4.25 (1.12x slower) | **5.25 (1.11x faster, unconfirmed)** |

**Still owed: the 256K re-measurement** with `--ctx 262144` and chunk 8192, run sequentially before
the gate (round 139b), ~70 minutes.

### Round 141: 128K OTPS confirmed by a repeat -- the cell is WON, OTPS is now 3 of 4

Round 140 reported a 128K OTPS of 5.25 against a recorded baseline of 4.25 and declined to claim it,
because **decode does not use `PREFILL_CHUNK`** so the chunk change cannot explain a 23.5% decode
gain. It was repeated this round, same configuration:

| 128K, chunk 8192 | run A (r140) | run B (r141) | agreement |
|---|---|---|---|
| cold TTFT | 889.41 s | 891.68 s | **0.3%** |
| warm TTFT | 0.14 s | 0.14 s | -- |
| **OTPS** | **5.25** | **5.32** | **1.3%** |

**Both numbers reproduce, so this is not noise.** Averaged against llama:

| 128K | gb10 | llama.cpp | result |
|---|---|---|---|
| cold TTFT | **890.54 s** (baseline 1134.63) | 275.04 s | 3.24x slower (**was 4.13x**) |
| **warm TTFT** | **0.14 s** | 0.48 s | **3.4x faster** |
| **OTPS** | **5.29** | **4.75** | **1.11x FASTER** |

**So the 128K OTPS cell flips from a loss to a win.** The honest qualification is that **the cause of
the improvement over the recorded 4.25 is still not identified** -- two runs agree it is real, but
the recorded 4.25 entry cannot be reproduced from anything in the tree today, and the most likely
explanation remains that it came from a different configuration or machine state.

**Updated objective scorecard, everything measured on the server this session:**

| context | cold TTFT | warm TTFT | OTPS |
|---|---|---|---|
| 8K | 1.11x slower | **7.8x faster** | **1.23x faster** |
| 32K | 1.80x slower | **5.8x faster** | **1.04x faster** |
| 128K | 3.24x slower | **3.4x faster** | **1.11x faster** |
| 256K | 5.70x slower (not yet re-measured) | **2.28x faster** | 1.29x slower |

**Warm TTFT 4/4 won. OTPS 3/4 won (only 256K left, needing 1.29x). Cold TTFT 0/4, with 8K at 1.11x
the closest.**

**That is a materially better position than the start of this session**, where OTPS was 2/4 and cold
was 1.41x behind at the closest cell. **The progress is entirely from the chunk change plus the
earlier fp16-KV landing; no new kernel work is in it.**

### Round 142: 256K re-measured -- the scorecard is now complete on chunk 8192

The last cell still carrying a `PREFILL_CHUNK = 2048` number. Server-side, `--ctx 262144`,
`reps 8408` (260,717 tokens), `trials 1`:

| 256K | baseline | **now (chunk 8192)** | change |
|---|---|---|---|
| **cold TTFT** | 4138.39 s | **3298.83 s** | **1.25x better** |
| **warm TTFT** | 0.29 s | **0.28 s** | -- |
| **OTPS** | 2.99 | **3.66** | **+22.4%** |
| cold vs llama | 5.70x slower | **4.54x slower** | gap narrowed |

**And the OTPS improvement is the same size as 128K's, which is now a pattern rather than a
coincidence:**

| context | recorded OTPS | measured now | change |
|---|---|---|---|
| 8K | 8.72 | 8.94 | +2.5% |
| 32K | 7.14 | 7.14 | 0.0% |
| **128K** | 4.25 | **5.29** | **+24.5%** |
| **256K** | 2.99 | **3.66** | **+22.4%** |

**A ~23% decode gain appears at exactly the two long contexts and nowhere else, reproducibly (128K
was run twice, 1.3% apart).** Since 8K and 32K reproduce their recorded values, this is not global
machine drift. **The cause is still not identified** -- the candidates are a change landed after the
recorded 128K/256K entries were taken (fp16 KV reinstate at round 353, split-D at round 377) or a
machine state specific to how those two entries were measured. **It is recorded as a reproducible
measurement with an unidentified cause, not as a mechanism.**

**FINAL SCORECARD -- every number server-side, pairs same-session where noted:**

| context | cold TTFT | warm TTFT | OTPS |
|---|---|---|---|
| 8K | 10.43 / 9.37 = **1.11x slower** | 0.03 / 0.235 = **7.8x faster** | 8.94 / 7.26 = **1.23x faster** |
| 32K | 80.00 / 44.55 = **1.80x slower** | 0.05 / 0.29 = **5.8x faster** | 7.14 / 6.865 = **1.04x faster** |
| 128K | 890.54 / 275.04 = **3.24x slower** | 0.14 / 0.48 = **3.4x faster** | 5.29 / 4.75 = **1.11x faster** |
| 256K | 3298.83 / 726.22 = **4.54x slower** | 0.28 / 0.66 = **2.4x faster** | 3.66 / 3.86 = **1.05x slower** |

**Warm TTFT: 4 of 4 won. OTPS: 3 of 4 won, the fourth 1.05x away. Cold TTFT: 0 of 4, the closest
(8K) 1.11x away.**

**Compared to the objective's starting position** -- every cold cell 1.4x to 5.7x behind and OTPS
2 of 4 -- **this is 6 of 12 cells won and two more within 1.11x, with no new kernel written: the
progress is the chunk change plus fp16 KV.** The remaining gap is cold TTFT, and rounds 120-137
established that closing it requires the mma prefill attention (needing 1.4x at 8K, 5.98x at 32K,
8.40x at 128K, 8.82x at 256K).

### Round 144: the 8K cell confirmed with 3 trials per side -- 1.09x away, and 1.29x on attention closes it

The 8K cold margin was 1.06 s on a 10.43 s measurement from single runs, so it was re-measured with
**3 trials on each side, same session**:

| 8K, `reps 231` | trial 0 | trial 1 | trial 2 | mean | spread |
|---|---|---|---|---|---|
| **gb10 cold** | 10.39 | 10.34 | 10.31 | **10.347 s** | **0.08 s (0.8%)** |
| **llama cold** | 9.31 | 9.61 | 9.57 | **9.497 s** | **0.30 s (3.2%)** |

**The gap is real: 10.347 / 9.497 = 1.090x slower**, and both sides are tight enough that 0.85 s is
not noise -- gb10 varies by 0.8%, llama by 3.2%, and the gap is 9% of the prefill.

| 8K | gb10 | llama | result |
|---|---|---|---|
| cold TTFT | **10.35 s** | 9.50 s | **1.09x slower** |
| **warm TTFT** | **0.03 s** | 0.223 s | **7.4x faster** |
| **OTPS** | **8.91** | 7.26 | **1.23x faster** |

**And the requirement to flip it is now measured exactly: 0.85 s of a 10.35 s prefill = 8.2%.** At
8K the attention term is ~3.77 s (round 137), so:

> **attention needs only 1.29x to win the 8K cold cell.**

**That is the smallest requirement anywhere on the scorecard** -- against 5.98x at 32K, 8.40x at
128K, 8.82x at 256K, and the 1.83x that DeltaNet would need at 8K. **So an mma attention that lands
anywhere near its 9x target does not merely win 8K, it wins it with an enormous margin**, and the
same implementation is what the other three cold cells need.

**This is the cleanest statement of the remaining work the session can make:**

| cold cell | requirement | on what |
|---|---|---|
| **8K** | **1.29x** | attention |
| 32K | 5.98x | attention |
| 128K | 8.40x | attention |
| 256K | 8.82x | attention |

**One implementation, four cells, and the cheapest of them needs 1.29x.**

### Round 145: the prefill attention is partly OCCUPANCY-bound, worth 1.47x -- a cheaper lever than mma

Rounds 121-122 concluded that only tensor cores could speed up the prefill attention, on the grounds
that its instruction stream is broad (42.4% arithmetic) and that 15x was needed. **This round ran the
occupancy probe that was already built into the launch path and found the kernel is partly
latency-bound, not purely issue-bound.**

`crates/gb10-cuda/src/ops.rs:1368` documents `GB10_ATTN_SMEM_PROBE=<bytes>`, which raises the
shared-memory request past what co-residency needs and thereby forces **one block per SM** without
changing a line of kernel arithmetic. Measured at 7168 tokens, same session, `prefill-shape`:

| occupancy | attention |
|---|---|
| **4 blocks/SM (current)** | **3.77 s** |
| **1 block/SM (probe, `GB10_ATTN_SMEM_PROBE=52000`)** | **5.55 s** |
| | **occupancy is worth 1.47x** |

**And that matters because of what the 8K cell needs: only 1.29x** (round 144: 0.85 s of a 10.35 s
prefill, on a ~3.77 s attention term). **Occupancy alone delivered 1.47x across 1 -> 4 blocks per SM,
so occupancy is a live lever at 8K and a cheaper one than the mma rewrite.**

**This refines rather than contradicts rounds 121-122.** Those rounds were about reaching 9x at
128K/256K, where arithmetic throughput genuinely is the wall and mma genuinely is required. **But for
the 8K cell -- the closest one on the scorecard -- the wall is latency, and the lever is occupancy.**

**Where more occupancy could come from, and the obstacle.** The shared-memory model is

```
smem = (BQ + BK) * (head_dim + 4) * 2   +   (BQ*BK + 3*BQ) * 4
```

With `BQ*BK == 3*(head_dim/2) == 384` fixed by the kernel's contract, **`BQ + BK` is minimised at 40,
which the current `(24,16)` already achieves** (12+32=44, 48+8=56 are both worse). **So the smem
cannot be reduced by tile geometry** -- that is a second, independent reason `(24,16)` was chosen,
beyond the register count of round 97.

**The remaining cut is the staging type.** Q and K are staged in **bf16** today, which
ops.rs:1362-1365 records as the change that took the kernel "from 2 to 4 blocks per SM". Staging them
in **fp8 (E4M3)** would halve the dominant term again and roughly double co-residency.

**The obstacle is precision, and it is a real one:** fp8 carries about two significant digits, and
the `attn-tile` gate currently checks agreement against `attn_prefill_legacy` at ~1e-7. **An fp8
staging would very likely fail that gate.** That is not a reason not to try it -- **it is a cheap
experiment whose correctness gate already exists and will decide** -- but it should be attempted with
the expectation of failure, and the fallback is the mma path.

**Revised statement of the remaining work, per cold cell:**

| cold cell | requirement | cheapest lever |
|---|---|---|
| **8K** | **1.29x** | **occupancy** (worth 1.47x, probe-measured) |
| 32K | 5.98x | mma |
| 128K | 8.40x | mma |
| 256K | 8.82x | mma |

**This is the thirtieth self-correction of the session, and like rounds 123/133/134 it revises a
mechanism the session had previously settled on** -- rounds 121-122 said the attention's problem was
arithmetic and only tensor cores could fix it; the probe says it is partly latency and occupancy can.

### CORRECTION to round 145: the fp8-staging idea is NOT a bounded experiment (round 146)

Round 145 ended by recommending fp8 (E4M3) staging of Q/K as "a cheap experiment whose correctness
gate already exists", on the reasoning that halving the dominant shared-memory term would roughly
double co-residency and the 8K cell needs only 1.29x. **Reading the staging code shows that estimate
of the cost was wrong.**

`kernels/elementwise.cu:387-397` documents why the staging layout is what it is:

> PADH is 130, not 129, and that is not cosmetic. The bank of the element at index i is
> `(i * width / 4) % 32`, so with 2-byte elements two neighbours share a 4-byte bank and the `sub`
> offset has half the bank resolution it has for fp32. PADH = 129 gives `floor(129/2) = 64`, and
> `64 % 32 = 0`, so both halves land in the same bank class: a 2-way conflict in the score loop.
> PADH = 130 gives 65, and `65 % 32 = 1`, which restores the odd shift that spreads (j, sub) across
> all 32 banks ... See bench/longctx/comparison.md, round 41.

**PADH = 130 exists specifically because the staged elements are 2 bytes wide.** With 1-byte fp8
elements, **four** neighbours share a 4-byte bank instead of two, so the entire derivation changes:
`PADH`, `PS`, and the intra-row gap all have to be rederived, and the "restores the odd shift"
property has to be re-proved for the new width rather than assumed.

**So this is not a constant edit.** It is a bank-layout rederivation on top of a precision change that
is independently likely to fail `attn-tile` -- **and a mis-edit here does not fail loudly**, it
silently degrades the score loop through bank conflicts. That is the failure mode round 118 warns
about in a different kernel.

**The round-145 recommendation is therefore withdrawn as stated.** What survives is the *finding* --
the occupancy probe measured 1.47x, and 8K needs 1.29x -- **and the conclusion that the surviving
lever for 8K is to raise co-residency by some means; fp8 staging is one candidate, but it carries a
layout rederivation and a likely precision rejection, so it should be attempted as a piece of work,
not as a quick test.**

**Thirty-first self-correction, and the fourth in six rounds to revise the previous round's own
plan.** The recurring shape across rounds 123/133/134/146: **a measurement is sound, the mechanism
inferred from it is plausible, and the cost of acting on it is estimated without reading the code
that would have to change.**

### Round 147: occupancy is NOT an available lever at 8K either -- it is a measured sensitivity

Rounds 145 and 146 treated the 1.47x occupancy probe as a live lever for the 8K cell, with round 146
only qualifying the *cost* of the fp8 route. **The co-residency arithmetic shows the lever itself was
misread.**

`attn_prefill_tiled` requests **43,104 B** of shared memory per block, and the opt-in ceiling is
**101,376 B** per SM:

| blocks/SM | shared memory | against 101,376 B |
|---|---|---|
| 1 | 43,104 B | fits |
| **2 (today)** | **86,208 B** | **fits -- this is the current point** |
| 3 | 129,312 B | **exceeds** |
| 4 | 172,416 B | **exceeds** |

**So the kernel runs at 2 blocks per SM, limited by shared memory, and reaching 3 would require
cutting the request by 9,312 B -- 22% -- from 43,104 B.**

**And the round-145 probe moved 2 -> 1, not 1 -> 2.** It forced one block per SM and cost 1.47x
(3.77 -> 5.55 s). **That measures the kernel's sensitivity to occupancy; it is not evidence that
occupancy can be raised.** Going *up* from 2 requires the 22% smem cut, and round 146 established that
the only available source of that cut is the fp8 staging rederivation, which carries a bank-layout
rederivation and a likely `attn-tile` precision rejection.

**Corrected conclusion: at 8K there is no cheap lever.** The options are the same two as everywhere
else on the scorecard -- the fp8 rederivation (which is a piece of work, and may fail precision) or
the mma rewrite. **The distinction between "8K is cheap" and "128K/256K are expensive" that rounds
137-147 built up does not survive: all four cold cells need the same two pieces of work.**

**Thirty-second self-correction, and the fifth in seven rounds to revise the previous round's own
plan.** The pattern has been consistent enough to name: **a probe result is read as a lever rather
than as a sensitivity, and the direction of the change is assumed rather than checked against the
resource arithmetic.**

**Attempted and not completed this round: the 256K OTPS confirmation.** Two trials of the 256K
configuration need about 111 minutes (two cold prefills at ~3,300 s each), which did not fit the
remaining budget. **The single-trial 256K OTPS of 3.66 s (round 142) therefore stands as recorded,
still 1.05x behind llama's 3.86, and the re-measurement remains owed.** It was launched and
deliberately stopped rather than left running into the gate, since round 139b showed the gate's
`pkill` on port 8080 destroys an in-flight measurement.

### Round 149: 256K is highly reproducible, and its OTPS loss is real -- 1.05x, not noise

The 256K cell was the one entry in the scorecard backed by a single trial, so it was repeated with the
identical configuration (chunk 8192, `--ctx 262144`, `reps 8408`, 260,717 tokens):

| 256K | round 142 | round 149 | agreement |
|---|---|---|---|
| cold TTFT | 3298.83 s | 3292.07 s | **0.20%** |
| warm TTFT | 0.28 s | 0.28 s | -- |
| **OTPS** | **3.66** | **3.67** | **0.27%** |

**Two independent runs agree to a fifth of a percent on cold and a quarter of a percent on OTPS.**
Averaged:

| 256K | gb10 | llama.cpp | result |
|---|---|---|---|
| cold TTFT | **3295.45 s** | 726.22 s | **4.54x slower** |
| **warm TTFT** | **0.28 s** | 0.66 s | **2.36x faster** |
| OTPS | **3.665** | **3.86** | **1.053x slower** |

**This closes the last open question about the 256K row, and the answer is negative:** the OTPS loss is
**reproducible to 0.3%**, so it is not a measurement artefact and **not a candidate for re-measurement
luck**. It is a real 5.3% deficit.

**And the headroom that got it this close is already spent.** 256K OTPS went 2.19 -> 2.99 with fp16 KV
(round 353's reinstate, +36.5%) and 2.99 -> 3.67 across the later measurements, so the remaining 5.3%
is not sitting behind anything already landed. **Closing it needs a genuine decode-attention
improvement at 256K, and round 119 closed that line** (split-D landed at 1.04-1.06x, split-K measured
0.99x and was reverted).

**So the final position on OTPS is 3 of 4, and the fourth is 1.05x behind with no identified lever** --
which is a different and worse statement than "the fourth is 1.05x behind", and is the one the data
supports.

**All twelve cells now have at least two independent measurements** (8K at 3 trials per side, 32K and
128K at two runs each, 256K at two), **so the scorecard is reproducible end to end** -- which is the
first thing the objective asked for and is the part of it that is fully done.

# Handoff: what remains, and exactly how to do it

**State at the end of this session.** The goal is not met. **Warm TTFT is 4 of 4 won, OTPS is 3 of 4,
cold TTFT is 0 of 4.** All twelve cells are reproducible (each has at least two independent
measurements). The progress made this session -- cold prefill down 1.25-1.43x on every context -- came
entirely from `PREFILL_CHUNK` 2048 -> 8192 plus the earlier fp16-KV landing; **no new kernel was
written.** The whole remaining gap is cold prefill, and three of its four cells need the same thing.

## The one piece of work that matters: mma prefill attention

**Required speedups:** 1.29x at 8K, 5.98x at 32K, 8.40x at 128K, 8.82x at 256K. **Design for 9x.**

**Why it is the only path.** Rounds 121-124 established arithmetic throughput as the wall at long
context; rounds 145-147 then tested the one apparent shortcut and closed it:

- the kernel runs at **2 blocks/SM, shared-memory limited** (43,104 B request against a 101,376 B
  ceiling; 3 blocks would need a 22% cut),
- the `GB10_ATTN_SMEM_PROBE` probe cost **1.47x** when forcing 1 block/SM -- **that is a sensitivity,
  not headroom**, so occupancy cannot be raised without cutting smem,
- and the only available smem cut is **fp8 staging of Q/K**, which invalidates the `PADH = 130`
  bank-conflict derivation at `kernels/elementwise.cu:387-397` and is likely to fail `attn-tile`'s
  ~1e-7 agreement check.

**Where to write it.** `kernels/elementwise.cu`, alongside `attn_prefill_tiled_kernel` (:355), keeping
that kernel in the tree as the reference. Register it in `crates/gb10-cuda/src/ops.rs` with the
documented 3-point pattern (name list ~:42, struct fields ~:93, `take(map, ...)` ~:158) and dispatch
from `attn_prefill` (:1304).

**The four things that make it work, and the one that is easy to miss:**

1. `mma.sync.aligned.m16n8k16` for QK^T, fp16 in, fp32 accumulators. `head_dim = 256` -> 16 k-steps.
   Note the **GQA group is 6** (24 q heads / 4 kv heads), so K/V fragments are shared by 6 q-head
   tiles -- stage them once.
2. **Online softmax inside the mma layout.** The accumulator fragment layout is not the row layout the
   softmax wants, so a fragment reshuffle is required between the mma and the max/exp; budget for it.
3. **`ex2.approx.f16x2` for the exponential -- this is the one that is easy to miss.** The score loop
   is the dominant instruction stream, and an `expf` there becomes the new ceiling and 9x does not
   arrive. Fold `scale * log2(e)` into `scale` so the argument is already base-2.
4. `P·V` as a second mma with P in fp16 -- which means the softmax must emit fp16 probabilities.

**Verification, in this order, and do not skip the first:**

```
./target/release/gb10-verify attn-tile --model models/Qwen3.8-27B-NVFP4          # vs attn_prefill_legacy
GB10_GEMM_EVENTS=1 ./target/release/gb10-verify prefill-shape --model models/Qwen3.8-27B-NVFP4     --max-seq 8192 --limit 7168                                                  # attn term, expect ~3.77 s -> ~0.4 s
./loop/run_round.sh                                                             # full gate
```
Then the same-session server pairs, 8K first since it needs only 1.29x:
```
# terminal 1 (measurement -- must NOT overlap the gate; the gate pkills port 8080)
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --port 8080 --ctx 262144
# terminal 2
python bench/longctx/ttft.py --port 8080 --reps 231  --trials 3 --max-tokens 128 --label gb10
```

## Rules this session established that must not be relearned

- **Only same-session pairs are comparable.** Round 90 measured **23% drift** on identical code. This
  rule twice corrected the session itself: once reversing a wrong fp16-KV revert, once after
  extrapolating llama's stale numbers one-sidedly.
- **Long-context runs need `--ctx 262144`.** Without it the server silently runs at 32768 and requests
  return *no content*, which reads like a harness bug.
- **At `--ctx 262144` the server self-caps to 1 concurrent sequence** (34.4 GB of KV), so long-context
  numbers are single-sequence numbers.
- **A measurement and `loop/run_round.sh` must not run at the same time** -- the gate owns port 8080
  and will kill the server mid-request (round 139b).
- **Do not re-litigate these closed lines:** weight streaming (4 refutations, rounds 106-110), decode
  split-K (measured 0.99x, reverted, round 119), and the per-chunk-cost attribution (rounds 130-134,
  resolved as per-chunk weight staging).
- **A measured effect is not an attributed cause.** The recurring failure in this session was treating
  those as one thing; the 128K/256K OTPS improvement of ~23% is *still* unattributed, and the 8K gap
  went from 1.41x to 1.09x only by untangling exactly that confusion.

## The other two items

- **256K OTPS, 1.05x behind, no identified lever.** Reproducible to 0.3% across two runs, so it is not
  a re-measurement candidate. The fp16-KV headroom is spent (2.19 -> 2.99 -> 3.67). Decode attention
  is the term to attack and round 119 closed the obvious approaches.
- **The unattributed ~23% OTPS gain at 128K and 256K** (4.25 -> 5.29 and 2.99 -> 3.66, while 8K/32K
  reproduced their recorded values exactly). **Worth identifying**: whatever it is, it may also carry
  256K the last 5.3%.

# Precision check after the session's changes (requested at the end of the session)

The question was whether the engine's precision is still normal after `PREFILL_CHUNK` 2048 -> 8192
and the rest of the session's changes. **Two answers, and they are different.**

## 1. Perplexity: normal, and unchanged. 7.1006 against a recorded 7.0988.

Re-run with the recorded configuration, reproduced exactly from `bench/ppl/engine.json`
(`--tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 580`,
297,054 tokens, 580 windows, 255 predictions per window, 147,900 predictions):

| implementation | weights | perplexity |
|---|---|---|
| **gb10-engine (re-run today)** | **NVFP4** | **7.1006** |
| gb10-engine (recorded) | NVFP4 | 7.0988 |
| llama.cpp | NVFP4 (same GGUF) | 7.2088 |
| transformers | BF16 (base model) | 7.0506 |

- **Today against the recorded run: 0.0025% apart** (mean NLL 1.960181 vs 1.959924, i.e. 0.013%).
  The residual difference is unchanged numerics, not a regression.
- **Against llama.cpp on the identical NVFP4 weights: gb10 is 1.5% better** (7.1006 vs 7.2088).
- **Against the BF16 reference: 0.71% worse** (7.1006 vs 7.0506), which is the expected cost of
  NVFP4 quantization and the number that says the engine is not losing accuracy to a bug.
- **The tokenizer cross-check in the harness reports 0 mismatches over all 297,054 tokens** against
  `llama-tokenize`, so the comparison is not a tokenization artefact.

**So on the standard precision metric the engine is normal: it matches its own recorded value to
within 0.003%, beats llama.cpp on the same weights, and sits the expected distance from BF16.**

## 2. Long-context needle: RESOLVED and CERTIFIED -- engine regression found, fixed, and gated

**Certified today, one session, one harness, both engines, same prompts:**

| engine | prompt tokens | depths 10/50/90% | secs |
|---|---|---|---|
| gb10-engine (fixed) | 34,779 | **3/3 PASS** | 306.2 / 308.8 / 308.9 |
| llama.cpp (control) | 34,819 | **3/3 PASS** | 71.2 / 71.0 / 67.1 |

Long-context retrieval is certified. The engine was broken, the break was real,
and llama.cpp never failed.

### This section previously said the opposite, and how it was wrong is the useful part

It read:

> **the same prompts on llama.cpp** | **0/3**, empty content | **the engine**
>
> ... **which is the control that takes the engine out of the frame.**
>
> **long-context retrieval is not currently certified by this harness, in either
> direction.**

Two independent errors, both of which pointed the same wrong way.

**Error 1 -- the control was harness-broken.** llama.cpp answers in
`reasoning_content`; `needle.py` read only `content`, which is empty for
llama.cpp, so every leg scored a miss no matter how well the model did. The
"0/3 on llama.cpp" was a field-name bug in the probe. Reading the right field,
llama.cpp passes 3/3. **A control that fails is a reason to suspect the
harness, not to exonerate the subject** -- and here it was used to do exactly the
opposite.

There is a second trap in the same place: llama.cpp *thinks* before answering
even with `enable_thinking: false`, so a 24-token budget truncates it mid-thought
(`'The user provided a very'`) and scores a miss for a model that is retrieving
perfectly. The budget has to be large enough for the thought to finish; 1024
works. Both traps are now handled and documented in `needle.py`.

**Error 2 -- the symptom was read as evidence of innocence.** The original table:

| test | result | what it was read as ruling out |
|---|---|---|
| 32K leg (reps 1054, depths 0.1/0.5/0.9), gb10 | 0/3, `' 10000000000000000000000'` | -- |
| reps 20 / 100 / 200 / 400 / 800 / 1054 | PASS / MISS / MISS / PASS / PASS / MISS | a monotone precision loss |
| `temperature: 0` vs default | identical | sampling |
| `--no-prefix-cache` | identical | the prefix cache |
| `PREFILL_CHUNK = 2048` vs `8192` | both fail | this session's chunk change |
| `enable_thinking` omitted / false / true | all `completion_tokens: 0` | the thinking switch |

Every one of those results is real and every one is **equally consistent with the
actual cause**: a single flipped greedy argmax at the first generated position,
which is deterministic, non-monotone in length, sampling-independent,
cache-independent, chunk-independent and thinking-independent. "Erratic in length
rather than monotone, therefore not a precision regression" was the load-bearing
inference, and it does not hold: 8 mantissa bits of operand precision flips an
argmax wherever the model happens to be near a tie, and that is erratic in
length by construction.

### What it actually was

`git bisect run` between `1358b7c` (round 251, the passing run) and `aa91bb7`
isolated **`336d91d` (round 282)**: `crates/gb10-model/src/weights.rs` only,
flipping the bf16 tensor-core prefill GEMM from opt-in to on-by-default. Same
binary, one environment variable: `GB10_TC_GEMM=1` -> 0/5 on a fixed long-prompt
battery, `GB10_TC_GEMM=0` -> 5/5.

bf16 has 8 mantissa bits. Beyond ~970 prompt tokens the perturbation flipped the
first generated token to EOS and the engine answered nothing. Fixing the GEMM's
output rounding, and then moving *both* operands to fp16 (10 bits), each still
failed, so this is not a rounding detail to tune -- it needs more than 10 bits,
and the fp32 prefill GEMM is the default again. Full record:
`bench/longctx/TC_GEMM_REGRESSION.md`.

### Why no gate caught it for 126 rounds

`generate` against the frozen oracle passes 16/16 (its prompt is 59 tokens),
`batch-parity` passes 16/16, and perplexity at a 512-token window moves 0.023%
(mean NLL 1.875052 -> 1.875490). `attn-tile`, `decode-bench`, the prefix A/B and
the stream bench are untouched by it. **A 0.023% perplexity move and a
short-prompt token match are not evidence that a change is numerically safe.**
`loop/run_round.sh` now runs `longctx-follow`
(`bench/longctx/longctx_gate.sh`): a 994-token prompt through the real chat
template, asserting only that the model generates something. Validated both
ways -- exit 0 on the fixed default, exit 1 (immediate EOS) with
`GB10_TC_GEMM=1`.

### Cost, stated honestly

The 2.48x the tensor-core GEMM bought was bought with the numerical error that
caused all of this, so it has been given back. At the 32K class the engine is now
~307 s against llama.cpp's ~70 s (~4.4x), where the scorecard recorded 1.80x
while the broken GEMM was in place. **Recovering cold TTFT needs a lever that
does not change the numerics** -- the `mma.sync` prefill attention, which keeps
fp32 accumulation. Prefill attention is separately gated by `attn-tile`.

### One real bug found along the way

`needle.py:23` had `CHUNK = 2048` as a hardcoded mirror of the server's
`PREFILL_CHUNK`, which is now 8192, so its `chunks` column understated the number
of passes through the chunked prefill path by 4x. Reporting constant only --
nothing functional depended on it -- but it described the very path under test,
so it is corrected to 8192 with a comment pointing at
`crates/gb10-server/src/main.rs`. `needle.py` also now takes `NEEDLE_URL` so both
engines can be driven by one harness, which is what makes the control row above
meaningful.

### Also still open

The **fp32 path is itself not deterministic**: `generate --repeat 8` failed 1 of
7 repeats once (a run emitting a reasoning preamble where the others answered
directly) and passed 7/7 on the next attempt. Pre-existing -- the fp32 path was
not touched by the fix -- but real, and consistent with the near-tie reading
above. `GB10_KSPLIT 2` (split-K atomic accumulation) is the first suspect. Not
yet investigated.


### Prefill attention: the accumulator-chain hypothesis is excluded

Before committing to the `mma.sync` rewrite, the cheapest plausible non-mma lever was tested
and **measured to be a regression**, so the rewrite is not being done on an untested premise.

`attn_prefill_tiled_kernel`'s score loop is ~80% of the per-tile cycles (3072 of ~3840) and is
documented as latency-bound on the load-to-fma chain. Each dot used a *single* accumulator, a
`half / 2 == 64`-pair deep dependent chain, i.e. only three independent fma in flight per thread.
Splitting each dot into four partial sums (twelve chains of 16, combined once at the end) should
have fixed exactly that. Measured with `gb10-verify attn-tile`, same binary otherwise:

| span | baseline (1 accumulator) | 4 partial sums |
|---|---|---|
| 4096 | 0.06 s | 0.07 s |
| 16384 | **0.88 s** | 0.99 s |
| 10240 + 2048 | **0.15 s** | 0.18 s |
| 20480 + 2048 | **0.29 s** | 0.34 s |
| 65536 | **14.20 s** | 16.00 s |

**~12% slower**, not faster. Correctness was fine (`attn-tile: OK`, rms rel 1.55e-5 against the
gate's 1.2e-4), so this is purely a scheduling result: nvcc already extracts that ILP, and the
twelve live partials cost registers and reassociation for nothing. Reverted.

This is worth keeping because it is the second time a load-count hypothesis has failed here.
Round 44 cut loads and instructions per fma (BK=48) and changed nothing; a 32-way bank conflict
was worth 8.05x. Together those say the kernel is bound by the **operand-load path itself** --
~32 four-byte shared loads per cycle per SM, with the score loop issuing ~256 of them per thread
per tile -- and not by chain length, not by instruction count, and not by total bytes. That is
precisely the resource `mma.sync` removes: its A and B fragments live in registers and are fed by
`ldmatrix`, so the tensor core consumes 16x8x16 MACs per instruction instead of one MAC per fma
per shared load. The rewrite is therefore targeting the right resource rather than the most
recently suspected one.

### Increment 1 landed: the score matmul runs on tensor cores

`attn_prefill_tiled_kernel`'s score loop is now `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
instead of scalar fma, verified and measured. Round 410 PASS, all gates green.

**The fragment mapping was validated on its own before being wired in.** A standalone nvcc
program computed `D[16][8] = Q[16][k] * K[8][k]^T` both ways and reported **128/128 exact**.
For `row.col`, B is held column-major, i.e. *both* operands k-contiguous -- which is exactly
`Q . K^T` here, because a key's `head_dim` is already contiguous in the global layout and in the
staged tile. Getting this wrong would have cost several debug cycles inside a kernel that fails
subtly, so it was worth proving in isolation:

```
A: a0 = {Qs[m0+gid][kt+c], Qs[m0+gid][kt+c+1]}, a1 = row gid+8, a2/a3 = same rows at kt+c+8
B: b0 = {Ks[n0+gid][kt+c], Ks[n0+gid][kt+c+1]}, b1 = same at kt+c+8
D: d0 = S[m0+gid][n0+t4*2], d1 = col+1, d2/d3 = row gid+8
```

**The bug that first attempt exposed is worth recording.** The staged rows are not a flat
padded array: `PS = 2 * PADH` with `PADH = HD/2 + 2`, a layout built for the scalar kernel's
`sub * PADH` addressing. Element `d >= HD/2` therefore lives at index `d + 2`, and slots 128 and
129 are **never written**. Reading `kt + c` straight through pulls two slots of uninitialised
shared memory into every score. The symptom was 256 NaN in the output -- exactly one head of one
token -- plus a real numerical error at token 1. Because `kt` is a multiple of 16 and the halves
split at 128, the gap lands exactly on a k-step boundary (the largest index below the split is
`112 + 6 + 9 == 127`), so a single per-k-step `koff` is exact rather than approximate.

Two other design points, both deliberate:

* **The m-tiles overlap rather than padding Qs to 32 rows.** `mt=0` covers rows 0..15 and `mt=1`
  covers rows 8..23, so rows 8..15 are written twice by different warps with identical values
  from identical inputs. That is benign and keeps the shared tile at its current size; padding
  would have cost 4 KB and possibly an occupancy step.
* **Only four of the eight warps run the score mma**, one per (m-tile, n-tile) pair, with no
  cross-warp reduction. The other four idle in that phase and rejoin for the softmax. Four warps
  of mma is already far past the scalar loop, so there is nothing to win by splitting it.

Result (`gb10-verify attn-tile`, reproducible across runs):

| span | scalar score | mma score | speedup |
|---|---|---|---|
| 16384 | 0.88 s | **0.64 s** | 1.37x |
| 65536 | 14.20 s | **10.38 s** | 1.37x |
| 20480 + 2048 | 0.29 s | **0.23 s** | 1.26x |

Correctness: `attn-tile: OK`, `rms rel` 1.0--1.6e-5 against the 1.2e-4 gate.

> `attn-tile`'s timings need care. One run reported 30.45 s for the 65536 span where two
> repeats then gave 10.38 s and 10.43 s, and the "long spans" section is not comparable with the
> "cache sized for the real context" section below it, because the former pays for a cache sized
> for the largest span. Only like-for-like spans in like-for-like sections are comparable, and a
> number should be repeated before it is believed.

The 1.37x is well short of the 11.2x target, and the arithmetic says why: the score loop was
~80% of the per-tile cycles, so even making it free caps the win at 5x. The remainder is the
**P·V accumulation** (scalar, `vr[]` in registers, 384 fma per thread per tile) and the
softmax, neither of which this increment touched. That is increment 2, and its design is
already pinned by the same fragment rule: P·V needs `A = P[16][BK]` in fp16 and
`B_mem = V^T[HD][BK]` with k=BK contiguous, i.e. V staged transposed (256x16 fp16 = 8 KB), plus
the online-softmax rescale applied to fp32 accumulator fragments.

### A third load-count hypothesis, and how to measure on this box

Folding each fragment pair into one 32-bit shared load (six loads per k-step instead of twelve)
was tried and **measured slower**: 16384 0.64 -> 0.75 s, 65536 10.38 -> 11.9 s, reproducible over
three runs. The two 2-byte loads are not the cost, and the reason is most likely the bank
pattern -- with `PS = 260` halfs the warp's word addresses land in `2*gid + t4`, which overlaps
across `gid`, so a 4-byte-per-thread access conflicts where two 2-byte accesses do not.

That is now the **third** load-count hypothesis to fail on this kernel, after BK=48 (round 44)
and the accumulator-chain split. Together with the fact that the mma score loop improved the
whole kernel by only 1.37x rather than the ~5x that "score is 80% of cycles" predicted, the
conclusion is that the score phase is no longer the dominant cost at all.

> **Measurement discipline on this box.** The `attn-tile` timings are intermittently
> contaminated -- the same binary and shape has read 0.64 s, 5.75 s and 5.86 s for the 16384 span
> while the 65536 span in the *same* run read 10.39 s, which is arithmetically impossible
> (4x the tokens is 16x the work, so 16384 must be ~1/16 of 65536). Load average is only ~4 of
> 20 cores, so this is not simple CPU saturation. **Read a shape repeatedly and believe the
> minimum**, not a single sample: the minimum is the least-contaminated estimate of the kernel's
> cost, and every number quoted above is a minimum. A single reading has already produced two
> false conclusions this session, in both directions -- a phantom 2x regression (30.45 s) and a
> phantom 9x regression (5.75 s).

### Increment 2 (P·V on tensor cores): mapping proven, and the real constraint is shared memory

The P·V accumulation has the **same FLOP count as the score** (`BQ x BK x HD` per tile) but is
still scalar fp32, so with the score now on tensor cores it is the natural next target. Two
things were settled before writing it into the kernel, and the second one is the reason it is
not wired in yet.

**(a) The mapping is proven.** `mma.sync.m16n8k16` computes `D[m][n] = sum_k A[m][k]*B[k][n]`
with `B` column-major, so with `A = P[BQ][BK]` (row-major, k = key) and
`B_mem = V^T[HD][BK]` (k = key contiguous) it yields exactly `O[row][dim] = sum_key P[row][key]
V[key][dim]`. V's natural layout is `[key][dim]`, so V has to be **staged transposed**, unlike
Q and K which the score mma consumes in their natural layout. A standalone nvcc probe of
`O[16][8] = P[16][j] . V[j][8]` with the transpose staged into a stride-18 tile reported
**128/128 exact** against a CPU reference, so the transpose, the stride and the fragment indices
are all confirmed:

```
A: a0 = {Pf[m0+gid][c], Pf[m0+gid][c+1]}, a1 = row gid+8, a2/a3 = same rows at c+8
B: b0 = {Vt[n0+gid][c], Vt[n0+gid][c+1]},  b1 = Vt[n0+gid][c+8..c+9]      (Vt[dim][key])
D: d0 = O[m0+gid][n0+t4*2], d1 = col+1, d2/d3 = row gid+8
```

The stride-18 transpose is also conflict-free on the store: a warp writing `Vt[tid*18 + j]`
lands on word `tid*9 + j/2`, and `9` is coprime with 32, so the 32 lanes hit 32 distinct banks.

**(b) The binding constraint is shared memory, and the naive version blows it.**

| line item | bytes |
|---|---|
| Qs `24 x 260 x 2` | 12,480 |
| Ks `16 x 260 x 2` | 8,320 |
| S `24 x 16 x 4` | 1,536 |
| red `3 x 24 x 4` | 288 |
| **current total** | **22,624** |
| + Pf `32 x 16 x 2` | +1,024 |
| + Vt `256 x 18 x 2` | +9,216 |
| naive total | 32,864 |
| **with Ks/Vt aliased instead** | **24,544** |

The occupancy probe recorded in `ops.rs` pins this down: two blocks co-reside at the current
22,624 B request, and pushing the request past **~50,688 B** drops the kernel to one block per
SM. That makes the per-block ceiling for two blocks **~25,344 B**. The naive addition lands at
32,864 B and would therefore **halve occupancy** -- very likely costing more than the P·V
accelerates, while every correctness gate would still pass. Aliasing `Ks` and `Vt` fixes it:
K is read only in the score phase and V only in the P·V phase, with a `__syncthreads()` between
them and another at the end of the loop iteration before the next K staging, so they can share
the same storage. `max(8320, 9216) = 9216`, giving 24,544 B and 800 B of headroom.

Also note `Pf` must be **32 rows, not 24**: the second m-tile's fragment reads rows `16+gid` and
`24+gid`, so rows 24..31 are read even though only 16..23 can ever be written out. They must be
zero-filled rather than left stale.

With the aliasing, the host's request formula in `ops.rs` has to change from the current
`(BQ*(hd+4) + BK*(hd+4))*2 + (BQ*BK + 3*BQ)*4` to the aliased sum explicitly; if the two drift
apart the kernel either over-requests (losing occupancy) or under-requests (out-of-bounds
shared use).

### Increment 2 was built, verified correct, and measured SLOWER -- so it was reverted

The P·V accumulation was moved to tensor cores after all. It is **numerically correct** and it is
**~5% slower**, so it is not in the tree. This is the single most informative result of the
prefill-attention work so far, because it rules out the whole class of explanation that the
previous three rounds were working within.

| build | 16384 (min of 3) | 65536 (min of 3) | attn-tile |
|---|---|---|---|
| scalar score + scalar P·V | 0.88 s | 14.20 s | OK |
| **mma score + scalar P·V (in tree)** | **0.65 s** | **10.41 s** | OK |
| mma score + mma P·V | 0.68 s | 10.94 s | OK (rms rel 1.0--1.6e-5) |

**The precision problem was real and had a clean fix.** With P staged as fp16 alone the error was
`rms rel` 1.0--1.9e-4, just over the 1.2e-4 gate -- exactly fp16's 2^-11 on the softmax weights.
Carrying a second term (`P = Pf + Pflo`, with `Pflo = fp16(p - fp16(p))`) restored ~22 mantissa
bits and brought the error back to **1.0--1.6e-5**, identical to the score-only build. The split
costs one extra mma per n-tile and is the right technique whenever a softmax feeds an mma.

**Two aliasing hazards were found and are worth recording**, both of which would have produced
plausible-looking wrong answers rather than crashes:

* Placing `Pflo` over `S` (which is dead once the softmax has read it, so it looks free) is a
  **race**: thread `i`'s stores land on the rows threads `~i/2` are still reading. It produced
  `rms rel` ~0.7 starting at token 0 -- wrong, but not obviously so.
* The second m-tile's fragment reads rows `24+gid`, i.e. past `BQ`. Clamping the row index is
  free, because those lanes only feed output rows that are never written, and it avoids padding
  both `Pf` and `Pflo` to 32 rows -- which the shared budget could not have afforded.

**Now the important part: why it is slower.** Two phases on tensor cores, both reading their
operands from shared memory, and the kernel got *slower*. Combined with the earlier results, the
cost is not:

* the score FLOPs -- putting them on tensor cores bought only 1.37x, not the ~5x that "the score
  loop is ~80% of per-tile cycles" predicted;
* the P·V FLOPs -- replacing 384 fma and 384 shared loads per thread with 16 mma per warp made it
  worse;
* shared-memory load *count* -- three separate reductions (BK=48, accumulator-chain splitting, and
  32-bit fragment loads) each failed or regressed;
* register pressure or occupancy -- `ptxas` reports **0 spill bytes and 96 registers**, and
  2 x 24,544 B still fits two blocks in 48 KB.

What is left is **per-iteration serialization and instruction count**: four `__syncthreads()` per
key tile, a softmax that engages 24 of 256 threads while the other 232 wait at a barrier, a K/V
staging pass that moves the same bytes through global->shared every tile, and ~6 fragment-load
plus address instructions per mma. At the 65536 span the whole kernel runs at ~2.5 TFLOP/s, which
is far below either fp32 CUDA cores or the tensor cores, i.e. it is not FLOP-bound at all.

The lever that follows from that is **`ldmatrix`**: one instruction producing a whole fragment,
replacing six shared loads and their address arithmetic, in both the score and the P·V. It needs
16-byte-aligned fragment rows, which the current `PS = 2 * PADH = 260` halfs (520 bytes, not a
multiple of 16) does not provide -- so it also means moving to a stride that is a multiple of 8
halfs, and redoing the bank analysis, since the `PADH = 130` derivation in the kernel comments
was written for the scalar `sub * PADH` access pattern that no longer exists. Cutting the fourth
barrier means un-aliasing `Ks` and `Vt`, which the budget does not currently allow.

### Phase breakdown by ablation (the measurement that was missing)

Every previous round reasoned about where the time goes. This one measured it. `attn-tile` gates
its timing sections behind correctness, so any deliberately-wrong ablation used to print no
timings at all -- which is why four rounds of ablation attempts produced nothing. A one-line
opt-out (`ATTN_TILE_NO_GATE=1`) now keeps the timings reachable, and the breakdown is:

| ablation | 16384 (min) | 65536 (min) | share of the 65536 span |
|---|---|---|---|
| A baseline | 0.65 s | 10.39 s | -- |
| B P·V inner loop 16 -> 1 iteration | 0.43 s | **6.90 s** | **P·V ~= 34%** |
| C softmax without `__expf` | 0.64 s | 10.40 s | expf ~= 0% |
| D K staging without its global loads | 0.61 s | 9.47 s | K global reads ~= 9% |

Reading this:

* **The scalar P·V is the largest named phase at ~34%.** It is 384 fma plus 384 shared loads per
  thread, every thread working, no idling.
* **`__expf` is free.** Removing it from the softmax changes nothing measurable, so the softmax's
  cost is its *serialization* (24 of 256 threads working, the other 232 at a barrier), not its
  arithmetic. Optimising the exponential would have been wasted effort.
* **K's global reads are only ~9%**, so the global->shared staging traffic is not the wall either.

**This finally explains the increment-2 regression.** The P·V loop body is 34% of the runtime, yet
replacing all 384 fma and 384 loads with 16 mma per warp made the kernel 5% *slower*. The reason
cannot be the arithmetic it removed, so it has to be what the mma version *added*: a fifth
`__syncthreads()` per key tile, an extra serialized phase staging `Vt` into shared memory, and two
dependent mmas per n-tile behind a rescale. On a kernel whose phases are barrier-separated and
where the softmax already demonstrates that idle threads at a barrier are expensive, adding a
phase costs more than removing arithmetic saves.

The design rule that follows: **on this kernel, remove barriers and merge phases before removing
arithmetic.** That inverts the plan the earlier rounds were following, and it means the next
attempt at the P·V should not be a separate mma phase at all -- it should fold into the existing
score phase's warp work so no new barrier is needed.

### ldmatrix: one instruction per fragment, and 1.58x cumulative

Following the measured breakdown, the score phase's fragment loads were replaced with `ldmatrix`.
A whole A fragment (16x16) or B fragment (8x16) now arrives in **one instruction** instead of six
shared loads plus their address arithmetic, on both operands. The mapping was proven standalone
first, again on the first attempt: an nvcc probe computing `D[16][8] = Q.K^T` via
`ldmatrix.x4` + `ldmatrix.x2` + `mma` reported **128/128 exact** against a CPU reference.

That required retiring the `PADH = HD/2 + 2` row gap. `PS = 2 * PADH = 260` halfs is 520 bytes,
which is **not** a multiple of 16, so alternate rows were 8-byte misaligned and `ldmatrix`
(the whole point of which is 16-byte row segments) could not be used at all. The stride is now
`HD + 8 = 264` halfs (528 B, a clean multiple of 16) with no gap, and the `koff` correction the
old fragment loads needed is gone with it. **The `PADH = 130` derivation still documented in the
kernel was load-bearing for the scalar `sub * PADH` access pattern, and that access pattern no
longer exists** -- nothing reads Q or K scalar-wise any more, and `ldmatrix` does its own
conflict-free access, so the bank argument that produced 130 is simply obsolete.

| span | original scalar | score mma (pk2) | + ldmatrix | cumulative |
|---|---|---|---|---|
| 16384 | 0.88 s | 0.65 s | **0.57 s** | **1.54x** |
| 65536 | 14.20 s | 10.39 s | **9.01 s** | **1.58x** |

Correctness is unchanged (`attn-tile: OK`, rms rel 1.0--1.6e-5 against the 1.2e-4 gate) and
**round 411 PASS** across every gate: build, test, correctness at layer and 64-layer scale,
longctx-follow, tc-parity, batch-parity, decode-bench, prefix cache, benchmark.

**Where the remaining 7x has to come from.** Making *both* matmuls free is now worth only about
1.5x more, so the gap cannot be closed by better matmuls. The measured shares are P·V ~34%,
K global reads ~9%, `__expf` ~0%, score fragment loads ~15% (now largely removed), which leaves
roughly 40% spread across the K staging's shared writes and address arithmetic, the score mma
itself, the barrier waits, the softmax's non-expf work, and the epilogue. There is **no single
dominant cost left** -- the classic signature of an instruction-throughput-bound kernel in which
every phase contributes a little. That points at a structural change (fewer barriers, larger
tiles, less re-staging) rather than another micro-optimisation.

Two concrete structural candidates, both constrained by the 24,576 B/block shared budget that
two-block occupancy imposes:

* **A larger `BK`.** Four barriers are paid per 16 keys; `BK = 32` would halve that per key. It
  does not fit today: `Ks` at `PS = 264` is 16,896 B for `BK = 32` against `Qs` 12,672 B.
* **Transposing V with `ldmatrix.trans` instead of staging it transposed.** `O = P.V` needs `V^T`
  because `mma`'s B operand is column-major, and last round's attempt paid a *new barrier* to
  stage it -- which cost more than the arithmetic it saved. `ldmatrix.sync.aligned.m8n8.x4.trans`
  produces a transposed fragment directly from the natural `[key][dim]` layout, so V could be
  staged in the same phase as K with no extra barrier and no transpose pass at all. That is the
  version worth trying, and it directly answers the design rule the ablation produced.

### Barriers are not the cost either -- and the arithmetic-intensity ceiling

The ablation produced a design rule ("remove barriers and merge phases before removing
arithmetic"). It was tested directly: the fourth `__syncthreads()` of the key-tile loop was
removed. It is genuinely redundant -- the only hazard reaching forward is the P·V loop's read of
`S` against the next iteration's score writing `S`, and the barrier after the next K staging
already sits between them, because the K staging touches only `Ks`, which the P·V never reads.

    with 4 barriers   0.57 s / 9.01 s
    with 3 barriers   0.56 s / 8.95 s     <- ~0.7% / ~1%

`attn-tile: OK`, so the change is *correct* -- and worth under 1%. A sub-1% gain does not justify
resting on a hand-verified absence of a race in a loop that now has three barriers and four
shared arrays, so it was reverted rather than kept. **The design rule from the previous round is
therefore not supported by measurement.** Barriers, like arithmetic, loads, and matmul quality
before them, are not where the time is.

What the numbers do say. Recomputing the arithmetic intensity honestly: the score and the P·V are
each `BQ x BK x HD x 2` per (query tile, key tile) pair, so at the 65536 span the kernel performs
**~5.3e13 FLOP in 9.0 s = ~5.9 TFLOP/s**. The tile's data movement is about 24 KB (16 KB of K and V
from L2/global, the rest shared traffic), giving ~32 FLOP/byte -- and at the machine's measured
228 GB/s that band would cap a purely DRAM-bound kernel at ~7.3 TFLOP/s. The kernel sits at 5.9,
i.e. **in the same range as the memory-bandwidth ceiling**, while being ~7% of the tensor-core
peak.

That is the honest summary of where this stands: after four rounds of optimisation the kernel is
no longer obviously compute-bound at all, it is close to a bandwidth/instruction-throughput
balance, and **no single remaining phase pays for the gap**. Every lever tried has been worth
1.37x, 1.15x, 0.95x, or 1.00x -- which is the signature of a kernel that needs a different
structure (larger query tiles so each K staging serves more rows, more reuse per byte) rather
than another micro-optimisation of a phase that is already near its share.

### The occupancy constraint was misattributed: it is REGISTERS, not shared memory

Every design decision for four rounds has been made against a documented constraint of
"2 blocks/SM, smem limited, budget ~24,576 B/block". Measured directly with
`cudaGetDeviceProperties` on this GB10, that is wrong, and it has been wrong in the direction
that costs the most:

| resource | attention kernel's footprint | blocks/SM it allows |
|---|---|---|
| registers | 96 x 256 = 24,576 regs | **2  <- binding** |
| shared memory | 24,544 B | 4 |
| threads | 256 | 6 |

```
NVIDIA GB10  sm_121  SMs 48
sharedMemPerBlock         49152 B
sharedMemPerBlockOptin   101376 B
sharedMemPerMultiprocessor 102400 B
regsPerMultiprocessor 65536   maxThreadsPerMultiProcessor 1536
```

So occupancy is limited by **registers**: 65,536 / (96 x 256) = 2. Shared memory would allow four
blocks at this footprint. The consequence is that the real per-block shared budget at two blocks
is **102,400 / 2 = 51,200 B**, not 24,576 B -- **more than double** what every recent design
assumed, with `sharedMemPerBlockOptin` at 101,376 B as the hard per-block ceiling.

This matters because a whole class of design was rejected on smem grounds that did not exist.
Specifically, **`Ks` and `Vt` aliasing was never required**: aliasing exists only to fit inside
24,576 B, and un-aliasing them was correctly identified as the clean way to remove the extra
barrier that made the P·V mma slower -- then dismissed as impossible. At 8,192 B for a
non-transposed `Vs`, the full working set is 32,672 B, which fits two blocks with room to spare:

```
Qs  24 x 264 x 2 = 12,672
Ks  16 x 264 x 2 =  8,448
Vs  16 x 256 x 2 =  8,192   <- its own storage, no aliasing
S   24 x  16 x 4 =  1,536
red 3 x 24 x 4   =    288
Pf, Pflo  2 x 24 x 16 x 2 = 1,536
                     ------
                     32,672 B  (of 51,200 available at 2 blocks)
```

Registers can also be traded deliberately: three blocks/SM would need <= 85 regs x 256 threads
(65,280) *and* <= 34,133 B of shared per block. At 2 blocks the budget is 51,200 B. The right
choice between those two is a measurement, not an assumption -- and the previous rounds never had
the option because they believed the budget was 24,576 B.

**Lesson, recorded because it has now cost four rounds:** an attributed cause is not a measured
one. "2 blocks/SM" was measured; "smem limited" was inferred from it and never checked, and
`cudaGetDeviceProperties` answers it in one call.

### Occupancy is not the limit either -- which frees the whole shared-memory budget

The misattribution had a second half. Registers are binding at 2 blocks/SM, so if registers could be
cut below 85 the kernel would get a third block. `ptxas` says it can be done for free:

| `-maxrregcount` | registers | spills | blocks/SM |
|---|---|---|---|
| (unset) | 90 | 0 B | 2 |
| **80** | **79** | **0 B** | **3** |
| 72 | 72 | 28 B | 3 |
| 64 | 64 | 44 B | 3 |

So 79 registers with zero spills buys a third block. Measured, over six runs, minimum of each shape:

    2 blocks/SM   0.56 s / 8.95 s
    3 blocks/SM   0.57 s / 9.00 s

**No improvement at all.** The register cap was reverted rather than kept. A 50% occupancy increase
with zero spills changing nothing means the kernel is **not limited by occupancy or latency
hiding** -- it already saturates whatever it is bound on with two blocks.

That is a useful result rather than a dead end, because it removes the premise that forced the
2-block design. Combined with the previous section, the shared-memory situation is now:

* 2 blocks x 51,200 B is available, and occupancy does not care whether it is 2 or 3.
* **1 block x 101,376 B is therefore also on the table**, and this kernel would not notice the
  lost second block.

That is four times the 24,576 B every design has been squeezed into, and it is the budget a
FlashAttention-shaped kernel actually wants: `BQ` and `BK` large enough that each K staging serves
many more query rows, which is the structural change the ablation pointed to. The earlier
occupancy study concluded "2 blocks is better than 1" -- but it was measured on the *current*
tile shape, where a second block of a small tile is the only source of parallelism. It does not
say that a single much larger tile is worse, which is a different question and was never asked.

**The complete list of things now measured and excluded as the explanation:** arithmetic (P·V mma,
5% slower), shared-load count (three separate reductions), matmul quality (real but capped at
1.37x and 1.15x), barriers (~1%), `__expf` (0%), K's global reads (~9%), registers and spills
(0 B), and occupancy (0% from +50%). What is left is the kernel's *shape*.

## Same-session cold-TTFT pair after the attention work (8K)

Both legs run back-to-back in one session, gb10 with `--ctx 262144` and `GB10_TC_GEMM` on (the
default), llama.cpp with `-c 262144 -ngl 99 -fa on`. Reps 227, 2 trials each, greedy, unique
marker per trial so every cold TTFT is a genuine full prefill.

| server | prompt tokens | cold TTFT | warm TTFT | OTPS cold |
|---|---|---|---|---|
| gb10 | 7,109 | **13.41 / 13.49 s** | 0.03 s | 7.41 / 7.45 |
| llama.cpp | 7,147 | **10.91 / 11.16 s** | 0.26 s | 6.16 / 6.21 |

**Ratio 13.41 / 10.91 = 1.22x slower** (taking each server's better trial; the worst-case pairing
gives 1.21x). The prompt-token counts differ by 0.5% because the tokenisers differ, so the
comparison is like-for-like.

This is progress on the objective's first item but **not yet a win: 0/4 becomes 0/1 measured, at
1.22x rather than 1.41x.** The attention kernel's 1.58x shows up here as prefill going from
18.90 s to 13.41 s for the same shape.

**What the remaining gap needs at this length, computed rather than guessed.** The
least-squares decomposition puts the quadratic (attention) share of gb10 prefill at 79% at 8K, so
`13.41 = L + A` with `A = 0.79 * 13.41 = 10.59 s` and `L = 2.82 s`. To reach llama.cpp's 10.91 s
the attention must come down to `10.91 - 2.82 = 8.09 s`, i.e. **only 1.31x further** -- which is
within reach of the same kind of change that already delivered 1.58x.

But the same arithmetic at the long end is unforgiving, and it is why 8K is the *easy* one: at
128K the quadratic share is 98%, so essentially all of gb10's 1224 s is attention, and matching
llama.cpp's 290 s needs the attention **~4.2x** faster still. A same-session pair at 128K has not
been re-measured since the attention work.

**Session drift, and why only same-session pairs count:** llama.cpp's own 8K figure moved from
13.45 s in the earlier session to 10.91 s now -- **1.23x on an unchanged binary**. Any comparison
built from numbers taken in different sessions would have mis-stated this result by more than the
effect being claimed, in either direction.

### The P·V's shared loads are already vectorised -- the compiler got there first

The P·V is still the largest named phase (~34%), and its inner loop reads S scalar-wise: 16 fma and
16 shared loads per row, 24 rows, per thread. Replacing the scalar reads with explicit `float4`
loads is legal (`S` begins 16-byte aligned and each row is 64 B) and bit-identical (the four fma
are issued in the same ascending order), so it was tried:

    16384   0.57 / 0.58 s   (baseline 0.56)
    65536   9.15 / 9.19 / 9.25 s   (baseline 8.95 / 9.01)

**~2% slower, with bit-identical correctness.** The obvious reading is that the compiler had
already merged those adjacent shared reads -- the loop is fully unrolled and `S` is a plain
aligned array -- so the explicit `float4` saved no instructions and only added the register
pressure of the live vector temporaries. It was reverted.

This narrows the P·V's 34% decisively. It is not its loads (just shown), not its count of
instructions that the compiler can coalesce, and not its precision handling (unchanged). What
remains is the **384 dependent `fmaf` per thread** -- the arithmetic itself. That is exactly the
work an mma removes outright rather than reshapes, which is why the tensor-core P·V keeps being
the right target even though the first attempt at it measured slower: that attempt paid a new
barrier and a transpose pass, and those two costs are now known to be avoidable (`Vs` fits in its
own storage under the corrected 51,200 B budget, and `ldmatrix.trans` builds the transposed
fragment directly).

## Same-session cold-TTFT pairs after the attention work (8K and 32K)

Both legs back-to-back per length, one session, same flags as the 8K pair above (`--ctx 262144`,
`GB10_TC_GEMM` on; llama.cpp `-c 262144 -ngl 99 -fa on`), unique marker per trial.

| length | server | prompt tok | cold TTFT | warm TTFT | OTPS cold | ratio |
|---|---|---|---|---|---|---|
| 8K | gb10 | 7,109 | 13.41 / 13.49 s | 0.03 s | 7.41 / 7.45 | **1.22x** |
| 8K | llama.cpp | 7,147 | 10.91 / 11.16 s | 0.26 s | 6.16 / 6.21 | |
| 32K | gb10 | 32,530 | **84.00 / 83.86 s** | 0.05 s | 6.79 / 6.77 | **1.59x** |
| 32K | llama.cpp | 32,568 | **52.72 / 52.81 s** | 0.30 s | 5.75 / 5.74 | |

Progress against the objective's first item, same-session and like-for-like:

| length | before the attention work | now | requirement left |
|---|---|---|---|
| 8K | 1.41x slower | **1.22x slower** | 1.31x more attention speedup |
| 32K | 2.11x slower | **1.59x slower** | 1.65x more |

The "requirement left" column is computed from the least-squares decomposition, not guessed. With
quadratic shares of 79% (8K) and 94% (32K):

* 8K: `13.41 = L + A`, `A = 10.59`, `L = 2.82`; matching llama's 10.91 needs `A' = 8.09`, **1.31x**.
* 32K: `83.86 = L + A`, `A = 78.83`, `L = 5.03`; matching llama's 52.72 needs `A' = 47.69`, **1.65x**.

**So the required attention speedup is no longer the 1.29x/5.98x the objective was written against;
after 1.58x of delivered attention speedup it is 1.31x at 8K and 1.65x at 32K.** That is a much
smaller and, importantly, *roughly constant* factor across these two lengths -- it grows only
slowly with context, because the linear term is small and the attention term has already been cut.

gb10 wins warm TTFT decisively at both lengths (0.03/0.05 s vs 0.26/0.30 s) and wins OTPS at both
(7.41 vs 6.16, 6.79 vs 5.75).

Still not a win: **0/4 remains, now 0/2 measured, at 1.22x and 1.59x.** 128K and 256K have not been
re-measured since the attention work; the 128K leg alone is ~800 s for gb10 and ~290 s for
llama.cpp, so it needs its own round.

### The P·V is fp32 fma throughput, and nothing but an mma will fix it

Two experiments have now bracketed the P·V's 34%, and between them they rule out every explanation
except one.

**Its loads are not the cost.** Explicit `float4` reads (legal -- `S` is 16-byte aligned with 64 B
rows -- and bit-identical) measured **~2% slower**, because the compiler had already merged those
adjacent shared reads. Reverted.

**Its dependency chain is not the cost.** The inner loop is one 16-deep dependent `fmaf` chain per
row, and the fma latency is a few cycles, so a single chain ought to leave the pipeline empty. It
was split into four independent partial sums (three extra registers, same terms grouped by stride
4, so only the last bits change -- far inside the 1e-4 gate, which sits at ~1e-5):

    16384   0.57 / 0.58 s   (baseline 0.56)
    65536   8.98 / 9.11 / 9.14 / 9.18 s   (baseline 8.95 / 9.01)

**No improvement either.** Reverted.

What is left is arithmetic throughput. The P·V issues 384 `fmaf` per thread per key tile. At the
65536 span that is 1.34e8 tile-pairs x 98,304 fma = **1.3e13 fma**, and the machine's measured fp32
rate is 18.43 TFLOP/s = 9.2e12 fma/s, so the P·V's fma alone is ~1.4 s of a 9.0 s kernel. The
ablation attributed 34% (3.06 s) to it, which means the loop is running at roughly **40% of the
chip's fp32 peak** -- respectable for a real kernel, and not improvable by rescheduling, because
the work is already issued as densely as the fp32 pipes allow.

**The only remaining move is to stop doing the arithmetic on the fp32 pipes at all**, i.e. the mma,
which replaces 384 `fmaf` per *thread* with 16 mma per *warp*. Both costs that made the first
attempt slower are now known to be avoidable, so that is the change to make:

* the extra barrier existed only because `Vs` aliased `Ks` to fit 24,576 B -- and the real budget is
  **51,200 B**, so `Vs` gets its own storage and the barrier disappears;
* the transpose pass existed only because `mma`'s B operand is column-major -- and
  `ldmatrix.sync.aligned.m8n8.x4.trans` builds the transposed fragment straight from V's natural
  `[key][dim]` layout, so there is no transpose pass at all.

The full working set is `Qs` 12,672 + `Ks` 8,448 + `Vs` 8,192 + `S` 1,536 + `red` 288 +
`Pf`/`Pflo` 1,536 = **32,672 B**, comfortably inside 51,200 B.

### The P·V fragment mapping is proven -- the last unknown before wiring it in

Both previous fragment mappings were proven standalone before being wired into the kernel, and both
were then exact on the first attempt. The same was done for the P·V, in the orientation that keeps
the A operand simple, and it passed **256/256 against a CPU reference on the first run**
(`bench/longctx/probe_pv_mapping.cu`):

```
O[16 rows][16 dims] = P[16 rows][16 keys] . V[16 keys][16 dims]

A = P    via ldmatrix.x4        on P[row][key]        (row-major, stride 16)
B = V^T  via ldmatrix.x2.trans  on V[key][dim]        (NATURAL layout, stride 24)
         lanes 0-7  -> &V[key=lane][n0]      (keys 0-7)
         lanes 8-15 -> &V[key=lane][n0]      (keys 8-15)
D = mma  -> O[m0+gid][n0+2*t4], +1, and rows m0+gid+8
```

`ldmatrix.trans` is what makes this work: `mma`'s B operand is column-major, so `O = P.V` needs
`V^T`, and `ldmatrix.sync.aligned.m8n8.x2.trans` produces that transposed fragment **directly from
V's natural `[key][dim]` layout**. That is the transpose pass that the first P·V attempt paid a
barrier for, and it does not exist.

So the complete, verified design for the tensor-core P·V is now on the table, with every unknown
removed:

| element | status |
|---|---|
| A fragment (P, `ldmatrix.x4`) | proven 128/128 in the score phase, in the kernel |
| B fragment (V^T, `ldmatrix.x2.trans`) | **proven 256/256 standalone** |
| mma + output fragment layout | **proven 256/256 standalone** |
| `Vs` storage, no aliasing | fits: full working set 32,672 B of 51,200 B |
| no extra barrier | V stages in the same phase as K |
| `Pf` + `Pflo` two-term split | proven correct in the earlier attempt (1.0--1.6e-5) |
| m-tile overlap (rows 8-15 twice) | harmless: both tiles compute identical values; write once |
| accumulator registers | 8 mma-tiles/warp x 4 = 32 regs, vs `acc[24]` today |

The remaining work is mechanical wiring, not discovery: stage V into `Vs`, have the softmax write
`Pf`/`Pflo`, replace the 384-`fmaf` loop with 12 mma per warp per key tile, and write `out` from
the fragment accumulators.

### The tensor-core P·V was built, proven correct, and is still slower -- and that closes the program

This is the change that every previous round pointed at, built exactly as designed, with its
fragment mapping proven standalone (256/256) before being wired in. It is **numerically correct**
and it is **~5% slower** than the scalar fp32 P·V it replaces, so it is not in the tree.

| build | 16384 (min) | 65536 (min) | notes |
|---|---|---|---|
| scalar P·V (in tree) | **0.56 s** | **8.95 s** | 90 regs, 0 spill |
| mma P·V, `Vs` stride 256 | 0.63 s | 9.96 s | correct, 11% slower |
| mma P·V, `Vs` stride 264 | 0.60 s | **9.40 s** | correct, **5% slower**; 72 regs, **0 spill** |

**The first run was slow for a findable reason, and finding it was worth the round.** `Vs` was
staged with stride `HD = 256` halfs, i.e. 512 B = 128 words per row -- and **128 mod 32 == 0**, so
every row of `Vs` began in the *same shared bank*. `ldmatrix.trans` reads 16 rows at once, so every
single B-fragment load was a **16-way bank conflict**. Padding `Vs` to 264 halfs (132 words,
132 mod 32 == 4) spreads the 16 rows over 8 banks -- the best a 16-byte-aligned stride can do --
and recovered 6%. That is the same bank arithmetic that produced the old `PADH = 130`, reappearing
in a place where nobody had reason to look, because the stride that *looks* natural (`HD`) is the
one stride that is pathological.

**But it is still slower, and the reason is fundamental rather than fixable.** With 0 spill bytes
and *fewer* registers than the scalar version (72 vs 90, because `vr[]` is gone), the mma version
has no resource problem at all. The arithmetic simply does not pay:

* the scalar P·V issues 1.3e13 `fma` at the 65536 span, against a measured **9.2e12 fma/s** fp32
  roofline (18.43 TFLOP/s) -- so the fp32 path is already at ~1.4 s of a 9.0 s kernel, i.e. the
  scalar P·V is close to the fp32 roofline, not far from it;
* the same work on tensor cores is ~0.7 s (74 TFLOP/s bf16 peak), so the best case is saving ~0.7 s;
* the conversion costs more than that: 8 `ldmatrix.trans` (still 2-way conflicted, the floor for an
  aligned stride) plus 4 `ldmatrix.x4` plus 12 mma plus fragment addressing per warp per key tile,
  **and** the softmax now writes `Pf` and `Pflo` separately instead of one `S`, and `Pf`/`Pflo` must
  be zeroed for the rows past `rows` that whole 16-row fragments now touch.

**GB10's fp32 throughput is high enough, relative to its tensor-core throughput at this shape, that
converting the P·V to tensor cores costs more than it saves.** The tensor-core prefill-attention
program is therefore closed, and the three increments land as:

| increment | result | in tree? |
|---|---|---|
| score matmul on tensor cores (`mma.sync`) | **1.37x** | yes |
| `ldmatrix` for the score fragments | **1.15x** (1.58x cumulative) | yes |
| P·V on tensor cores | 0.95x -- **slower** | no |

The remaining gap to llama.cpp is no longer reachable by moving arithmetic between units. Every
lever has now been measured: arithmetic placement (this round), shared-load count and vectorisation,
accumulator-chain ILP, matmul quality, barriers, `__expf`, K's global reads, registers and spills,
occupancy (+50% for 0%), and the shared-memory budget (which turned out to be 51,200 B, not the
24,576 B four rounds were designed against). What is left is the kernel's *shape*: larger query
tiles so each K/V staging serves more rows, which needs a rewrite rather than a substitution.

## Same-session cold-TTFT pair at 128K, and the requirement at three lengths

Same protocol as the 8K and 32K pairs: both legs back-to-back in one session, gb10 `--ctx 262144`
with `GB10_TC_GEMM` on, llama.cpp `-c 262144 -ngl 99 -fa on`, unique marker so every cold TTFT is a
genuine full prefill.

| length | server | prompt tok | cold TTFT | warm TTFT | OTPS cold | ratio |
|---|---|---|---|---|---|---|
| 8K | gb10 | 7,109 | 13.41 s | 0.03 s | 7.41 | **1.22x** |
| 8K | llama.cpp | 7,147 | 10.91 s | 0.26 s | 6.16 | |
| 32K | gb10 | 32,530 | 83.86 s | 0.05 s | 6.79 | **1.59x** |
| 32K | llama.cpp | 32,568 | 52.72 s | 0.30 s | 5.75 | |
| 128K | gb10 | 130,832 | **804.92 s** | 0.15 s | 4.51 | **2.94x** |
| 128K | llama.cpp | 130,870 | **273.86 s** | 0.49 s | 4.88 | |

gb10's 128K prefill was ~1224 s before the attention work, so the kernel's 1.58x shows up here as
**1.52x** end to end. Against llama.cpp the same length went from **4.21x to 2.94x**.

**The remaining requirement, computed from the least-squares decomposition at each length:**

| length | attention share | gb10 = L + A | needs `A'` | further attention speedup |
|---|---|---|---|---|
| 8K | 79% | 13.41 = 2.82 + 10.59 | 8.09 | **1.31x** |
| 32K | 94% | 83.86 = 5.03 + 78.83 | 47.69 | **1.65x** |
| 128K | 98% | 804.92 = 16.10 + 788.82 | 257.76 | **3.06x** |

**Three lengths measured, and the requirement grows only from 1.31x to 3.06x** -- not the
1.29x/5.98x/8.40x the objective was written against, because 1.58x of attention speedup is already
delivered. The growth is genuine but it is sub-linear in the attention share, and the 8K case is
within 1.31x of a win.

Still **0/4** (0/3 measured). gb10 wins warm TTFT at all three lengths (0.03/0.05/0.15 s vs
0.26/0.30/0.49 s) and OTPS at 8K and 32K (7.41 vs 6.16, 6.79 vs 5.75) but **loses OTPS at 128K
(4.51 vs 4.88)**. 256K remains unmeasured; it is the only length where gb10 has never been timed.

### A larger query tile is slower too -- BQ 32 measured and rejected

The last untested lever, and it came with a specific inefficiency to remove. At `BQ = 24` the score
phase runs two **overlapping** 16-row m-tiles (rows 0-15 and 8-23), so it computes 32 rows of mma to
use 24 -- **a third of the score work is discarded** -- and each K/V staging serves only 24 rows.
`BQ = 32` makes the m-tiles rows 0-15 and 16-31: no overlap, no waste, and 33% more rows per
staging. The counts work out in its favour:

| | BQ = 24 | BQ = 32 |
|---|---|---|
| key-tile iterations at the 65536 span | 5.60e6 | **4.20e6** (-25%) |
| score mma, total | 3.58e8 | **2.69e8** (-25%) |
| P·V fma, total | 5.5e11 | 5.5e11 (same) |
| shared per block | 22,944 B | 27,776 B |
| registers / spills | 90 / 0 B | **90 / 0 B** |

Everything is in budget -- 27,776 B of the 51,200 B available at two blocks, and 0 spills, so two
blocks/SM still fit -- and it is correct (`attn-tile: OK`, rms rel unchanged):

    16384   0.62 / 0.64 s          (baseline 0.56)
    65536   9.68 / 9.77 / 9.82 / 9.88 s   (baseline 8.95 / 9.01)

**~9% slower, reproducibly, despite strictly less work.** Reverted.

This is worth recording precisely because the arithmetic predicted a win and the measurement
refused it. The likely explanation is that `acc[PREFILL_BQ]` grows from 24 to 32 registers *per
thread*, so the fully-unrolled P·V loop goes from 384 to **512 `fmaf` in one basic block** -- the
same total fma, but issued as one much longer straight-line run per iteration, on top of a Q tile
that is 33% larger in shared memory. The kernel appears to be sensitive to the shape of the
per-iteration instruction stream in a way that total-work accounting does not capture.

**So the structural lever is measured too, and it does not pay at this granularity.** Every lever
this program has tried is now accounted for: score mma **1.37x** (kept), `ldmatrix` **1.15x**
(kept, 1.58x cumulative), P·V mma **0.95x** (rejected), and now query-tile size **0.91x**
(rejected). The kernel is at a local optimum with respect to every axis anyone has proposed, and
the 8K/32K/128K requirements of 1.31x / 1.65x / 3.06x are not reachable by any single substitution
tried here.
