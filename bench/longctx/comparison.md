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
