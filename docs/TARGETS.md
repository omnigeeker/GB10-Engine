# GB10-Engine — performance contract

This is the **definition of done**. It supersedes the original request, whose
numeric targets were shown to be physically unreachable in `docs/PHYSICS.md`
(agreed with the owner before implementation began).

## Hardware

DGX Spark GB10, `sm_121`, 48 SMs, 121 GiB unified LPDDR5X, CUDA 13.0.
Read bandwidth measured at **228 GB/s** (incompressible probe) and **250 GB/s**
demonstrated on the real weight set.

## Model

`nv-community/Qwen3.8-27B-NVFP4` — `qwen3_5` hybrid, 64 layers
(48 Gated-DeltaNet linear-attention + 16 full-attention), hidden 5120,
vocab 248320, **1 MTP layer**, modelopt MIXED_PRECISION
(NVFP4 group-16 MLP + lm_head, FP8 attention projections).
**17.608 GB of dense text weights read per token.**

## Targets (all must be measured, not estimated)

| # | Metric | Target | Rationale |
|---|---|---|---|
| T1 | Single-stream decode | **>= 12.5 tok/s** | 95 % of the 12.95 tok/s conservative roofline |
| T2 | Single-stream with MTP | **>= 25.0 tok/s** | >= 2.0 accepted tokens per weight read |
| T3 | 16-concurrent aggregate | **>= 195 tok/s** | 16 x 12.95 tok/s roofline, minus overhead |
| T4 | 16-concurrent with MTP | **>= 390 tok/s** | MTP amortisation at batch 16 |
| T5 | TTFT @ 1 concurrent, 512-token prompt | **better than Ollama** | prefill is compute-bound, not BW-bound |
| T6 | TTFT @ 16 concurrent | **better than Ollama** | chunked prefill + continuous batching |
| T7 | Correctness | **top-1 token match vs the NVFP4 reference** on the fixed prompt set, >= 32 greedy tokens | no quality regression vs reference |
| T8 | Endpoints | OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/models`; Anthropic `/v1/messages`; streaming and non-streaming | drop-in local replacement |
| T9 | Concurrency | 16 simultaneous requests served without error | required by owner |

### Note on T7

The HF oracle in `/home/wayne/KimiCode/Deploy_QWen3.8-27B/oracle` runs the
**BF16** base model, because `transformers` cannot load a pre-quantized
`modelopt` NVFP4 checkpoint. This engine runs the **NVFP4** weights, so
token-exact agreement with a BF16 reference is not expected: 4-bit
quantization perturbs logits by design.

The correctness gate is therefore two-tiered:

1. **Exact**: top-1 token match against an NVFP4 reference produced by
   dequantizing this checkpoint and running it in bf16 (to be generated in M3).
2. **Agreement**: top-1 agreement rate against the BF16 oracle, reported as a
   percentage, with a floor of 85 % on the frozen prompt set.

Both numbers are reported; neither is presented as the other.

### Explicit non-goals

- **100 tok/s single-stream on this model is not a target** — it requires
  1.76 TB/s, 7.7x the measured hardware limit. It is recorded as impossible.
- Metal/macOS backend — the owner selected CUDA-on-GB10 only.
- Vision/multimodal inference. The vision tower is loaded but not exercised;
  text-only is the contract.

## MTP is the strongest remaining lever (measured round 40)

`gb10-verify mtp-probe` loads the head and prices it. The head is unquantized
BF16 while the decoder it drafts for is FP4, which is the thing that could have
made it too expensive to be worth running:

| | B/token | vs decoder |
|---|---|---|
| decoder step | 17,602,479,684 | 100% |
| MTP head | 849,451,008 | **4.8%** |

So one MTP step costs `17.60 + 0.85 = 18.45 GB` and, when the draft is
accepted, emits **two** tokens instead of one:

| acceptance | GB per token | speedup |
|---|---|---|
| 1.00 | 9.23 | 1.91x |
| 0.80 | 10.25 | 1.72x |
| 0.70 | 10.85 | 1.62x |
| 0.50 | 12.30 | 1.43x |

Even at 50% acceptance the head pays for itself with room to spare, because it
is 21x cheaper than the step it is drafting for. This is a far better lever than
further tuning of the batch GEMV, which is already at the register wall
(ROWS=4 is optimal, 128 registers + 256-byte spill; every alternative measured
slower). At the current 42.44 tok/s at n_seq=16, a 1.6x gain lands at ~68 tok/s
-- past the "50+ at concurrency 16" target.

### The head is implemented and verified with a control (rounds 41-42)

`Mtp::forward` runs the full chain -- gather the next token's embedding, norm
both inputs, `concat2`, `mtp.fc`, the full-attention layer with its own KV
cache, `mtp.norm`, shared `lm_head` -- and `mtp-probe` measures how often its
draft matches what the decoder actually emits next. Greedy acceptance over 48
steps, four unrelated prompts:

| prompt | acceptance |
|---|---|
| "What is the capital of France?" | 91.7% |
| "Explain how a transformer attention mechanism works, step by step." | 85.4% |
| "Write a Rust function that reverses a linked list and explain it." | 95.8% |
| "List the planets in order, then describe each one in a sentence." | 89.6% |

Average ~90%, and the spread across unrelated subjects is what rules out
repetition inflating the number.

### The 1.81x table above was WRONG -- drafting does not replace a decoder step

A naive per-token speculative loop (draft with the head, verify with one
`Model::step`, accept/reject) was implemented and measured:

```
greedy reference :  48 tokens in 5.492s -> 8.74 tok/s
MTP speculative  :  48 tokens in 5.809s -> 8.26 tok/s
draft acceptance : 43/47 = 91.5%
speedup          : 0.95x
token-exact vs greedy: YES
```

Token-exact and 91.5% acceptance, but **no speedup at all**. The reasoning error
in the table above was treating the draft as a *replacement* for a decoder step.
It is not. The decoder step on `x_{t+1}` is what produces `h_{t+1}` and the true
`x_{t+2}`; the draft only tells you what that step will say, and the step still
has to run to move the state forward. Since every emitted token needs one
17.6 GB weight read either way, the loop is exactly break-even -- the algebra
confirms it: a round emits `1 + accepted` tokens but only when the previous
round was rejected, so the long-run rate is `P(reject) * (1 + a) + P(accept) * a
= 1` token per step, independent of acceptance.

### Where the speedup actually has to come from

The head is only cheap because it reuses the decoder's hidden state. To turn
that into throughput, **several drafted tokens must be verified in one decoder
forward** -- a `K+1`-token batched forward reads the weights once, so `K+1`
tokens cost `17.6 + K * 0.85` GB instead of `(K+1) * 17.6`. That is the real
1.9x, and it needs the one thing the engine does not have yet: a multi-token
forward that works from a **non-empty** KV cache. `attn_prefill_kernel`
hardcodes the causal window as `0..=t` rather than `0..=start+t`, which is
exactly why `forward_prefill` is documented as valid only on an empty cache.

So the cost of the verify step grows with `K` while its benefit also grows: each
extra drafted token adds only 0.85 GB, so the marginal drafted token is 21x
cheaper than a decoder step. Realizing it is gated on the batched-verify path,
not on the head.

### The non-empty-cache prerequisite is now in place (round 44)

`attn_prefill_kernel` was querying the *new tokens'* k/v with the query count set
to `pos + t`, and its causal window was hardcoded to `0..=t`. That only works
when `pos == 0` -- with a non-empty cache the query rows ran off the end of the
buffer and every row attended to the wrong key range.

It now takes `start` (keys already in the cache) and `kv_base` (this sequence's
slice), queries the **cache** rather than the scratch k/v, and windows each
query row over `0..=start+t`. The single call site in `forward_prefill` passes
`t` queries, the cache, and `pos`. The dynamic `scores[]` allocation is sized
`start + n_tokens`.

This is behaviour-preserving on the path that exists today -- prefill from an
empty cache, where the cache slice just written equals the old scratch k/v -- and
both gates confirm it:

```
generate      : oracle agreement 16/16, exact match          OK
batch-parity  : 16/16 sequences exact, 40.75 tok/s @ n_seq=16 OK
```

Note the honesty of that evidence: it proves **no regression**, not that the new
capability works, because nothing on the existing path exercises `start > 0`.

### ...and is now covered by a test that drives it (round 45)

`gb10-verify chunked-prefill` prefills the same prompt one-shot and in two
chunks, then compares the generated tokens. The second chunk is the only place
in the engine where attention runs with a non-empty cache:

```
prompt 32 tokens, split 16+16
one shot : [271, 16, 11, 220, 17, 11]
two chunk: [271, 16, 11, 220, 17, 11]
decoded  : "\n\n1, 2,"
agree: YES
```

The decoded text is a sensible continuation of the prompt, which matters: two
identical-but-degenerate outputs would satisfy the comparison while proving
nothing. A `start`-related break can only affect the two-chunk path, so this
comparison is exactly the discriminator the round-44 change needed.

The non-empty-cache forward is therefore verified, and the batched MTP verify
path can now be built on it.

### ...and the second prerequisite, per-row scoring (round 46)

Verifying `K` drafted tokens in one forward needs the decoder's prediction at
*every* drafted row, but `prefill_seq` scores only the last row -- deliberately,
since over a full prompt that would re-read 715 MB of weights per token.
`Model::all_logits` adds the multi-row case for the verify path only, sized to
`VERIFY_MAX = 64` rows (63 MB of logits buffers, negligible against 121 GB).

It is checked against the path it parallels, not left to trust:

```
all_logits row 31 = 271, prefill_seq = 271
chunked-prefill: OK
```

### ...and the third: making a speculative round reversible (round 47)

Composing the loop surfaced a blocker that is not obvious until you try to write
the reject path. The KV cache is append-ordered, so discarding rejected drafts
is just rewinding `n_keys` -- the slots get overwritten next append. **The Gated
DeltaNet recurrence is not like that.** Its 48 layers carry 150 MB per sequence
of running state that a rejected draft has already folded in, and there is no
way to subtract it back out. A verify pass that runs 4 drafted tokens and
accepts 1 would leave the recurrence polluted by 3 tokens that were never
emitted.

`ModelState::snapshot_recurrent` / `restore_recurrent` copy every layer's
recurrent state and cache lengths before a verify pass and put them back after.
That costs ~300 MB of traffic per round against the 17.6 GB a decoder step
reads -- about 1.7% -- which is a fair price for the multi-token verify it
unlocks. `positions` is rebuilt from `n_keys` rather than snapshotted, since it
is a pure function of it.

Verified by making a step happen twice:

```
snapshot/restore: step gave 16 then 16 -> reversible
all_logits row 31 = 271, prefill_seq = 271
```

All three prerequisites for the batched MTP verify now exist and are
individually verified: a forward from a non-empty cache, per-row scoring, and a
reversible round.

### The composed loop is correct but 3x SLOWER (round 48)

`mtp-generate` now runs the full batched scheme: draft `K=4` tokens by chaining
the head's own hidden, snapshot, verify all 4 in one forward, accept the longest
matching prefix, restore, then replay only the accepted tokens to commit.

```
greedy reference :  48 tokens in 5.527s -> 8.68 tok/s
MTP speculative  :  48 tokens in 17.381s -> 2.76 tok/s
draft acceptance : 7/63 = 11.1%  over 21 rounds
speedup          : 0.32x
token-exact vs greedy: YES
```

Correct output, but far slower than even the naive 0.95x version. Two distinct
faults, and they need separating:

1. **Acceptance collapsed to 11.1%** from 91.5% for a single draft. The first
   draft does not depend on the chain, so something in the new plumbing is
   wrong, not the head. Suspects, in order: the chained `ms.out` fed back as the
   next hidden (a bug there would hurt drafts 2-4 but not the 11.1% figure),
   `snapshot`/`restore` completeness, and the row index passed to
   `copy_last_row`.
2. **Each verify/commit forward is far more expensive than a decode step.**
   Both read all 17.6 GB, but at 4-5 rows the prefill GEMM runs at a small
   fraction of the bandwidth a 1-row GEMV manages -- the same small-token GEMM
   inefficiency as M7. So "two weight reads per round" is not two *decode-step*
   costs; the replay roughly triples the round.

### The decisive measurement: rows are nearly free, but the prefill path is 3x off (round 49)

`gb10-verify forward-cost` compares one `t`-row `prefill_seq` forward against `t`
single-row `step`s:

| t | t x step | one t-row forward | ratio | per row |
|---|---|---|---|---|
| 1 | 119.30 ms | 387.81 ms | 3.25 | 387.81 ms |
| 2 | 235.22 ms | 385.18 ms | 1.64 | 192.59 ms |
| 4 | 454.94 ms | 388.08 ms | 0.85 | 97.02 ms |
| 8 | 930.59 ms | 396.03 ms | 0.43 | 49.50 ms |
| 16 | 1844.14 ms | 404.08 ms | 0.22 | 25.25 ms |

Two conclusions, and they point the same way:

**1. Marginal rows are essentially free.** The forward costs ~390 ms whether it
processes 1 row or 16. So batched verification *can* pay -- the round-40
arithmetic was not wrong about that. The fault is the commit replay: two ~390 ms
forwards per round is worse than `accepted + 2` decode steps at 119 ms each. One
forward that could be used directly would emit up to 5 tokens for 390 ms, i.e.
78 ms/token against 119 ms/token -- roughly **1.5x**.

**2. The prefill path runs at 45 GB/s, 20% of the 228 GB/s roofline**, against
the decode GEMV's 148 GB/s (65%). At `t=1` a prefill-style forward is 3.25x
*slower* than a decode step that reads exactly the same 17.6 GB. That is a
kernel-efficiency defect, not a fundamental limit, and it is the same defect
behind the TTFT gap (452.6 ms vs llama.cpp's ~74 ms).

So the highest-value target is now the prefill/GEMM path, and it is doubly
motivated: it is the TTFT number the objective asks for, and it is the
prerequisite for the MTP scheme to pay off at all. Continuing to patch the MTP
pipeline while its verify forward is 3x off would be optimizing the wrong
thing.

Neither the naive nor the batched MTP loop is wired into the server; both are
diagnostics.

### The head was verified with a control, not just a happy path

The two candidate hidden inputs (post-final-norm vs pre-norm residual) produced
*identical* acceptance counts on every prompt tried, which is exactly what
dropping the hidden half of the concatenation would also look like. So the
probe runs a third variant that feeds a **zero** hidden state:

| draft input | acceptance |
|---|---|
| post-final-norm hidden | 44/48 = 91.7% |
| pre-norm residual | 44/48 = 91.7% |
| **hidden half zeroed** | **5/48 = 10.4%** |

The control collapses to near-chance, so the hidden half is genuinely driving
the draft and `concat2` plus the second half of `mtp.fc` are both wired
correctly. The identical counts from the two real variants are therefore a
coincidence of those two vectors agreeing on the argmax, not a dropped input.

This is the same discipline the round-37 failure taught: a passing result on the
happy path proves nothing until a control that *should* fail actually does.

The head's structure is confirmed: `mtp.fc` is `[5120, 10240]` (the
concatenation of the two normalised inputs back down to hidden),
`mtp.layers.0` is a single full-attention layer with the same geometry as the
decoder's full-attention blocks, `mtp.norm` feeds the shared `lm_head`, and
`mtp_use_dedicated_embeddings: false` means it shares `embed_tokens`/`lm_head`.
All 15 tensors load through the existing `Store::linear` BF16 path.

## Measured: 16-way batched decode

Correctness: `gb10-verify batch-parity --n-seq 16 --n 16` reports **16/16
sequences token-exact** against decoding them one at a time, over prompts of
deliberately *different* lengths. Equal-length prompts are not a valid gate:
every sequence then sits at the same position, so a per-sequence addressing bug
reads equivalent data and passes.

Throughput, same kernel set, measured this round:

| n_seq | ms/step | aggregate tok/s | per-sequence tok/s |
|---|---|---|---|
| 1 | 115.1 | 8.69 | 8.69 |
| 2 | 124.9 | 16.01 | 8.00 |
| 4 | 163.2 | 24.51 | 6.13 |
| 8 | 207.1 | 38.63 | 4.83 |
| 16 | 380.3 | **42.08** | 2.63 |

T3 asks for >=195 tok/s aggregate; T1 asks for >=12.5 tok/s single-stream
(roofline ceiling 12.95).

Three steps got here, each measured:

1. The GEMV originally launched with `gridDim.y = batch`, so every block
   streamed the whole weight matrix for its own sequence: 16 sequences cost 16x
   the weight traffic (17.6 GB x 16 = 281 GB ~ 1233 ms at 228 GB/s, against
   1453 ms measured). 11.01 tok/s.
2. Multi-sequence kernels loop over the batch inside the block and load each
   weight tile once. 25.52 tok/s.
3. Profiling showed the batch GEMV was still 545 of ~645 ms/step, ~11x off the
   weight roofline, because with one row per warp each warp pulled all B
   x-vectors through L2 for its own row -- x traffic was ~8x the weight traffic.
   Staging x in `__shared__` once per block made it *worse* (837 ms): two
   `__syncthreads()` per k-tile cost more than the L2 traffic they saved.
   Templating on `ROWS` so one x load feeds ROWS rows worked. On NVFP4:

   | ROWS | ms/step | tok/s |
   |---|---|---|
   | 1 | 624.0 | 25.64 |
   | 2 | 474.7 | 33.70 |
   | 4 | **447.9** | **35.72** |
   | 6 | 470.2 | 34.03 |
   | 8 | 476.5 | 33.58 |

   ROWS=4 is the optimum; beyond it `acc[ROWS][BMAX]` spills. Applying the same
   to FP8 gave 380.3 ms and **42.08 tok/s**.

### Where the remaining time actually is (profiled, not guessed)

Re-profiling after the ROWS tuning, per batched step:

| kernel | ms/step | share |
|---|---|---|
| `nvfp4_gemv_batch` | 202.4 | 53% |
| `fp8_gemv_batch` | 104.3 | 27% |
| `gated_delta_rule_step_multi` | 27.8 | 7% |
| `bf16_gemv_batch` | 8.3 | 2% |
| `attn_decode_multi` | 1.7 | <1% |

The Gated DeltaNet recurrence was **not** the bottleneck -- it is 7%. The GEMV
is 80%, and it runs at 17.6 GB / 380 ms = **46 GB/s, only 20% of the 228 GB/s
roofline**, where the single-sequence kernel reaches 76%.

### It is a register/occupancy problem, not a traffic problem

`ptxas -Xptxas -v` on the batch kernels:

| variant | registers | stack frame |
|---|---|---|
| single-sequence `nvfp4_gemv_kernel` | 40 | 0 |
| batch, ROWS=4, `lo`/`hi` materialised | 128 | 256 B |
| batch, ROWS=4, weights kept packed | **210** | 256 B |
| batch, ROWS=1 + shared x staging | 40 | 0 |

At ROWS=4 the kernel needs `acc[4][16]` + `lo[4][8]` + `hi[4][8]` = 128 live
floats, which exactly exhausts the register file: ptxas spills 256 bytes to
local memory and occupancy falls to 2 blocks/SM. Keeping the weights packed and
re-running the dequant per sequence was *worse* (210 registers -- the compiler
hoists more aggressively without the arrays).

The x-vectors are only 328 KB for batch 16 and stay L2-resident, so x traffic
is **not** the limiter: staging x in `__shared__` (with float4 loads, at 40
registers) still measured 582 ms, worse than 448 ms, because the two
`__syncthreads()` per k-tile are paid against only 8 rows of work per block.

So the remaining fix is to give each block many more rows of work per staged
x-tile -- roughly 64-128 rows -- with the per-row partial sums held in
`__shared__` rather than registers, since 64 rows x 16 sequences cannot fit in
registers at all. Shared budget: `xs[16][512]` = 32 KB plus `part[64][16]` =
4 KB = 36 KB, inside the 48 KB static limit.

### The round-37 failure, explained (round 39)

The shared-partial kernel was **wrong, and the cause was a real semantic trap in
the NVFP4 GEMV**:

```cuda
const int so = i * kWarp + lane;                        // note: depends on lane
sc[r] = e4m3_to_float(wscale[row * scalerow + so]);
acc[r] = fmaf(t, sc[r], acc[r]);                        // scale BEFORE reduce
...
const float a = warp_reduce_sum(acc[r]);                // reduce afterwards
```

`wscale[row][i*kWarp + lane]` is the block-16 scale covering exactly the 16
elements that *lane* owns, so **every lane has a different scale** and it must
be applied before the warp reduction. The shared-partial version wrote
`warp_reduce_sum(t) * sc`, which scales the summed dot by lane 0's scale alone
-- hence 0/16 sequences exact, first mismatch at token 1.

With that corrected to `warp_reduce_sum(t * sc)` the kernel passes both the
isolation harness and the full 16-sequence gate, so the diagnosis is confirmed.

### ...but the design is slower anyway, so it is not shipped

| RB | ms/step | aggregate tok/s | correct |
|---|---|---|---|
| 16 | 662.3 | 24.16 | yes |
| 32 | 637.5 | 25.10 | yes |
| 64 | 620.1 | 25.80 | yes |

Against 377 ms for the ROWS=4 kernel. The reason is structural, not tuning:
holding partial sums in shared forces a `warp_reduce_sum` per
(row, sequence, k-tile) -- 64 x 16 x 10 = 10240 per block -- whereas the
ROWS=4 kernel accumulates over the whole of K in registers and reduces **once**
per (row, sequence), 64 per warp. That is ~160x more reductions, and it costs
more than the shared-staging saves.

So the register pressure at ROWS=4 (128 registers, 256-byte spill) is the
cheaper problem to have, and the shipped kernel stays as it was:
**40.84-42.44 tok/s, 16/16 exact, ~377-392 ms/step** (run-to-run spread).

Reverted to the ROWS=4 kernels, which remain the verified state at **42.44
tok/s, 16/16 exact, 377.0 ms/step**.

### The isolation harness now exists (round 38)

`gb10-bench gemv-parity` gained a batched case: it builds a 16-row activation
matrix whose slots differ from each other (so a wrong per-sequence stride cannot
read equivalent data and pass) and checks **every row** of the batched result
against the independent CPU reference.

```
batched batch=1   n=256 k=5120 b=1   err/scale=1.286e-6  OK
batched batch=16  n=256 k=5120 b=16  err/scale=1.572e-6  OK
```

So the shipped ROWS=4 batch kernel is numerically sound at batch 16, and the
1e-4 tolerance is a real gate rather than a rubber stamp.

This is the tool the failed round-37 design needed: re-applying it and running
this command names the offending sequence and row directly, instead of
reasoning about the kernel. That is the next step.

## Optimisation order (roofline-driven)

1. **Reach the bandwidth roofline.** At batch 1 every kernel must be a
   streaming GEMV that saturates ~230 GB/s. *Done for the three weight formats:
   the M1 benchmark measures 234.9 – 249.9 GB/s over the real 17.56 GB weight
   set, projecting 13.4 – 14.2 tok/s against a 12.95 tok/s conservative
   roofline.*
2. **MTP speculative decoding.** Reduce bytes-per-token by accepting multiple
   tokens per weight read. This is the only lever that beats the roofline.
3. **Prefill/TTFT.** Chunked prefill, FP8/NVFP4 tensor-core GEMM, prefix
   caching. This is where we beat Ollama decisively.
4. **Continuous batching + paged KV.** Aggregate throughput at 16 concurrent.

## Verification gates

Every round runs `loop/run_round.sh`, which fails the round unless:

- `cargo build --release` succeeds,
- `cargo test` passes,
- the correctness gate passes against the frozen reference traces,
- any perf claim is backed by a fresh benchmark artifact under `bench/results/`.

Numbers are only ever reported from artifacts, never from estimates.
