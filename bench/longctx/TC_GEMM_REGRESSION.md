# The tensor-core prefill GEMM regression (rounds 282 → 408)

How a 2.48x speedup silently broke the engine for 126 rounds, how it was found,
what was actually wrong, and what now prevents it from happening again.

## Symptom

Long-context retrieval stopped working, and then every long prompt stopped
working:

- The 32K needle test that had previously scored 3/3 no longer reproduced.
- Through the server, **any** prompt over roughly 970 tokens returned an empty
  answer, with `completion_tokens: 0` and `finish_reason: "stop"` — zero tokens
  generated, immediate EOS as the very first sampled token.
- At longer lengths the model sometimes emitted unrelated garbage instead
  (Python-source fragments with line numbers, digit runs), always nearly
  identical text across unrelated prompts, which is what an EOS-adjacent
  near-tie looks like when it lands elsewhere.

It was initially misattributed to quantization, to the model, and to the
harness — including one wrong conclusion (mine) that "llama.cpp fails
identically, so the engine is exonerated". That was a bug in the probe: it read
only `content`, while llama.cpp routes text to `reasoning_content`. llama.cpp
answers every one of these prompts correctly.

## Finding it

`git bisect run` between `1358b7c` (round 251, known good) and `aa91bb7` (HEAD)
isolated a single commit:

**`336d91d` (round 282)** — `crates/gb10-model/src/weights.rs` only, flipping the
bf16 tensor-core prefill GEMM from opt-in to on-by-default.

Same binary, one environment variable, causal:

| configuration | fixed 5-prompt long-context battery |
|---|---|
| HEAD `GB10_TC_GEMM=1` (the committed default) | **0 / 5** |
| HEAD `GB10_TC_GEMM=0` | **5 / 5** |
| round 251 `1358b7c` (before the flip) | **5 / 5** |

The failure is length-driven, not content-driven (repeated filler, natural
wiki text, and varied filler all fail; no haystack always works), and the cliff
is one filler block wide: 939 tokens answers, 970 tokens returns nothing.

## Root cause

The tensor-core GEMM computed its operands in bf16. bf16 has 8 mantissa bits.
Casting a full-precision activation to bf16 perturbs it by ~0.2%, and those
perturbations feed a 48-layer Gated-DeltaNet recurrence whose state compounds
them with sequence length.

The clean server-free reproduction, `gb10-verify generate` on one fixed
958-token prompt:

| prompt form | fp32 CUDA-core | tensor-core |
|---|---|---|
| raw, no chat template | `[271, 14556]` = `"\n\nhello"` | `[271, 14556]` — identical |
| with chat template | `[1596, 1144, 4087, ...]` = `"We need answer user's request..."` | `[]` — EOS as the first token |

So it is a genuine near-tie at the first generated position, and the chat
template's thinking block is what puts it on the knife edge. Reduced-precision
operands land on the wrong side of it. One flipped argmax is the entire
long-context failure.

### It is not a rounding detail to be tuned

Fixing the obvious half of the precision loss was not enough, and neither was
fixing both halves:

1. **fp32 output.** The GEMM originally rounded its fp32 accumulator to bf16
   before converting back to fp32, discarding precision cuBLAS had already
   computed. Rewritten to have `cublasGemmEx` write fp32 straight into the
   output buffer (and to apply the NVFP4 per-tensor scale with a separate
   tiny kernel). Still 0/5.
2. **fp16 operands.** Both operands moved to fp16 — 10 mantissa bits, at
   identical tensor-core throughput, and still lossless for the 4-bit NVFP4 and
   FP8 weights. Still 0/5.

Ten mantissa bits are not enough. The required precision is above that, so the
reduced-precision approach is a dead end for this model rather than something to
tune, and the fp32 CUDA-core GEMM stays the correct path.

Note that mixed bf16 weights with fp16 activations does **not** work: this cuBLAS
rejects that pair with a buffer-size error even though the documented rules
allow differing `Atype`/`Btype` under `CUBLAS_COMPUTE_32F`.

## What changed

- `forward_prefill_tensor_core` is opt-in again (`GB10_TC_GEMM=1`), off by
  default, with the measurements above recorded at the decision site.
- The fp32-output GEMM and the fp16 staging kernels are kept, because they are
  strictly more accurate than what was there and they document the search. They
  are unreachable unless the experiment is deliberately re-enabled.
- **A new gate: `bench/longctx/longctx_gate.sh`, wired into `loop/run_round.sh`
  as `longctx-follow`.** It is deliberately the opposite of every other gate: a
  994-token prompt through the real chat template, asserting only that the model
  generates *something*. Validated both directions — it exits 0 on the fixed
  default and exits 1 (immediate EOS) with `GB10_TC_GEMM=1`.

## Why the gates missed it for 126 rounds

Every gate in the round loop used short prompts, and both plausible
precision-related signals were too weak to see this:

| gate | result with the bug present |
|---|---|
| `generate` 16/16 vs the frozen HF oracle | **passes** — the oracle prompt is 59 tokens |
| `batch-parity` 16/16 over 16 sequences | **passes** |
| perplexity (mean NLL) at a 512-token window | 0.023% worse (1.875052 → 1.875490) |
| `attn-tile`, `decode-bench`, prefix A/B, stream bench | **pass** — untouched by this |

A 0.023% perplexity move and a 16/16 short-prompt token match are not evidence
that a change is numerically safe. The only thing that caught it was a long
prompt through the real template.

## Open observation: the fp32 path is not deterministic

While validating the fix, `generate --repeat 8` failed 1 of 7 repeats on one
attempt (`6/7 extra repeats identical`) and passed fully on the next
(`7/7`, oracle 16/16). The differing run produced
`[1596, 1144, 4087, 4145, ...]` = `"We need answer user's request..."` where the
others produced the direct answer.

This is **pre-existing and not caused by the fix** — the fp32 path's code was
not touched by any of these changes — but it is a real reproducibility bug and
it is consistent with the near-tie diagnosis above. `GB10_KSPLIT 2` (split-K)
is the obvious first suspect, since an atomic accumulation order would explain
run-to-run variation exactly like this. Not yet investigated.

## Certification (the point of the fix)

One session, one harness, both engines, same prompts, after the fix:

| engine | prompt tokens | depths 10/50/90% | secs |
|---|---|---|---|
| gb10-engine (fixed) | 34,779 | **3/3 PASS** | 306.2 / 308.8 / 308.9 |
| llama.cpp (control) | 34,819 | **3/3 PASS** | 71.2 / 71.0 / 67.1 |

Transcript: `bench/longctx/results-needle-cert-32k.log`.

The control leg only passes once the harness is fixed. llama.cpp answers in
`reasoning_content` (the probe read only `content`, which is empty for llama.cpp)
and it needs a budget above 24 tokens, because it reasons before answering and 24
truncates it mid-thought. Both were wrong in the original harness, which is why
the control appeared to fail and why the engine was declared exonerated. A control
that fails is a reason to doubt the harness, not the subject.

## Cost

The 2.48x was bought with this bug, so it is gone: the 32 K class is ~307 s
against llama.cpp's ~70 s (~4.4x), where the scorecard recorded 1.80x with the
broken GEMM in place. The lever that can get it back without changing the
numerics is the `mma.sync` prefill attention, which keeps fp32 accumulation.

## Why the speed cannot simply be bought back from the GEMM

The obvious response to "the fp32 GEMM is 2.48x slower" is to find a
reduced-precision GEMM that is numerically good enough. Five operand formats
were tried, all behind the same opt-in flag, all measured on the same
958-token templated prompt:

| operands | effective mantissa bits | first generated token |
|---|---|---|
| bf16 (the original, default-on regression) | 8 | **EOS** — nothing generated |
| fp16, fp32 accumulator, fp32 output | 10 | **EOS** |
| bf16 hi+lo split (split-precision) | ~16 | **EOS** |
| bf16 hi+lo split, fp8 scale moved to the accumulator | ~16 | **EOS** |
| **fp32 CUDA core (the reference)** | **24** | `"We need answer user's request..."` |

Each step was a real improvement in accuracy and each one still flipped the
argmax. The fp8 fix in particular is a genuine bug that was fixed along the way:
the fp8 per-tensor scale is a single fp32 constant, and the tensor-core path was
folding it into the bf16 weights -- rounding every attention projection to 8
mantissa bits -- where the reference applies it to the fp32 accumulator. That is
now applied on the accumulator, like NVFP4's `s2`.

So the requirement is not a few more bits: 16 bits of activation precision is
still not enough where 24 works. That is consistent with the prompt sitting on a
genuine near-tie at the first generated position -- the fp32 path is *itself* not
deterministic here (1 of 7 repeats diverged once) -- and it means **no
reduced-precision GEMM is going to reproduce the fp32 path's behaviour on this
model.** The weights are already lossless in bf16 (4-bit NVFP4 / FP8), so the
loss is entirely in the activation, and the activation needs effectively full
precision.

That is why the remaining lever for cold TTFT is the `mma.sync` **prefill
attention**, not the GEMM: attention keeps an fp32 softmax and fp32 accumulation,
so it can be made much faster without touching the numerics that this model is
sensitive to. The tensor-core GEMM code is kept behind `GB10_TC_GEMM=1` as a
record of the search, not as a candidate.

## The three-way split, and what it rules out

A three-way bf16 split (`hi`/`mid`/`lo`, each rounding the remainder of the
previous part) carries ~24 mantissa bits, which is fp32's own significand width.
It was expected to be the fix. It is not:

| operands | effective mantissa bits | first generated token |
|---|---|---|
| bf16 | 8 | EOS |
| fp16 (fp32 accum, fp32 out) | 10 | EOS |
| bf16 two-way split | ~16 | EOS |
| bf16 two-way split + fp8 scale on the accumulator | ~16 | EOS |
| **bf16 three-way split** | **~24** | **EOS** |
| **fp32 CUDA core (reference)** | **24+** | `"We need answer user's request..."` |

At ~24 bits the activation operand is fp32-grade, so **this is no longer a
precision problem.** Something else systematically separates the two paths.

### What the reference path actually computes

Reading the fp32 prefill GEMM rather than assuming its precision:

* `kernels/gemm.cu:104,143` -- the **weights** are staged as bf16
  (`__bfloat16_as_ushort(__float2bfloat16_rn(...))`). That is lossless here:
  4-bit NVFP4 / FP8 has fewer mantissa bits than bf16 carries, so both paths
  agree on W exactly.
* `kernels/gemm.cu:232` -- the **activation** is only rounded to bf16 under
  `#ifdef GB10_SIM_BF16_ACT`, an explicitly-labelled diagnostic that is **off by
  default**. So the reference computes `bf16(W) x fp32(x)`, accumulated in fp32.

So the two paths should now agree to ~1e-7 relative, which is the same order as
the reference's own run-to-run variation. Yet fp32 is 5/5 on the long-prompt
battery and every tensor-core variant is 0/5, reproducibly, in one direction.

That is a contradiction, and it means one of two things: either there is a
systematic difference between the paths that has **not** been found (candidates
not yet excluded: the per-tensor scale value used for the fp8 layers, the
accumulation order interacting with a very tight argmax, or something in the
layernorm/DeltaNet numerics that only the tensor-core path perturbs), or the
decision margin at this prompt is below ~1e-7 and effectively arbitrary.

The fp32 path's own non-determinism (1 of 7 repeats diverged once) is evidence
for the second, but a 5/5-vs-0/5 split in a *consistent direction* is evidence
against it. **This is the open question, and it is now the blocker for reusing
the tensor-core GEMM at all.** It is recorded as unresolved rather than guessed.

What is *not* in doubt: the fp32 default is correct, and every reduced-precision
variant tried is not.

## The reframing: this is a bug, not a precision floor

The measurements above were read for a while as "this model needs more than 24 bits of
activation precision". That reading is wrong, and llama.cpp is the counter-example that
kills it:

**llama.cpp reaches 13.62 s at 8K and 57.23 s at 32K on this same model (NVFP4 GGUF) by
running tensor-core MMA over quantized weights.** Its activation operand is quantized to
roughly 8 bits with per-block scales -- *lower* nominal precision than the bf16 path that
fails here -- and it answers correctly at 34.8K, 3/3, while the engine's bf16 path emits
EOS. So:

* a fast tensor-core path that is accurate enough for this model **demonstrably exists**;
* the engine's tensor-core path is **wrong for a reason that has not been identified**;
* and "16 bits is not enough, 24 works" was measuring a symptom, not the cause.

That makes the tensor-core GEMM a **debugging target rather than a dead end**, which
matters because the measured scaling (see the scorecard in `comparison.md`) shows the cold
TTFT deficit at 8K/32K is dominated by exactly this O(T) work -- attention is only 7% of
8K and 23% of 32K prefill, so faster attention alone cannot close a 4.71x/5.40x gap.

Candidates not yet excluded, in the order worth testing:

1. **The per-tensor scale value used for the fp8 layers.** The reference reads `s1` inside
   the kernel (`__ldg(s1)`); the tensor-core path passes `LinearData::Fp8.scale`. Nothing
   has verified that these are the same quantity rather than two similarly-named ones.
2. **The `s2` path for NVFP4** -- same question, applied to the accumulator after the GEMM.
3. **A shape- or layer-dependent defect.** The tensor-core path is gated on `n >= 256 &&
   t > 16`; if some layer that the reference handles per-tile is mishandled at larger `t`,
   it would look exactly like this (fine on the 59-token oracle, broken past ~970 tokens).
4. **Accumulation order interacting with a genuinely tight argmax.** This is the
   explanation the data currently favours *least*, because the failure is 5/5 vs 0/5 in a
   consistent direction rather than scattered -- but the fp32 path's own non-determinism
   (1 of 7 repeats diverged once) means it cannot be dismissed.

A direct numerical comparison of the two paths' GEMM output on a real layer -- not a
48x32x17 integer fixture -- is the tool that is missing, and is the first thing to build.

---

# ROOT CAUSE FOUND: a truncated grid, not precision

Everything above that treats this as a precision floor is superseded. The defect was
found by building the tool that was missing -- `gb10-bench tc-parity`, which runs BOTH
real prefill paths over the same activation on real model weights and reports the error
of the tensor-core path against the fp32 reference.

## The measurement that found it

`tc-parity` at t=1024, per-matrix `rel_rms` of the tensor-core path against fp32:

| matrix | n | k | before the fix | after |
|---|---|---|---|---|
| `layers.0.mlp.gate_proj` (nvfp4) | 17408 | 5120 | **1.545e3** | 1.300e-6 |
| `layers.0.mlp.down_proj` (nvfp4) | 5120 | 17408 | **2.426e-1** | 5.278e-6 |
| `layers.3.self_attn.o_proj` (fp8) | 5120 | 6144 | 2.019e-3 | 2.019e-3 |

`mlp.gate_proj` was **1500x too large**. That is not a rounding error, and it cannot be
fixed by adding mantissa bits -- which is exactly what five precision rewrites had been
trying to do.

The `t`-dependence was the tell:

| t | gate_proj rel_rms | down_proj rel_rms |
|---|---|---|
| 64 | 1.3e-6 | 5.3e-6 |
| 512 | 1.3e-6 | 5.3e-6 |
| **1024** | **1.5e3** | **2.4e-1** |

Exact at t=512, destroyed at t=1024, on identical weights.

## The defect

The element-wise staging/scale kernels -- `f32_scale`, `f32_to_bf16`, `f32_to_f16`,
`f32_split_bf16`, `f32_split3_bf16`, `u16_to_bf16`, `u16_to_f16`,
`bf16_to_f32_scaled` -- indexed as

```cuda
const int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) ...
```

with **no grid-stride loop**, while the host capped the launch at

```rust
let grid = cdiv(n, 256).min(65535) as u32;
```

65535 blocks x 256 threads = **16,776,960 elements**. Everything past that index was
silently never processed.

For `mlp.gate_proj`, `n = 17408`, so the activation buffer has `t * 17408` elements and
crosses 16,776,960 at

```
t = 16_776_960 / 17408 = 963.6
```

**The engine answered 950-token prompts and emitted EOS from 964.** The cliff that had
been attributed to "the argmax is numerically ill-conditioned beyond ~970 tokens" was an
integer division. It is deterministic precisely *because* it is not a numerical effect.

Two things made this look like precision:

* the skipped tail was the region the per-tensor `s2` scale had never been applied to, so
  the symptom was "values at the wrong magnitude", which reads as an arithmetic-precision
  failure;
* the fp8 path *did* show a genuine ~2e-3 difference, which is real (see below) and
  supplied a plausible story that the whole thing was about mantissa bits.

`.min(65535)` is not a hardware limit either: for a 1-D grid `gridDim.x` may reach 2^31-1.
The cap looks like it was carried over from the 2-D convention, where `gridDim.y/z` are
indeed limited to 65535.

## The fix

All eight kernels became grid-stride:

```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
     i += gridDim.x * blockDim.x) ...
```

so their correctness no longer depends on the grid size at all, rather than merely raising
a magic constant.

## Why the gates missed it for so long

Every gate ran at a prompt length below the threshold, and the threshold is a *count of
elements*, not a context length. `generate` 16/16 uses a 59-token prompt; `batch-parity`
16/16 likewise; perplexity uses a 512-token window. `512 * 17408 = 8.9M < 16.78M`, so the
broken path was **mathematically exact** on every gate the project had -- 1.3e-6 agreement
at t=512, verified above. Perplexity moving 0.023% was not a fuzzy warning sign; at that
length there was nothing to warn about.

This is the same lesson as the original regression, one level deeper: `longctx-follow`
was added because a short-prompt token match is not evidence a change is numerically
safe. It was necessary but **not sufficient** -- it tested one 994-token prompt, which
caught the symptom while hiding that the cause was a buffer-size cliff rather than a
context-length cliff. `tc-parity` closes that gap by testing the arithmetic directly at a
length past the cliff, with no prompt, no chat template, and no sampling in the way.

## Consequence: the tensor-core GEMM is usable

With the fix, `GB10_TC_GEMM` is **on by default** (`GB10_TC_GEMM=0` opts out to the fp32
reference), and all of these pass on the default path:

| gate | result |
|---|---|
| `tc-parity` (t=1024, real weights) | within **1.3e-6** of fp32 on nvfp4 |
| `longctx-follow` (994-token prompt) | **OK** |
| long-context battery | **5/5 HELLO** |
| `generate` vs frozen HF oracle | **16/16 exact**, determinism 3/3 |

And the three-way split is no longer needed. It only ever existed to add mantissa bits to
a computation that was not short of mantissa bits. `GB10_TC_SPLIT` now selects
1/2/3 bf16 parts (default **1**, one GEMM), and one bf16 operand passes every gate above.
The measured cost of the extra parts is why this matters: three GEMMs where one will do is
a 3x regression on the dominant term of prefill.

The residual **fp8** difference (~2e-3) is a separate and benign finding, not this bug: the
fp32 reference rounds `e4m3(w) * wscale` into a **bf16 staged weight** (`kernels/gemm.cu:143`),
losing the scale's low bits, whereas the tensor-core path applies the scale in fp32 on the
accumulator. The tensor-core path is the *more* accurate of the two there; the 2e-3 is the
reference's rounding, and it is above the `tc-parity` noise floor because it is real.

## The lesson worth keeping

The failure mode this whole episode shares is **treating a measured effect as an
attributed cause**. "8 bits fail, 10 fail, 16 fail, 24 work" was a real measurement. "The
model therefore needs 24 bits of activation precision" was an assumption, and it was wrong
on the first test that compared the two paths directly instead of through a prompt. Five
precision rewrites were spent on it, and the actual defect -- 40 lines away, in the
launch geometry -- was never in the numerics at all.
