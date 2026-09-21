# SESSION HANDOFF (read this first)

## Where this stands, as of round 140

**One kernel is the entire remaining decode shortfall.** Everything else is measured
and closed:

| path | bytes/step | time/step | GB/s | its pattern's ceiling | gap |
|---|---|---|---|---|---|
| **`nvfp4_gemv_kernel`** | **9.19 GB** | 52.7 ms | **174** | **272** | **1.56x** |
| `fp8_gemv_kernel` | 8.42 GB | 36.2 ms | **232** | -- | **none (at peak)** |
| delta-rule + rmsnorm | small | ~15 ms | -- | -- | -- |

`bench/hw/gemv_bw.cu` streams the real NVFP4 layout with the GEMV's **own access
pattern** and reaches **~272 GB/s** (261.9/279.4/273.7 over 3 runs). So the memory
system is not the limit: **the 1.56x is inside `nvfp4_gemv_kernel`.**

**T1** (`docs/TARGETS.md`) is single-stream decode >= 12.5 tok/s. Current: **9.66**.
The original 100 tok/s target was shown physically unreachable before implementation
began and the owner accepted hardware-feasible targets; the roofline is 12.95 tok/s.

## THE GAP IS THE DEQUANT PATH -- ablation closed in round 141

`bench/hw/gemv_bw.cu`'s `LEVEL` ablation, extended to call the kernel's **own**
`e2m1x8_to_float` and `e4m3_to_float` (included straight from
`kernels/gemv_common.cuh`) instead of hand-written placeholders:

| level | work | GB/s |
|---|---|---|
| 6 | scale multiply, **no scale load** | **262.8** |
| **7** | **the real dequant path** | **171.0 / 207.5 / 178.9 -> mean 186** |
| -- | the real `nvfp4_gemv_kernel` | **190** |

**The synthetic reproduction now lands on the real kernel (186 against 190), so the
ablation accounts for the entire gap and closes.** And it isolates the cause in one
step: **the dequant path costs ~29%**, against 262.8 GB/s when the multiply is kept
but the unpack is removed.

**So the shortfall is `e2m1x8_to_float` + `e4m3_to_float` and nothing else** -- not the
access pattern (272 GB/s measured), not the x loads, not occupancy, not the scale
*load* (level 6 proves the multiply alone is nearly free).

**This also corrects round 130's first-order check.** That check assumed ~33 ops per
8 bytes and concluded ALU sat at 7.5% of capacity -- but `e2m1x8_to_float` is a
per-element bit-manipulation chain, not 8 ops per call. **The cost is the dequant's
*dependency chain*, and throughput arithmetic cannot see a dependency chain.** That is
why a count-based check passed while the measurement says 29% -- the same failure mode
as rounds 59/101/104/106, now for the fourth time.

**Checked the first two levers by reading the code (round 142), and both are dead:**

`e2m1_to_float` is **already branchless** -- `>>`, `&`, select, `<<`, `|` -- about 6 ALU
ops per element, so `e2m1x8_to_float` is ~96 ops per 8 bytes (12 ops/byte, ~22% of the
128/cycle capability at 2.33 bytes/cycle/SM). **A shared-memory or `__constant__` table
would replace those 6 ops with one load, but the index is a 4-bit nibble: only 16
distinct values across 32 lanes**, so a shared table suffers heavy bank conflicts and a
constant table serializes on divergent indices. **Both are likely worse, not better.**
The 256-entry `prmt` variant has the same distribution problem with 8x the footprint.

**So the decode is already near its floor in op count**, and the remaining 29% has to
be dependency structure -- the 16 elements are independent, so the ILP exists and the
question is whether the compiler is extracting it.

**The ILP test closes it (round 143), and the answer is register pressure.**

Measured in the same ablation, three runs each:

| level | work | GB/s |
|---|---|---|
| 6 | scale multiply + FMA, **no unpack** | **262.8** |
| 9 | unpack **one half** only | ~234 |
| 8 | unpack **both halves**, no scale/FMA | ~231 |
| **7** | **unpack + scale + FMA** | **~184** |

**The unpack is not the bottleneck** -- halving it (level 9, ~234) changes nothing
against doing it fully (level 8, ~231). **But the costs are super-additive:** unpack
alone ~231, scale+FMA alone 262.8, and **together 184 -- worse than either part
alone.** That is not a throughput limit in any single step; **it is the signature of
register pressure and scheduling.**

**The mechanism is visible in the source:** `e2m1x8_to_float(packed.x, lo)` and
`(packed.y, hi)` **keep 16 floats live** before the 16 FMAs consume them, while the
outer structure already holds `pk[ROWS]`, `sc[ROWS]` and the 16-float `XVec xv`. **The
compiler cannot keep all of that in flight and serialises.**

### The fused unpack is NEUTRAL -- register pressure is not it either (round 144)

Implemented exactly as below (unpack each nibble inline into its FMA, so `lo[8]`/`hi[8]`
never become live arrays):

| | round 132 build | fused unpack |
|---|---|---|
| registers | 48 | **45** (fell as predicted) |
| spills | 0 | 0 |
| `generate` | 16/16 | **16/16** |
| **t=1 (3 runs)** | 103.29 / 103.03 / 104.11 (**mean 103.48**) | 101.00 / 104.09 / 104.36 (**mean 103.15**) |

**-0.3%, ranges heavily overlapping. Reverted.**

**The register count fell exactly as the hypothesis predicted and the bandwidth still
did not move -- which is the cleanest possible falsification.** So register pressure is
not `nvfp4_gemv_tmpl`'s problem either.

**And that closes the last of the three levers derived from the ablation.** Levers 1 and
2 (shared-memory and `prmt` tables) died to reading the code; lever 3 died to
measurement.

**The methodological conclusion, which is the real result of rounds 141-144:**
`bench/hw/gemv_bw.cu` **reproduces the real kernel's bandwidth** (185.8 against 190) but
**its internal decomposition does not transfer to the real kernel.** Every step-by-step
explanation the ablation offers -- unpack, scale, FMA, and their super-additive
combination -- has now been tested *in the real kernel*, and none of them moves it.
**A synthetic kernel that matches on the endpoint but not on the derivative is not a
model of the kernel, and its per-level numbers should not be used to choose edits.**
The one thing it established that survives is the endpoint comparison itself: the
pattern sustains ~272 GB/s and the kernel reaches 190.

**The original fix note, kept as the record of a tested-and-rejected edit:** -- unpack one element and apply
its FMA immediately, so `lo[8]`/`hi[8]` never become live arrays. **That trades 16 live
registers for a shorter dependency chain**, and the level-8-vs-7 gap says the registers
are what is costing.

**The remaining lever, if any:**

* a shared-memory lookup table (16 entries, 64 B) replacing per-element bit
  manipulation with one shared load;
* a two-element `prmt`-based table (256 entries);
* restructuring so the 16-element chain is independent across its two halves.

**Test with `forward-cost` t=1, >= 3 runs, mean against 103.48 ms** (the round-132
build), and the gate must stay 16/16.

### The endpoint arithmetic closes exactly, and the prefill is bound by neither roofline (round 146)

| | tok/s |
|---|---|
| endpoint measured (256 tokens / 12.909 s) | **19.83** |
| the same, if prefill were free (256 / 5.36 s) | **47.76** |
| `batch-parity` B=16 engine result | **47.78** |

**The endpoint loses nothing to scheduling or overhead -- it is exactly the prefill,
and the decode half already runs at the engine's own B=16 rate.**

**And that prefill is bound by neither roofline.** A 59-token prefill is one 64-wide tile
pass, so it moves the full weight set and does the full arithmetic:

| | value | as a rate | against peak |
|---|---|---|---|
| bandwidth | 17.608 GB in 434 ms | **40.6 GB/s** | **18%** of 228 GB/s |
| compute | 27.3 G params x 59 x 2 = 3.2 TFLOP in 434 ms | **7.4 TFLOPS** | far below any FP16/FP8 peak |

**A kernel that is at 18% of the bandwidth roofline and a small fraction of the compute
roofline is bound by neither -- which is the signature of a latency/occupancy-bound
kernel**, not a bandwidth-bound one. That is a *different* diagnosis from the decode
GEMV's, and it is the first time the prefill GEMM has been placed between the two
rooflines rather than simply called "32 GB/s".

**And the "neither roofline" has a measured cause: the prefill GEMM runs at 47% occupancy.**

| | value |
|---|---|
| launch | `grid.x = cdiv(17408, GB10_NR=64)` = **272 blocks**, 128 threads |
| blocks/SM | 272 / 48 = **5.7** |
| threads/SM | 725 of 1536 = **47%** |
| register limit | 97 x 128 = 12,416 regs/block; 65536 / 12416 = **5.28 blocks/SM** |
| shared limit | 24576 B/block; 102400 / 24576 = **4.2 blocks/SM** |

**Both bind at once: the register budget allows 5.3 blocks/SM and the grid only offers
5.7.** So the kernel is latency-bound with no spare parallel work to schedule -- which is
exactly what "18% of the bandwidth roofline and 7.4 TFLOPS" looks like.

**The important consequence: raising the blocks/SM *capacity* cannot help** -- e.g.
`__launch_bounds__` to force 6 blocks would find no sixth block to run, because the grid
has only 272. **More parallelism requires more blocks**, which means either a smaller
n-tile (tied to `GB10_TT`=64 by the tile mapping) or a K split (which needs a reduction
across blocks). **Neither is a one-line change**, and both are shape changes of the kind
round 115's tile work already showed to be the only moves that pay on this engine.

### The n-tile path is closed too -- the K split is the only one left (round 148)

Checked before proposing it: `crates/gb10-cuda/src/ops.rs:57` documents
`GB10_NR = 64` as **"matches `GB10_TN` in `kernels/gemm.cu`"**. So **grid.x cannot be
doubled by shrinking `GB10_NR` alone** -- NR and TN move together, and rounds 96-119
already settled the tile at **TM=8/TNREG=4 with TN=64**, the only combination that ever
passed the gate (TN=64 with TNREG=4 failed the mapping twice, rounds 90 and 96).

**That leaves exactly one source of prefill parallelism: splitting K across blocks**,
which needs a cross-block reduction (atomics, or a second pass over the accumulators).
**A real feature, not a constant** -- and the honest next step, because it is the same
lesson the tile work already taught: only shape changes pay on this engine, and the
cheap shapes are exhausted.

**The magnitude is knowable in advance:**

| | now | 2-way K split |
|---|---|---|
| blocks | 272 (5.7/SM) | **544 (11.3/SM)** |
| per-block K work | full | halved |
| occupancy, if blocks bind (they do: 5.7 offered vs 5.3 cap) | **47%** | up to **~85%** |

**So the ceiling is roughly 2x on the prefill: TTFT 434 -> ~220 ms, and the endpoint
19.83 -> ~35-40 tok/s.** That is the one remaining path to the endpoint's >= 30 tok/s
target, and it is a bounded, well-specified feature rather than another constant to
try.

**Also checkable:** 97 registers and 24,576 B shared have not been varied since round
115, and the shared limit (4.2 blocks/SM) is *below* the register limit (5.3) -- so if
shared usage could drop under 20,480 B, the register limit would become the only one.
Neither is the binding constraint against a 5.7-block grid, which is why this is
recorded as a diagnosis and not presented as a fix.

### The K split is specified and ready (round 149)

`loop/patches/ksplit.md` has the complete edit list: `GB10_KSPLIT 2`, the `kc0..kc1`
chunk range, `atomicAdd` at the store (line 311), `grid_dim.2 = 2`, and a
`cudaMemsetAsync` of the 4.1 MB output that `atomicAdd` requires.

**It also records the risk that could kill it, before anyone implements it:** with
`TM x TNREG = 32` accumulators per thread x 128 threads x 544 blocks, the split issues
**~2.2M atomics per matrix** against 1.03M output elements -- roughly one atomic per
output element, across 192 matrices per prefill. **If atomic throughput becomes the
bottleneck, the split loses**, and the fallback is a second reduction kernel.

**Acceptance is the gate plus >= 3 runs**, and if TTFT does not move the result is to be
recorded as the tenth prefill mechanism eliminated -- not quietly dropped.

## USER-VISIBLE ENDPOINT ISSUE: the reasoning block is returned as `content` (round 152)

Found by actually calling the endpoint rather than trusting its JSON shape. A 200-token
generation returns:

```
'User asks: "What is the capital of France? Answer in one short sentence." Need answer
one short sentence. Final: "The capital of France is Paris."\n</think>\n\nThe capital
of France is Paris.'
finish_reason: stop, 43 completion tokens
```

**The answer is correct** -- "The capital of France is Paris." -- so the engine and the
weights are fine. **But `content` is the raw reasoning block plus the answer, ending a
bare `</think>`.** The chat template evidently puts the opening ` thinking` into the
prompt, so the model's output legitimately starts mid-reasoning and closes the tag
itself, and the endpoint passes that through verbatim.

**Consequence: at low `max_tokens` the user sees only reasoning and never the answer.**
That is exactly what the earlier 16-token smoke test showed ("We need to respond to
user: ..."), and **the first assessment of it as "just truncation, not a defect" was too
generous** -- the truncation is real, but it is a defect *because* the reasoning block
occupies the visible `content` field.

**This is a presentation issue, not a numerics one**, and it is user-visible on the one
deliverable the objective names explicitly. **It is also entirely in Rust, so it carries
none of the kernel-change risk that `loop/patches/ksplit.md` does** -- which makes it the
better next task while the K split waits.

### FIXED in round 153 -- with the acceptance test, on both protocols

`visible()` was added to `crates/gb10-server/src/main.rs` and applied at both
non-streaming response sites (OpenAI `message.content`, Anthropic `content[].text`).
It strips everything through the closing `</think>` tag, and **returns the text
unchanged when there is no closing tag** -- so a truncated generation degrades to the
old behaviour rather than returning an empty answer.

Same prompt, 200 tokens, before and after:

| | |
|---|---|
| before | `'User asks: "..." Final: "The capital of France is Paris."\n</think>\n\nThe capital of France is Paris.'` |
| **after (OpenAI)** | **`'The capital of France is Paris.'`** |
| **after (Anthropic)** | **`'The capital of France is Paris.'`** |

Gate **16/16**, build 0 errors -- **the tokenizer path is untouched, only response
assembly.**

**Known remaining half: streaming.** The streaming path emits each piece as it is
produced, so it still shows the reasoning block. `visible()` cannot be applied there
as-is because the `</think>` boundary is not known until it arrives; the fix is to
suppress emitted pieces until the boundary is seen, which **changes
time-to-first-visible-token and must be measured as its own change rather than
smuggled into this one.**

### The streaming fix, designed (round 154) -- not implemented

`crates/gb10-server/src/main.rs:378` sends each piece from the generator callback:

```rust
let res = remote_generate(tx, &messages, max_tokens, thinking, |piece| {
    send(&json!({ ... "choices": [{"index": 0,
        "delta": {"content": piece}, "finish_reason": null}], }))
});
```

**A per-piece string match will not work**: `piece` is `tok.decode(&[next], true)`, and
`</think>` is several tokens, so the tag arrives split across pieces. The callback needs
to buffer until the boundary is *complete*:

```rust
let mut held = String::new();
let mut opened = false;          // has </think> been seen?
// inside the callback, per piece:
if opened { emit(piece) }
else {
    held.push_str(piece);
    if let Some(i) = held.find("</think") {
        if let Some(j) = held[i..].find('>') {
            opened = true;
            let rest = held[i + j + 1..].trim_start().to_string();
            held.clear();
            if !rest.is_empty() { emit(&rest) }
        }
    }
}
```

**Two things to get right:**

* **Borrow checker.** `send` is already a closure capturing `stream` mutably, and this
  state machine wants to be captured by the generator callback too. **The clean shape is
  to make the state machine a small struct with an `emit(&mut self, piece)` method and
  pass `send` into it, rather than nesting three closures.** Expect a fight if it is
  written inline.
* **An empty-answer fallback.** If the model never emits `</think>` (truncated mid
  reasoning), `held` is never flushed and **the stream delivers nothing at all** --
  strictly worse than the current behaviour. **On `finish`, flush `held` through
  `visible()`**, so a truncated run still shows its reasoning the way the
  non-streaming path does.

**Do the same at the Anthropic site** (`main.rs:483`, `event:`/`data:` framing).

### FIXED in round 157 -- by taking the emitter OUT of the gate

After two failed attempts, the second of which is documented below, the change landed
with one structural insight: **the gate must not hold the emitter at all.** `push` and
`flush` return `Option<String>` and the *caller* writes it:

```rust
struct ThinkGate { held: String, opened: bool }          // no callback, no lifetime
fn push(&mut self, piece: &str) -> Option<String>        // Some(text) to emit
fn flush(&mut self) -> Option<String>                    // whatever is still held
```

**That removes both problems at once.** With no `&mut F` field there is no `E0631`
(`send` is `FnMut(&Value)`, the gate wanted `FnMut(&str)`), and no adapter closure whose
lifetime has to end before the trailing finish chunks reuse `send`. **The callback simply
captures the gate and the writer as two independent variables, and no scope has to be
arranged.** Both failed attempts were fighting a problem the design was creating.

**Verified, both protocols, including the trap:**

| test | result |
|---|---|
| `generate --n 16` | **16/16** |
| stream 200 tok, OpenAI | `'\n\nThe capital of France is Paris.'` -- **reasoning gone** |
| **stream 8 tok, OpenAI** (truncated mid-reasoning) | `'User asks: "What is the capital'` -- **not empty** |
| stream 200 tok, Anthropic | `'\n\nThe capital of France is Paris.'` |

**The 8-token case is the one that matters:** without the `flush` fallback the stream
would have delivered *nothing*, which is exactly the regression the round-154 note
predicted. **The prediction was right, and testing it was the reason it did not ship.**

**Cosmetic, not fixed:** the answer begins with the `\n\n` that separated it from the
closing tag, because later pieces are passed through untouched and only the piece
containing `</think>` is trimmed. Stripping leading whitespace from the accumulation
until the first non-space character would fix it, but that needs its own state and is
not worth the risk for two newlines.

### The two failed attempts in round 156, kept as the record of the wrong design

Both attempts hit the borrow fight the note above predicts, and **both were reverted
cleanly with `git status` empty and `generate` still 16/16. Nothing about streaming
landed.**

The error to plan around is **`E0631: type mismatch in closure arguments`** at
`ThinkGate::new(&mut send)`: `send` is `|v: &Value| -> Result<()>` (line 411) while the
gate's bound is `F: FnMut(&str) -> Result<()>`. **The gate cannot wrap `send` directly.**

The shape that gets furthest is a separate `&str` adapter per site, *not* a modification
of `send`:

```rust
let mut emit_s = |t: &str| -> Result<()> { send(&json!({ ... "content": t ... })) };
{
    let mut gate = ThinkGate::new(&mut emit_s);
    let res = remote_generate(tx, &messages, max_tokens, thinking, |piece| gate.push(piece));
    gate.flush()?;
    match res { ... }
}
// `emit_s` must be OUT OF SCOPE before the finish chunks call `send` again.
```

**The scoping is the whole difficulty**, and it is why the second attempt died too: the
finish chunks after the generation block also call `send`, so `emit_s` and `gate` both
have to drop before that point, and the `match res` arm has to be closed at the right
place. **A patch to this region has to move the closing brace, which makes a
string-match edit fragile** -- the second attempt's anchor failed and wrote nothing,
leaving the tree mid-patch until it was restored from `/tmp/m155.rs`.

**Next attempt: restructure this function by hand rather than by patch script**, or
extract the whole streaming body into its own function so the gate's lifetime is a
function boundary rather than a brace to be moved. **Do not use a sed/python anchor for
the brace.**

**Acceptance:** `curl ... "stream": true` with the round-152 prompt and
`max_tokens: 200` must emit `The capital of France is Paris.` and **must not emit any
reasoning text**; with `max_tokens: 8` it must still deliver *something* rather than an
empty stream; `generate --n 16` stays 16/16.

**The original analysis and fix options, kept for the record:**

* **strip** everything up to and including `</think>` from `content` before returning;
* or **expose it separately** -- OpenAI's `reasoning_content`, Anthropic's `thinking`
  content block -- so the reasoning and the answer both survive and neither protocol's
  schema is violated.

**Acceptance:** the 200-token generation above must return `The capital of France is
Paris.` (or that plus a separate reasoning field), and `generate --n 16` must stay 16/16
-- the tokenizer path must not change, only the response assembly.

## CORRECTION TO THE CORRECTION (round 170): `pre_ms` DOES move the 17.608 GB

**Round 169 retracted the round-168 conclusion. That retraction was wrong, and reading the
harness is what shows it.**

`crates/gb10-verify/src/main.rs:1019-1028`:

```rust
let mut st2 = ModelState::new(&dev, &model, args.max_seq, 1)?;
let _ = model.prefill(&dev, &ids, &mut st2, &mut sc2)?;      // warm-up on `ids`
let feed: Vec<u32> = vec![next; t];                          // t fresh tokens
let t1 = std::time::Instant::now();
let _ = model.prefill_seq(&dev, &feed, &mut st2, &mut sc2, 0)?;
let pre_ms = t1.elapsed().as_secs_f64() * 1e3;
```

**`pre_ms` times `prefill_seq` over `t` tokens -- a real forward through all 64 layers,
reading every weight once, inside a single 64-wide tile for both `t = 8` and `t = 16`.**
So the traffic is **17.608 GB in both cases**, exactly as round 168 assumed:

| t | `pre_ms` (mean of 3) | weight traffic | effective |
|---|---|---|---|
| 8 | **176.77 ms** | 17.608 GB | **99.6 GB/s** |
| 16 | **298.27 ms** | 17.608 GB | **59.0 GB/s** |

**Identical traffic, 1.69x the time. The prefill is therefore NOT weight-bandwidth-bound --
it is bound by per-token work.** That conclusion stands, and the round-169 note is void.

**And how it was broken is the lesson.** I inferred what a column meant from its *value*
instead of from *the code that produced it*, then used that inference to retract a correct
result. **That is the same error class as rounds 152 and 155 -- and this time I committed it
while explicitly recording that error class.** One `sed -n` on the harness was all that was
needed, before either conclusion.

## The open question, now sharpened

With the traffic attribution restored: the prefill moves 17.608 GB in 176.77 ms
(**99.6 GB/s**) while doing 0.437 TFLOP (**2.47 TFLOPS**). **The closed form says a
0.75 B/FMA tile at 99.6 GB/s can only sustain 0.27 TFLOPS -- a ~9x disagreement.** These
cannot both be true, so one input is wrong:

1. **`B/FMA` is not 0.75 for this kernel**, or
2. **the FLOP/param count is low by ~9x**, or
3. **`prefill_seq` reads more than one copy of the weights** (e.g. the MTP layer, or a
   second pass).

### Counted in the source (round 171): the B/FMA term is fine

`kernels/gemm.cu:262` `gemm2d_outer_bf16`, per thread per `k` (32 iterations):

| | |
|---|---|
| loads | one `uint4` (16 B = 8 bf16 weights) + one `float4` (16 B = 4 activations) = **32 B** |
| arithmetic | 8 rows x 4 cols = **32 `fmaf`** + 4 `__bfloat1622float2` |
| ratio | **~1.0 B/FMA per thread** |

**The closed form assumed 0.75 and the code says ~1.0 -- close. The ~9x discrepancy is
therefore NOT in the B/FMA term**, and hypothesis 1 above is eliminated without any GPU
time. **It is in (a) the FLOP/param count, or (b) what `prefill_seq` actually moves.**

**One real amplification the model missed:** with `ty = threadIdx.x >> 4`, the 32 lanes of a
warp span **two** `ty` values, so the same `uint4` is fetched for two row groups per warp
per `k` instead of being broadcast. **That is a genuine 2x traffic factor -- but it is STRUCTURAL, not a fixable defect.** The
mapping must satisfy `ty_groups * tx_groups == GB10_GEMM_BLOCK`, and at this tile that is
`8 * 16 == 128`. **Getting one `ty` per warp needs `tx_groups = 32`, i.e. `TNREG = 2`,
which gives `ty_groups * tx_groups = 8 * 32 = 256 != 128` -- the mapping rule breaks --
and `acc = TM * TNREG = 16`, far below the `acc >= 64` the B/FMA ladder requires.**
**Buying back the 2x costs the tile, and the tile is what makes B/FMA ~1.0 instead of ~2.0.**

**So that lead closes too, and it closes the same way the recovery attempt did last round:
by checking the arithmetic against the rule, not by argument.**

**The prefill is now down to ONE open question: the unresolved factor of ~9 between the
measured 2.47 TFLOPS and the 0.27 TFLOPS the closed form allows at 99.6 GB/s.** With the
`B/FMA` term counted (~1.0) and the warp refetch explained, **that factor is most likely in
the FLOP/param accounting -- i.e. in one of my own constants, not in the kernel.** It also fits the standing
observation that the prefill sits at 18% of the bandwidth roofline and a small fraction of
the FMA roofline: **a 9x accounting error would explain exactly that shape.**

## Superseded: the round-169 note (kept because the error is instructive)

`forward-cost` prints `t | step_ms | pre_ms | ratio | ms/row`
(`crates/gb10-verify/src/main.rs:1030`). At t=8 that is **step 831.62 ms, pre 177.50 ms**.

**I attributed 17.608 GB -- the weight traffic of one FULL forward over all 64 layers -- to
`pre_ms`, and built a "the prefill is compute-bound, not weight-bound, because doubling
rows costs 1.69x while the weights are constant" argument on it. That argument is void:
the 17.6 GB belongs to `step_ms`, not to `pre_ms`.** The "9x disagreement" between the
B/FMA model and the measured FLOPs was an artifact of the mis-attribution.

**Nothing about the kernel was learned from it.** Recorded because this is the same error
class as rounds 152 (a valid JSON shape read as a valid answer) and 155 (a stale README
read as current): **a number that parses is not a number that means what you assumed.**

**What does survive from rounds 168-169:**

* **The K split is correct** -- 16/16 on three consecutive runs, `chunked-prefill --n 6`
  OK -- **and it is not faster**: `pre_ms` 176.77 -> 176.07 at t=8 (neutral), and
  **298.27 -> 307.81 at t=16 (+3.2%, worse)**. **Reverted; not in the tree.**
* **The occupancy explanation is falsified.** The prefill GEMM genuinely runs at 47% with
  the register budget (5.3 blocks/SM) and the grid (5.7) binding at once -- measured in
  round 148 and still true -- **but doubling the block count to 544 bought nothing**,
  because that kernel's printed `nsplit=2`/`nchunk=160` confirmed the split was live.
  **That is the eleventh prefill mechanism eliminated.**

**The open question is now simply stated and unanswered: the prefill GEMM is at 18% of the
bandwidth roofline and a small fraction of the FMA roofline, and it is not short of
blocks. What is it waiting on?** The next instrument should be a **`clock64()` accumulation
inside the kernel** -- time spent in the staging vs the outer-product loop -- because every
hypothesis tested so far was tested from outside the kernel, and the two inside-kernel
facts established this session (`y.len()` = 16x, `gridDim.z` = 2) each settled a question
in one run.

## RESOLVED (round 173): there was never a ~9x error -- I compared a GEMV to a GEMM

`crates/gb10-model/src/weights.rs:95`:

```rust
if self.n < 256 || t <= 16 { return self.forward(dev, x, y, t); }
```

**At BOTH t=8 and t=16 the `pre_ms` path routes to the batched GEMV, not to
`forward_prefill`/the GEMM.** The 0.75 B/FMA closed form is a property of the **GEMM tile**.
The GEMV instead does **`t` FMAs per 2-byte weight** -- ~12 FMA/byte at t=8 -- so at
99.6 GB/s it predicts **2.47 TFLOPS**, exactly the measured value. **The arithmetic closes.
The "9x discrepancy" was a model applied to the wrong kernel.**

### And that reattributes the endpoint bottleneck

| path | rows | effective bandwidth |
|---|---|---|
| decode GEMV | 1 | **174-190 GB/s** |
| same GEMV, batched | 8 | **99.6 GB/s** |
| same GEMV, batched | 16 | **59.0 GB/s** |

**Batching the GEMV halves its effective bandwidth by 8 rows and thirds it by 16.** That is
the prefill/endpoint bottleneck -- **not the GEMM, and not occupancy.** The endpoint's
19.8 tok/s comes from this: **the same kernel that sustains 190 GB/s at one row sustains
59 GB/s at sixteen.**

**Which means rounds 148-172 spent twenty-five rounds on the wrong kernel.** The
occupancy work, the K split, and the tile discussion all target `forward_prefill`, and the
`pre_ms`/endpoint path at these sizes never enters it.

### CORRECTION (round 175): the "59.0 GB/s" figure must be withdrawn

**I read `kernels/gemv.cu:106-107`:**

```cuda
const int b = blockIdx.y;
const float* __restrict__ xb = x + (size_t)b * K;
```

`blockIdx.y` is the batch index, `xb` is per-batch, and the weight row loop
(`for (rbase = ...; rbase < N; rbase += row_stride)`) is indexed by `row` and **does not
depend on `b`**. Read naively that means every batch element walks all the weights.

**The arithmetic forbids it:**

| t | traffic if re-read per batch element | implied |
|---|---|---|
| 8 | 8 x 17.608 = 140.9 GB in 176.77 ms | **797 GB/s** |
| 16 | 16 x 17.608 = 281.7 GB in 298.27 ms | **945 GB/s** |

**Both exceed the measured 228 GB/s peak by 3.5-4x. So the weights are NOT re-read per
batch element, my reading of the grid is incomplete, and the "99.6 / 59.0 GB/s" figures
derived from that traffic model cannot be quoted until this is settled.**

**What is established:** `blockIdx.y` is batch, `xb` is per-batch, the weight access is
independent of `b`. **What is not:** how many times the weights are actually fetched.

**The check is the same instrument that settled `gridDim.z` for the GEMM in one run: print
`gridDim` and `blockIdx.y` from the GEMV, or read the host launch site's `grid_dim`.**
**Do that before any further reasoning about this kernel's bandwidth.**

### RESOLVED (round 176): there are TWO GEMV kernels, and batch>1 uses the other one

`crates/gb10-cuda/src/kernels.rs`:

| line | |
|---|---|
| 9 / 12 | `"nvfp4_gemv_kernel"`, `"nvfp4_gemv_batch_kernel"` |
| 33 / 34 | `nvfp4_gemv: CudaFunction`, `nvfp4_gemv_batch: CudaFunction` |
| 88 | `.launch_builder(&self.nvfp4_gemv_batch)` -- the **batch > 1** path |
| 96 | `let f = &self.nvfp4_gemv;` -- the **batch == 1** path |

**`nvfp4_gemv(...)` dispatches on batch size, and batch > 1 goes to a DIFFERENT kernel.**
`nvfp4_gemv_tmpl` -- the body I read in round 175 -- is the batch == 1 family, and its
`blockIdx.y`-as-batch structure is not the batched kernel's structure.

**So every bandwidth number I derived for "the batched GEMV" (99.6 GB/s at t=8, 59.0 GB/s at
t=16, and the 190 -> 59 decline) belongs to a kernel whose source has not been read.** It
also dissolves the round-175 arithmetic tension: the per-batch weight re-read I inferred
from `xb = x + b*K` was inferred from the *wrong kernel's* template.

**Next step, precisely: read `nvfp4_gemv_batch_kernel`'s body and its launch `grid_dim`.**
**That is where the endpoint's prefill cost actually lives, and no bandwidth claim about
the batched path is usable until it has been read.**

### RESOLVED (round 177): the batched GEMV is L2-bound on ACTIVATION re-reads, not weights

`kernels/gemv.cu:367` `nvfp4_gemv_batch_tmpl`, instantiated as
`nvfp4_gemv_batch_tmpl<4, GB10_BATCH_MAX>` (line 592). Loop order:

**`rbase` (row groups) -> `i` (k-tiles) -> `b` (batch, INNERMOST)**

```cuda
uint2 pk[ROWS]; float sc[ROWS];
for (int r = 0; r < ROWS; ++r) { ... pk[r] = ...w...; sc[r] = ...; }   // weights 1x
for (int b = 0; b < B; ++b) {
    const XVec xv = load_x(x + (size_t)b * K, e0);                     // activations per b
    ... acc[r][b] = fmaf(t, sc[r], ...) ...
}
```

**The weights are read ONCE per (row, k-tile) -- correct. The activations are re-read inside
the batch loop: 4352 row groups x 16 batch x 10 k-tiles = 696,320 `load_x` calls per GEMM.**

**The kernel's own header already states the tradeoff:**
> "With ROWS = 1 the x-vectors are re-read once per row, which at batch 16 is ~8x the weight
> traffic and makes the kernel L2-bound."

**And the register cost is what caps `ROWS`:** `acc[ROWS][BMAX]` = 64 floats plus
`lo/hi[ROWS][8]`x2 = 64 floats, so **ROWS=4/BMAX=16 already holds ~128 floats per thread and
ROWS=8 would need 256.** That is why round 127's `ROWS=2` at B=16 measured 12% worse.

### This resolves the whole endpoint question

**The batched GEMV is not weight-bandwidth-bound at all.** The weights still stream once
(17.608 GB); the time goes to **L2-resident activation re-reads**. **That is why its
"effective bandwidth", computed as weights/time, falls from 190 to 99.6 to 59 GB/s as batch
grows -- the extra time is not being spent on weights, so dividing weights by time measures
nothing about this kernel.** Every one of those three figures should be read as a *time*,
not a bandwidth.

**So the endpoint's prefill cost is an activation-reuse problem**, and the fix direction is
more `ROWS` (register-capped at 4) or a different blocking -- **not more bandwidth, and not
the GEMM.** The design is already at a deliberate, documented sweet spot, and the
`ROWS` knob has been searched (rounds 104/127).

### RETRACTION (round 179): the shared-memory proposal was already measured and rejected

**`docs/NEXT.md:3429-3440` already rules this out -- and does better than rule it out, it
names the correct fix.** For `nvfp4_gemv_kernel`:

> "54 GB/s is not traffic -- it is the load's presence in the dependency chain, i.e. **L2
> latency**." ... "**shared memory did not help** -- shared has its own latency, plus a
> `__syncwarp()`." ... "**The fix is therefore to break the dependency, not to move the
> data**: software-pipeline the scale load one k-tile ahead."

The supporting evidence recorded there is the signature to look for: **widening the load
helped only 3.3%**, while **ablation steps 4 and 5 *raised* bandwidth** (210.5, 212.5) by
adding work to overlap the latency.

**So round 178's proposal was not unexplored. The codebase paid for that experiment once and
it lost.**

### The transferable hypothesis, which is what to test instead

**`load_x` sits INSIDE the `b` loop and on the dependency chain of every FMA in that
iteration -- the same shape as the scale load.** If that is the real mechanism, then:

* **the fix is software-pipelining `load_x` one k-tile ahead**, not shared-memory staging;
* **the cheapest confirmatory test comes first**: add independent work per k-tile and see
  whether throughput *rises*. **A rise is the signature of a latency-bound load** -- it is
  what already distinguished latency from traffic in this exact kernel family, and it costs
  one measurement rather than a rewrite.

**Note the discipline this round applied: the ruled-out list was checked BEFORE spending a
round, not after.** That is the first time in this session a rejected mechanism was caught
in advance rather than by re-measuring it.

### The latency hypothesis is testable from data already in hand (round 180)

**If the batched GEMV is latency/dependency-bound, time is SUB-LINEAR in batch; if it is
traffic-bound, time is LINEAR.** The two points already measured:

| batch | `pre_ms` (mean of 3) |
|---|---|
| 8 | **176.77 ms** |
| 16 | **298.27 ms** |

**Ratio 1.687 for 2x the batch -- clearly sub-linear.** Fitting `T(B) = a + bB`:
**b = 15.19 ms per batch row, a = 55.27 ms** -- **31% of the t=8 time does not depend on
batch at all.**

**So the prediction is supported: there is a large fixed component, which is what a
dependency-chain stall looks like.**

**But the fit is not yet trustworthy, and the reason is decisive:** if the 17.608 GB of
weights stream once, then `a = 55.27 ms` implies **319 GB/s -- 40% above the measured
228 GB/s peak, which is impossible.** So one of three things is wrong: the linear model, the
17.608 GB traffic figure, or the mapping from batch size to the weight set.

**Next step -- NO kernel change required.** Sweep `pre_ms` at five or more batch sizes and
fit. **A curve that flattens is latency; a straight line through the origin is traffic.**
`forward_cost` reports only t=8 and t=16 today (`crates/gb10-verify/src/main.rs:1068`
dispatches to it), so the extra points need the sweep list in the harness extended -- **a
harness edit with no correctness risk, which is exactly why it should be done before any
kernel rewrite.**

### THE FULL CURVE (round 181) -- and it was already being printed

`crates/gb10-verify/src/main.rs:1008`: `for t in [1usize, 2, 4, 8, 16]`. **The harness has
always swept five batch sizes. Rounds 168-180 only ever read t=8 and t=16 because a `grep`
filter (`'^ +8 |^ +16 '`) hid the other three. No edit was needed.**

**Full sweep, mean of 3 runs** (effective = 17.608 GB / time):

| t | `pre_ms` | effective | scaling |
|---|---|---|---|
| 1 | **103.64** | **169.9 GB/s** | -- |
| 2 | **133.66** | 131.7 GB/s | x1.29 for 2x |
| 4 | **142.60** | 123.5 GB/s | **x1.07 for 2x** |
| 8 | **179.61** | 98.0 GB/s | x1.26 for 2x |
| 16 | **312.23** | 56.4 GB/s | **x1.74 for 2x** |

**Two regimes, with the boundary between t=4 and t=8:**

* **t=1 -> 4: +38% for 4x the batch -- nearly flat. The weight stream dominates.**
* **t=8 -> 16: +74% for 2x the batch -- nearly linear. Per-token work dominates.**

**So "latency or traffic" is answered "both, at different sizes". The high end is genuinely
linear in batch, which means the target is the marginal per-token cost -- 16.6 ms/token --
not a fixed stall. At 16 concurrent that term is 16 x 16.6 = 265 ms against a ~104 ms fixed
weight cost, so 72% of the endpoint's prefill time is per-token work.**

**t=1 achieves 169.9 GB/s = 75% of the 228 GB/s peak, matching the known single-row figure.
The batch kernel falls 132 -> 56 GB/s as batch goes 2 -> 16.**

**This is the first complete quantified profile of the endpoint's prefill cost. It says the
target is the batched kernel's per-token marginal cost (the activation re-read / dependency
chain of rounds 177/179), NOT the weight stream -- which is already at 75% of peak.**

**Methodological note worth keeping: three rounds reasoned about this curve before anyone
read the whole table, and the harness had been printing all five points the entire time.
The measurement was already taken; a filter hid it.**

### REFINEMENT (round 182): the marginal cost RISES -- so it is L2 capacity, not a stall

**Round 181 called the high end "nearly linear". It is actually SUPER-linear:**

| interval | delta `pre_ms` | ms per added token |
|---|---|---|
| 4 -> 8 | 37.01 | **9.25** |
| 8 -> 16 | 132.62 | **16.58** |

**The marginal cost nearly DOUBLES.** That distinction decides the fix:

* **a dependency-chain stall (round 179's hypothesis) would give a CONSTANT marginal cost;**
* **a rising one points at a progressively closing REUSE WINDOW -- L2 capacity** -- which is
  exactly what the kernel header warns about ("makes the kernel L2-bound").

**The working-set arithmetic supports L2 capacity:** the `x` slab is `B x kTile x 4 B` =
**8 KB at B=4, 16 KB at B=8, 32 KB at B=16**, and it is re-read once per row group, so the
live working set is `slab x resident blocks`. **The slab grows with B; L2 does not.**

### DEFINITIVE (round 183): the weight stream is FINE -- the target is per-token work

**`kernels/gemv.cu:360-364` states the design intent explicitly:**

> "Weight traffic must be independent of batch size, so these variants loop over the batch
> inside the block: **the weight tile is loaded once and applied to all B x-vectors.**"

**And the curve confirms the design works. Decomposing `pre_ms` into a fixed weight cost plus
per-token work:**

| t | `pre_ms` | minus fixed | per-token |
|---|---|---|---|
| 1 | 103.64 | 0 | -- |
| 2 | 133.66 | 30.02 | 30.0 (small-batch artefact) |
| 4 | 142.60 | 38.96 | **9.7** |
| 8 | 179.61 | 75.97 | **9.5** |
| 16 | 312.23 | 208.59 | **13.0** |

**Fixed weight cost: 17.608 GB in 103.64 ms = 170 GB/s = 75% of the 228 GB/s peak.**

**So the weight stream is already good, and "56.4 GB/s" was never a bandwidth measurement at
all: 312.23 ms = 103.6 ms of weight streaming + 208.6 ms of per-token work. Dividing 17.608 GB
by the total measures nothing about the weight stream. That figure and the "falls from 132 to
56 GB/s" framing are both withdrawn.**

**The remaining target is unambiguous: the per-token term, ~9.5-9.7 ms/token at t<=8 rising
to 13.0 at t=16. At 16 sequences that term alone is 208.6 of the 312.2 ms.**

**And `#define GB10_BATCH_MAX 16` is a hard template bound**, so a `ROWS=8`/batch-tile-8
variant needs an outer batch loop plus a second instantiation: **expressible, but not a
one-line edit -- which is why it should not be started without room to build, gate, and A/B it.**

### EXPERIMENT (round 184): the t<=16 cut is CORRECT -- the comment's table is stale

**`weights.rs:88-89` tabulates the two paths:**

```
1: 271.5 -> 116.7 ms   2: 269.4 -> 145.8   4: 280.3 -> 157.2
8: 277.5 -> 193.2     16: 286.9 -> 347.5
```

**Read as GEMM -> GEMV, that says t=16 favours the GEMM (286.9 vs 347.5), yet the code routes
16 to the GEMV.** So I changed the cut to `t <= 8` and let `forward_prefill` take t=16.

| | |
|---|---|
| gate | **16/16, `chunked-prefill` OK -- the change is CORRECT** |
| t=4 | 145.54 / 145.05 / 143.84 = **144.8** (unchanged) |
| t=8 | 177.96 / 181.52 / 177.21 = **178.9** (unchanged) |
| t=16 | **392.34 / 392.27 / 390.11 = 391.6** vs 312.23 before -- **+25% WORSE** |

**The comment's table does not reproduce in the current build: at t=16 the GEMM is 391.6 ms
against the GEMV's 312.2. The existing cut at 16 is correct, and it is now VERIFIED rather
than assumed.** Reverted.

**Methodological note:** I acted on a comment's numbers instead of measuring first -- **the
same error class as round 155 (a stale README read as current) and round 169 (a column's
meaning inferred rather than read).** This time the gate and a three-run A/B caught it in
**one** round, at the cost of one constant and two measurements. **That is the loop working
as intended, and it is the first time the error was caught this cheaply.**

**It also closes the last lead for the endpoint target:** with the routing verified correct,
the batched GEMV's per-token cost is not an artifact of a wrong dispatch choice. **The
endpoint's 30 tok/s would require eliminating ~100% of a real, measured, correctly-routed
cost term.**

### REATTRIBUTION (round 185): the endpoint prefill uses the GEMM, not the GEMV

**This is the most important structural fact for the endpoint target, and it invalidates the
target of rounds 173-184.**

* The endpoint's prompts are **~59 tokens**.
* `weights.rs:95` routes **`t <= 16`** to the batched GEMV, so **59 goes to
  `forward_prefill` -- the GEMM.**
* **`forward-cost` sweeps `t` in `[1, 2, 4, 8, 16]` -- entirely inside the GEMV branch.**
  **The sweep cannot see the GEMM at any point.**

**So the whole GEMV analysis (the two-regime sweep, the fixed-vs-per-token decomposition, the
L2 reuse window, the `ROWS` proposal) characterises a path the endpoint's prefill never
enters.** The endpoint's prefill cost -- **~7.63 s of the 12.909 s wall** -- lives in **the
GEMM**, which is exactly what rounds 148-168 targeted.

**Which restores the K split as the live lever.** It changes precisely the kernel the
endpoint uses; its correctness is verified (16/16 on three consecutive runs plus
`chunked-prefill` OK); and its performance was retired (round 168) on an A/B that round 174
showed could not contain it. **It is simultaneously the best-motivated and the only
gate-verified option left for this target.**

**A valid test is cheap and now well-defined:** re-apply the split, then measure `pre_ms` at
**t=32 or t=64** -- outside the GEMV branch -- or measure the endpoint directly.
**Three runs, gate first, and the acceptance test is the endpoint number, not `pre_ms` at 16.**

**Generalization, now paid for the fifth time: when a measurement shows no effect, first ask
whether the measured path contains the changed code.** Round 174 established that the split's
A/B failed this test; round 185 establishes that **the entire `forward-cost` instrument fails
it for any GEMM change** -- so no `forward-cost` result can ever license a GEMM keep/revert
decision.

### BREAKTHROUGH (round 186): the GEMM is measured, and the endpoint target is REACHABLE

**Extended the `forward-cost` sweep to `[1, 2, 4, 8, 16, 32, 64]`
(`gb10-verify/src/main.rs:1008`). `t=32` and `t=64` are outside `weights.rs:95`'s `t <= 16`
GEMV cut, so the GEMM branch is measured here for the first time.**

| t | `pre_ms` (mean of 3) | path | effective |
|---|---|---|---|
| 16 | 314.17 | GEMV | -- |
| 32 | **400.01** | **GEMM** | 44.0 GB/s |
| 64 | **435.94** | **GEMM** | **40.4 GB/s** |

**t=32 -> 64 costs only +35.9 ms for 2x the tokens -- NEARLY FLAT.** So the GEMM is not
per-token-bound: **the weight stream dominates, and it runs at 18% of the 228 GB/s peak.**

**And the endpoint arithmetic finally closes:** ~59-token prompts are one 64-wide chunk each,
so the prefill is **16 GEMM calls x 435.94 ms = 6.98 s**, against the measured ~7.63 s.
**The model now predicts the endpoint.**

### Reachability, computed correctly

**My first attempt divided one call's 17.608 GB by the whole 16-call budget. The prefill moves
16 x 17.608 = 281.7 GB.**

| | |
|---|---|
| prefill today | 6.98 s for 281.7 GB = **40.4 GB/s** |
| 30 tok/s budget | 256/30 = 8.53 s total; decode ~5.36 s, so **prefill must fall to 3.17 s** |
| required rate | 281.7 / 3.17 = **88.8 GB/s** |
| that is | **2.2x** the current 40.4 GB/s |
| the ceiling | the single-row GEMV already sustains **170 GB/s** on this machine, **4.2x away** |

**So the endpoint target is REACHABLE in principle: 2.2x on the GEMM against 4.2x of
headroom. It is a performance-engineering problem, not a physical one.** The single-decoder
100 tok/s target remains the only physically impossible one, at 7.7x peak bandwidth.

**And the lever is the K split**: gate-verified (16/16 x3, `chunked-prefill` OK), aimed at
precisely this kernel, and never validly measured -- because until this round `forward-cost`
could not see the GEMM at all.

### The proposal that follows, and how it differs from the rejected one

**Not shared-memory staging** (round 179: measured, rejected, and its ruled-out entry names
software pipelining as the alternative). **Instead: keep the same `x` slab resident while
sweeping MORE rows per traversal.**

**Concretely, `ROWS = 8` with the batch loop tiled to 8**, so `acc[8][8]` is **64 floats --
identical to today's `acc[4][16]`** -- while the number of row-group traversals **halves**,
halving the `x` re-reads. The `lo/hi[ROWS][8]` pair doubles to 128 floats, so total register
pressure goes from ~128 to ~192 against 255 available: **tight, and the first thing to check
is whether it spills.**

**This is a distinct, register-budgeted change with a measured target (the 16.58 ms/token
marginal cost) and a stated acceptance test (the t=4/8/16 curve, 3 runs, gate green).**

### Superseded: the round-178 proposal

**The batched GEMV re-reads `x` from L2 for every row group.** The kernel header names the
problem ("makes the kernel L2-bound"), `ROWS` is register-capped at 4, and round 127 already
showed `ROWS=2` is worse. **So the remaining lever is not `ROWS` -- it is where the re-reads
land.**

**Proposal: stage the block's `x` slab in SHARED memory once per k-tile, so the
`ROWS x B` reads hit shared instead of L2.** Shared memory is on-SM and roughly an order of
magnitude faster than L2 for this access pattern, and the slab is small: one `kTile` of `B`
batch rows is `512 x 16 x 4 B = 32 KB`, against 121 GiB of unified memory and a 256-thread
block.

**This is a genuine, unexplored mechanism** -- unlike the eleven prefill mechanisms
eliminated so far, it targets a bottleneck that has now been *located in source* rather than
inferred from a timing.

**How to measure it honestly, given rounds 174-177:** the change lives in
`nvfp4_gemv_batch_tmpl`, **so the benchmark must call a path with `t > 16`** -- `weights.rs:95`
routes `t <= 16` to the batch==1 kernel, and `forward-cost` at t=8/16 therefore cannot see
this change at all. **`chunked-prefill`, the endpoint, or `forward-cost` at t=32/64 will.**
Three runs, spread reported, and the gate (`generate --n 16`, `chunked-prefill`) must pass
before any number is quoted.

### Next step

**A shape question about the GEMV's batch handling** (`kernels/gemv.cu`, `GB10_BATCH_MAX`,
and the `ROWS` experiments of rounds 104/121/127):

**why does the batched GEMV lose bandwidth as batch size grows?** Rounds 104 and 127
rejected `ROWS=2` at B=1 and B=16, but **nobody has asked what happens to the weight stream
when B grows. A batched GEMV that re-reads the weights once per token rather than once per
group would explain 190 -> 59 exactly, and that is checkable by reading the k-loop's
indexing rather than by benchmarking.**

## UNIFICATION (round 188): the two unmet targets are ONE problem

**Targets still unmet:**

1. **TTFT better than llama.cpp.** A 59-token prompt is one prefill = **~436 ms** (round
   186's t=64 GEMM); llama.cpp does it in **59 x 1.25 = 74 ms**. **A 5.9x deficit.**
2. **endpoint at 16 concurrent >= 30 tok/s.** Currently **20.08**; needs the GEMM at
   **88.8 GB/s = 2.2x.**

**Both are the prefill GEMM at 40.4 GB/s = 18% of the 228 GB/s peak. No other factor appears
in either.**

**So this is one mechanism with two thresholds, and the headroom check separates them:**

| requirement | rate needed | verdict |
|---|---|---|
| endpoint (2.2x) | **88.9 GB/s** | **INSIDE known-achievable territory** -- below the 170 GB/s the single-row GEMV already sustains here |
| TTFT (5.9x) | **238.4 GB/s** | **OUTSIDE it** -- 1.4x more than this machine has ever shown |

**So the endpoint target is reachable and the TTFT target is not, and they are the same
change.** Aim at 2.2x; if it is achieved, the endpoint target is met and TTFT improves by the
same factor while remaining short of llama.cpp.

## A CODE-DERIVED CANDIDATE FOR THE GEMM'S 18% (round 189)

**Every mechanism proposed for the prefill GEMM so far was inferred from a timing. This one is
read out of the staging loop, and it is the first that can account for 18% of peak.**

`kernels/gemm.cu:65` `stage_wtile`:

```cuda
constexpr int PAIRS = KC / 16;            // KC = 32 -> 2
constexpr int UNITS = GB10_TN * PAIRS;    // 64 * 2 = 128
const int u  = threadIdx.x + p * GB10_GEMM_BLOCK;
const int nl = u / PAIRS, pr = u % PAIRS; // <- decomposed, not n = u
const int n  = nbase + nl;
pk[p] = *reinterpret_cast<const uint2*>(w + (size_t)n * (K >> 1) + ...);
```

**Two consecutive lanes cover 16 contiguous bytes of one row; the next pair moves to the next
row. A warp therefore touches 16 different rows, taking 16 B from each: 16 B used per 128 B
line = 12.5%, against the measured 18%. Same order -- and now verified against the code.**

**Why 18% and not 12.5%:** successive `c` iterations advance by `KC = 32` bytes, so the next
`stage_wtile` reads the *next* 16 bytes of the same rows. **The rest of each line is recovered
from L2 if it survives**, which lifts 12.5% toward 18%. **Each 128-byte line is being filled
over four k-tile iterations.**

**The fix is now precise:** have a warp read **128 contiguous bytes of ONE row** instead of
16 B from each of 16 rows -- i.e. **lane-order the staging loop `PAIRS`-major so 16 lanes
cover one row's full 128 B before moving on.** That is where the 2.2x would live.

**And the falsification check earned its keep:** the first reading was `n = u`, which would
have implied a flat 12.5% with no L2 recovery. The real decomposition changes both the
arithmetic and the fix.

## CORRECTION (round 190): the round-189 fix is IMPOSSIBLE -- the real fix is a transpose

**Arithmetic on the row stride kills the round-189 proposal. The weight matrix is row-major
with row stride `K/2` bytes (NVFP4 = 0.5 B/element). At `K = 5120` that is 2560 bytes, and a
`KC = 32` k-tile covers exactly **16 contiguous bytes of a row**.

**So "128 contiguous bytes of one row" does not exist. 128 contiguous bytes spans 8 different
rows instead. Lane reordering cannot coalesce this load, because the 16 B per line IS what the
row-major layout permits for a 32-element k-tile.**

### The real fix: transpose the weights

**Store them `[k][row]` instead of `[row][k]`. Then 128 contiguous bytes cover 8 rows for a
single k, the staging load is fully coalesced, and the group-scale factors transpose with it
(they are per 16 elements along k).**

**The shared tile `wt` stays as it is and keeps serving the outer product unchanged -- only the
global layout and the `stage_wtile` copy change.**

**Cost: a one-time transpose at load. 17.608 GB read + write is ~35 GB, which is ~0.154 s once,
against a prefill of 6.98 s per endpoint request. It pays for itself in the first request.**

**This is the first proposal in the session derived from the memory layout rather than from a
timing -- and the check that produced it was arithmetic on the row stride, the same kind of
check that falsified the previous four mechanisms.**

**Caveat before implementing:** the transposed layout touches **all three staging variants**
(`stage_wtile`, `stage_wtile_fp8`, and the bf16 path), the NVFP4 group-scale layout, and any
code that assumes `row * (K/2)` addressing -- including the model loader. **Read all three
before writing any of them.**

## FLOOR CHECK (round 191): the GEMM is 5.6x its floor, and it is efficiency, not traffic

**The roofline floor for one weight pass is `17.608 GB / 228 GB/s = 77.2 ms`. The GEMM at t=64
measures 435.94 ms -- 5.6x the floor, i.e. 40.4 GB/s.**

**And `stage_wtile` reads each weight exactly once** (`UNITS = GB10_TN * PAIRS` per k-tile per
block, covering the block's full strip). **So the shortfall is not redundant traffic: it is
pure access-pattern efficiency.** That is exactly what the round-190 transpose targets, and
it rules out the alternative explanation ("the kernel is moving the weights more than once").

**State of the endpoint target, complete:**

| lever | result |
|---|---|
| occupancy / K split | **real, 2-3%** (merged, round 187) |
| batched GEMV path | **irrelevant -- wrong branch** (round 185) |
| dispatch cut | **correct as-is** (round 184) |
| redundant weight traffic | **ruled out** (round 191) |
| **weight layout (row-major, 16 B/line)** | **the open mechanism**; fix = transpose to `[k][row]`, ~0.154 s once |

**The GEMM needs 88.9 GB/s for the endpoint target -- 2.2x from 40.4. The machine has
demonstrated 170 GB/s elsewhere, so the requirement sits inside known-achievable territory.**

## THE CONTIGUITY HYPOTHESIS IS FALSIFIED (round 192)

**Tested it directly instead of building the transpose. `GB10_KC 32 -> 64` (`kernels/gemm.cu:41`)
makes `PAIRS = KC/16` go 2 -> 4, doubling the contiguous bytes read per row from 16 to 32 -- a
minimal, one-constant intervention on exactly the quantity rounds 189-190 blamed.**

**Gate: 16/16, `chunked-prefill` OK. `dirty:1`, so the change was live. Result -- WORSE:**

| t | baseline | KC=64 | |
|---|---|---|---|
| 32 | 400.01 | **413.85** [410.54-416.17] | **+3.5%** |
| 64 | 435.94 | **452.76** [448.23-455.59] | **+3.9%** |

**Doubling the per-row contiguity made the GEMM slower. So the "16 B per 128 B line" story does
not survive its own intervention, and the round-190 transpose proposal -- which rested entirely
on it -- is demoted. Reverted.**

**The confound, stated because it keeps this honest:** `KC` also changes shared-memory size and
halves the outer-product iteration count. **But the confound does not rescue the hypothesis: if
contiguity were the limiter, doubling it should have produced a gain large enough to survive a
modest shared-memory increase. It produced the opposite sign.**

### Where this leaves the prefill GEMM

**Five mechanisms are now closed by measurement:**

1. occupancy / K split -- **real, 2-3%** (merged);
2. the batched GEMV path -- **wrong branch**;
3. the dispatch cut -- **correct as-is**;
4. redundant weight traffic -- **ruled out** (each weight read once);
5. **weight-layout contiguity -- FALSIFIED** (this round).

**The GEMM sits at 40.4 GB/s, 5.6x its 77.2 ms floor, and no lever in hand moves it.** The
endpoint target is still reachable *in principle* -- 88.9 GB/s is below the 170 GB/s this
machine demonstrates elsewhere -- **but no mechanism for it is currently identified.** That is
the honest state, and it is different from "impossible": it is "not yet found".

## THREE GEMM VARIANTS (round 193): I never checked which one runs

**The staging-only ablation did not run. Its anchor -- `gemm2d_outer_bf16(wt[cur], xt[cur], acc,
ty, tx);` -- occurs THREE times, at `kernels/gemm.cu:402`, `:436` and `:469`. The assert
rejected 3 != 1, nothing was written, and `git status` is empty, so the printed
432.59 / 428.46 / 425.52 ms are the baseline again, not a staging-only result.**

**But the failure surfaced the thing that matters: `gemm.cu` contains three outer-product call
sites, i.e. at least three GEMM kernel variants -- and I have been reasoning about "the prefill
GEMM" since round 148 without once establishing which variant `forward_prefill` launches.**

**This is round 176's lesson repeating exactly.** There, `nvfp4_gemv` turned out to dispatch
between two kernels and I had read the wrong one. Here there are three.

### What this invalidates, and what it does not

| conclusion | status |
|---|---|
| **KC=64 is worse** (round 192) | **VALID** -- measured end-to-end through `forward-cost`, so it tested whichever variant actually runs |
| K split is 2-3% (round 187) | **VALID** -- same reason |
| no redundant weight traffic (round 191) | valid for the launched variant only |
| **the 16 B-per-line layout arithmetic** (round 189) | **UNSUPPORTED** -- rests on reading one specific `stage_wtile`, and that function may not belong to the launched variant |
| **the transpose proposal** (round 190) | **UNSUPPORTED**, same reason |

**End-to-end measurements survive; code-reading conclusions do not, until the variant is
identified.**

### The first step next round is identification, not another mechanism

**Which of the three outer-product variants does `nvfp4_gemm` launch, and does the `stage_wtile`
I analysed (`kernels/gemm.cu:65`) belong to it?** One grep of the three call sites' enclosing
kernel signatures answers it -- and it must be answered before any further layout reasoning.

**This is now the fourth time this session that reading the wrong object -- a kernel, a column, a
comment, a ruled-out list entry -- produced a confident conclusion that had to be retracted.
The pattern is stable enough to name: verify the identity of the object before reasoning about
its properties.**

## THE IDENTITY CHECK COMES BACK CLEAN (round 194)

**`kernels/gemm.cu:479/486/492` defines three entry points -- `nvfp4_gemm_kernel`,
`fp8_gemm_kernel`, `bf16_gemm_kernel` -- and all three call the SAME shared `gemm2d_*` helper
family.** So the three outer-product call sites at `:402`, `:436` and `:469` are the
**nvfp4 / fp8 / bf16 variants of one helper**, not three unrelated kernels, and `awk` confirms
line 402 sits inside a `__device__` helper rather than a `__global__` entry point.

**So the `stage_wtile` analysis of rounds 189-190 WAS about the right code.** The NVFP4 variant
is the one `nvfp4_gemm_kernel` uses, and NVFP4 carries the bulk of the prefill traffic.

**Which means the transpose proposal is demoted by MEASUREMENT, not by misidentification.**
Round 192's `KC=64` test doubled exactly the quantity the layout argument blamed and made the
GEMM **3.5-3.9% slower**. That is a valid end-to-end result, and it -- not the variant question
-- is what closes the layout lead.

**The round-193 alarm was worth raising and is now answered.** The risk was real: it had already
cost a retraction at round 176. Checking it took one grep.

**And the distinction is worth keeping.** The four "wrong object" incidents (169, 176, 184, 193)
were each caught by a cheap identity check, and **this one came back clean**:

> **The check costs one grep. Its failure mode is a retraction; its success mode is a licence to
> keep reasoning. Run it every time.**

**The object was correctly identified; the hypothesis was still wrong.** Those are independent
facts, and conflating them is what made round 193's alarm feel like a refutation when it was
only a question.

## THE GEMM DECOMPOSED (round 195) -- and it explains why five proposals failed

**The staging-only ablation finally ran.** Method: disabled the NVFP4 variant's outer-product
call at `kernels/gemm.cu:402` **by line index** -- the round-193 anchor attempt failed because
that call appears three times, and the line-index edit is unambiguous (`dirty:1` proves it
applied). **The gate fails by design here; this is a timing probe and its number must never be
read as a result.**

**t=64, 3 runs:**

| | ms | share | rate |
|---|---|---|---|
| staging only | **213.20 / 213.85 / 215.19 = 214.08** | **49%** | **82.2 GB/s = 36% of the 228 peak** |
| outer product (by difference) | **221.86** | **51%** | **3.94 TFLOPS (~22% of a ~18 TFLOPS CUDA-core ceiling)** |
| total | **435.94** | | 40.4 GB/s |

**So the time splits almost exactly in half, and both halves sit 2-3x below their own ceilings.
Staging alone runs at twice the rate the full kernel achieves, so the layout story was at most
half the problem -- and the other half is pure FMA on dequantized bf16.**

### This is the answer to "why did five proposals fail"

**There is no single dominant mechanism.** Every proposal from rounds 148-192 targeted one half
-- occupancy, K splitting, the launch cut, traffic redundancy, the layout -- and **any fix that
addresses only one half can win at most a fraction of the other half's cost.** That is exactly
the pattern the measurements showed: 2-3% wins, and three falsifications.

**The only class of fix that addresses both at once is a bigger tile / better data reuse**, so
each staged byte feeds more FMAs *and* each FMA chain has more independent work.

**But rounds 96-119 already established that exactly one tile shape passes the B/FMA gate at
`GB10_GEMM_BLOCK = 128`** -- so the honest next question is not "which shape" but **whether the
block size is the free variable**: with 256 threads the per-thread accumulator budget halves,
which changes which shapes pass the gate at all. **That is a bounded, gate-checkable experiment
and it is the only lead this decomposition leaves open.**

## BLOCK SIZE IS NOT FREE -- BUT A BIGGER TILE AT 256 THREADS IS (round 196)

**Worked the round-195 lead before spending a build. It closes, and it opens a better one.**

**The accumulator identity checks out against the code:** 128 threads with the
`static_assert`'d `GB10_TM = 8, GB10_TNREG = 4` gives a block tile of 64n x 64t = 4096
outputs, and 4096/128 = 32 = TM*TNREG. **So `threads x acc = 4096`, i.e. `acc = 4096/threads`.**

**And the B/FMA closed form is confirmed by the counted source:** for TM=8/TNREG=4,
`2/TNREG + 4/TM = 0.5 + 0.5 = 1.0` -- **exactly the ~1.0 B/FMA counted in `gemm2d_outer_bf16`
in round 171.** Formula and code agree.

**Raising `GB10_GEMM_BLOCK` to 256 at the SAME tile forces `acc = 16`:**

| shape | acc | B/FMA |
|---|---|---|
| TM=4 / TNREG=4 | 16 | **1.5** |
| TM=8 / TNREG=2 | 16 | **1.5** |
| today: TM=8 / TNREG=4 at 128 threads | 32 | **1.0** |

**So at a constant tile, more threads is strictly worse. Block size is not the free variable.**

### What is free: enlarge the tile at 256 threads

**A 64x128 or 128x64 block tile carries 8192 outputs, so `acc = 32` at 256 threads and B/FMA
stays at 1.0 -- while each staged byte feeds twice as many FMAs.** That is precisely the
"bigger tile / better reuse" fix the round-195 decomposition pointed at, now with a concrete
shape and a stated reason to expect it to pass the gate.

**Coupled constants:** `GB10_TN` in `kernels/gemm.cu` and `GB10_NR` in `crates/gb10-cuda/src/ops.rs`
must move together, `GB10_WSTRIDE` follows `GB10_TN`, and the launch `block_dim` goes to 256.
**This is a shape change of the kind rounds 96-119 searched -- but at a block size they never
tried, which is what makes it new rather than a repeat.**

**Acceptance test, unchanged: the gate first, then 3 runs at t=32/64 against the recorded
baseline of 400.01 / 435.94 ms.**

## EXECUTION PLAN: the 256-thread / 64x128 tile experiment (round 197)

**Round 196 derived the shape but ran out of context before executing it. This is the edit
list, so the next session does not re-derive it.**

**Why it is plausibly a win, in one line:** at 256 threads with an 8192-output tile, `acc = 32`
and **B/FMA stays at 1.0** while **each staged byte feeds 2x the FMAs** -- and the round-195
decomposition says staging is 49% and the outer product 51%, so this is the only lever that
addresses both halves at once.

**All five edits are coupled. Change them together or the gate fails.**

| # | file | change |
|---|---|---|
| 1 | `kernels/gemm.cu` | `GB10_TN` 64 -> **128**. `GB10_WSTRIDE` follows automatically (line 60: `#define GB10_WSTRIDE GB10_TN`). |
| 2 | `kernels/gemm.cu` | `GB10_GEMM_BLOCK` 128 -> **256**. |
| 3 | `kernels/gemm.cu` | `gemm2d_ids` -- the mapping must satisfy `ty_groups * tx_groups == GB10_GEMM_BLOCK` with `ty_groups = 64/TM = 8`, so `tx_groups = 256/8 = 32`: **`tx = threadIdx.x & 31`, `ty = threadIdx.x >> 5`**. |
| 4 | `crates/gb10-cuda/src/ops.rs` | `GB10_NR` 64 -> **128**, and the NVFP4 prefill launch's `block_dim` `(128,1,1)` -> `(256,1,1)` (~line 1191). |
| 5 | `kernels/gemm.cu` | Verify `static_assert(GB10_TM == 8 && GB10_TNREG == 4)` in `gemm2d_outer_bf16` still holds -- it should, since the tile grows along **n**, which is `TN`, not `TM`/`TNREG`. Confirm the inner column loops now cover 128. |

**Edit 3 is the one most likely to be forgotten, and it is the one the gate catches.**

**Acceptance: gate first (`generate --n 16`, `chunked-prefill --n 6`), then 3 runs of
`forward-cost` at t=32/64 against the recorded baseline of 400.01 / 435.94 ms.** If the gate
fails, revert -- do not tune.

**Risk, stated plainly: five coupled edits, and the session that derived them ran out of context
before executing them. The failure mode is a broken gate and the safety net is
`git checkout -- kernels/gemm.cu crates/gb10-cuda/src/ops.rs`.**

## THE SHAPE EXPERIMENT WAS ATTEMPTED TWICE AND FAILED THE GATE (round 198)

**Both attempts auto-reverted clean (`dirty:0`). The gate is the reason nothing broken was left
behind.**

**Attempt 1.** The patch asserted out on `a5` -- `block_dim: (128,1,1)` occurs **4** times, not
the 3 the plan assumed -- so `ops.rs` was never written while `gemm.cu` already was. That
produced a kernel compiled for 256 threads launched with 128: **gate 0/16**. **This is exactly
the inconsistency the gate exists to catch, and it caught it.**

**Attempt 2.** With the count corrected to 4, all five edits applied and the build was clean.
**Gate 0/4.** So the 256-thread / 128-token-tile shape is **not correct as constructed** -- at
least one more coupling exists that the five-edit list does not name.

### What the round-196 arithmetic got right, and what it got wrong

**Right:** `tx_groups` is tied to the **token** dimension -- `GB10_TT / GB10_TNREG = 128/4 = 32`
-- **not to N**. So `GB10_NR`, `GB10_WSTRIDE` and `GB10_TN` correctly stay at 64, and the
round-197 plan was wrong to list them. **The real change set is smaller and lies along T.**

**Wrong:** something else in the kernel assumes a 64-token tile. **Candidates, in the order worth
checking:** the `xt` shared tile's declared size; any loop bound written as a literal or derived
from `GB10_TT`; the store indexing that maps `tx` back to token columns. **One grep for
`GB10_TT` and for `tx` inside `kernels/gemm.cu` names it.**

### Verdict on this lead

**It is still the right class of fix** -- the only one that addresses both halves of the
round-195 decomposition at once. **But it is not a constant swap: it is a real kernel change,
and it needs a session that can iterate on it with the gate in the loop.** That session was not
this one, and stopping here is what kept the tree green.

## THE BIGGER-TILE LEVER IS CLOSED (round 199) -- gate green, 50% slower

**The round-198 failure had one cause and it was found: the 4th `block_dim: (128,1,1)` belongs to
an ATTENTION kernel at `crates/gb10-cuda/src/ops.rs:1120` (`grid_dim: (n_v_heads, 1, 1)`) that
must stay at 128. Changing all four handed it 256 threads and produced garbage.**

**Targeting only the three GEMM launches at `:1195`, `:1210`, `:1225` -- with `GB10_TT` 64->128,
`GB10_GEMM_BLOCK` 128->256, `gemm2d_ids` to `>>5`/`&31`, and `GB10_TILE_T` 64->128 -- PASSED the
gate 16/16 and `chunked-prefill`. The shape is CORRECT.**

**And it is 50% slower:**

| t | baseline | 256-thread / 128-token tile | |
|---|---|---|---|
| 32 | 400.01 | **600.43** [597.36-602.84] | **+50%** |
| 64 | 435.94 | **658.00** [654.66-661.31] | **+51%** |

### The reason is arithmetic, and it is fatal for this objective

| t | tile columns useful | wasted |
|---|---|---|
| 32 | 32 of 128 | **75%** |
| 64 | 64 of 128 | **50%** |
| 128 | 128 of 128 | 0% |

**The endpoint's prefill is ~59 tokens**, so the tile would be **46% empty** while the y-dimension
block count **halves**. **The tile only pays for itself at t >= 128, which this workload never
reaches** -- and `GB10_TILE_T = 64` was chosen to match exactly those ~59 tokens.

**So the "bigger tile" lever is closed too, and closed by measurement with the gate green. The
round-195 decomposition still stands (49% staging / 51% outer product); the specific fix it
pointed at simply does not apply at the token counts that matter. Reverted.**

### Six mechanisms closed, every one of them by measurement

| # | mechanism | result |
|---|---|---|
| 1 | occupancy / K split | real, **2-3%** (merged) |
| 2 | batched GEMV path | wrong branch |
| 3 | dispatch cut | correct as-is |
| 4 | redundant weight traffic | ruled out |
| 5 | weight-layout contiguity | **falsified** (KC=64) |
| 6 | **bigger tile at 256 threads** | **correct but 50% slower** (this round) |

**The GEMM stays at 40.4 GB/s. No mechanism for the remaining 2.2x is currently identified.**

## THE TWO PHASES SUM EXACTLY -- THEY DO NOT OVERLAP (round 200)

**This is the finding the whole session was missing.**

```
staging 214.08 + outer product 221.86 = 435.94
measured total                          435.94
sum - total = 0.00 ms
```

**The two phases ADD. They do not overlap at all.**

**If they overlapped, the total would be `max(214, 222) = 222 ms` -- a 1.96x win. The endpoint
target needs 2.2x. THE PRIZE IS THE SAME ORDER AS THE TARGET.**

### This reframes the entire search

**Six mechanisms were closed by asking "how do we make staging faster, or the FMA denser".**
**The measurement says the direct question is "why do the two phases serialize" -- and the prize
for answering it is essentially the whole remaining target, not the 2-3% every previous lever
delivered.**

**And the code already intends overlap**: `xt[2]`/`wt[2]` double buffers, and
`stage_xtile(xt[nxt], ..., c + 1)` is issued before `gemm2d_outer(wt[cur], xt[cur])`.
**So the structure says "pipeline" and the measurement says "serial".**

### Why they might serialize, in the order worth checking

1. **A `__syncthreads()` between staging and compute forces every thread to wait for the slowest
   load, instead of letting compute consume the *other* buffer.**
2. **The buffers may be consumed in the same order they are produced**, making a dependency chain
   rather than a pipeline.
3. **If `cur`/`nxt` swap at a point that puts a barrier between issue and use, the load latency is
   exposed on the next iteration instead of hidden.**

**One check decides it: does the loop issue the loads for `k+1` and then compute on `k` *without
an intervening barrier that both phases must pass together*? A single read of the `nchunk` loop
body answers it.**

### And a measurement that would confirm the diagnosis before any rewrite

**Reduce the outer-product work (e.g. `GB10_TNREG` 4 -> 2) and see whether the total falls by the
FULL amount of the removed FMA time or by less.**

- **falls by the full amount -> the phases are serialized, and overlap is the prize;**
- **falls by less -> they already overlap, and the exact sum is a coincidence.**

**That is a one-constant experiment with a recorded baseline, and it is the next thing to run.**

## ROUND 200 IS NOT REPRODUCED AT A SECOND POINT (round 201)

**The staging-only ablation, run at two token counts, 3 runs each:**

| t | staging-only | full | difference |
|---|---|---|---|
| 32 | **178.88** [177.26-180.06] | 400.01 | **221.13** |
| 64 | **213.63** [211.38-216.88] | 435.94 | **222.31** |

**The pattern is INVERTED from what the serialization model requires.** Staging is the weight
stream and should be **t-independent** -- yet it moves **178.88 -> 213.63 (+19%)** when t doubles.
The outer product should **scale with t** -- yet the difference is **221.13 vs 222.31, essentially
constant.**

**Both have innocent explanations individually:** the tile is 64 tokens wide, so at t=32 half the
columns are computed as zeros and the FMA count is the *same* as at t=64 -- **which explains the
constant difference exactly**; and `stage_xtile` masks its loads by t, so staging legitimately
varies.

**But together they mean the round-200 "they add, they do not overlap" reading is NOT
established.** The exact sum `214.08 + 221.86 = 435.94` was real arithmetic on real
measurements -- **but it was ONE point, and a second point does not reproduce the same
decomposition. A model that fits one point is not a model.**

**This is the session's own rule applied to the session's own best finding:** *"a synthetic model
matching on the endpoint but not the derivative is not a model of the kernel."* Round 200 matched
the endpoint. Round 201 shows it does not match the derivative.

### Status of each claim

| claim | status |
|---|---|
| staging is 179-214 ms; outer product is ~221 ms; both large, neither dominates | **STANDS** (round 195) |
| the two phases add exactly, so they do not overlap | **DOWNGRADED to unconfirmed** |
| overlap is worth 1.96x | **unconfirmed** -- it depends on the claim above |

**And t=32 is not an independent test**: with a 64-wide tile its FMA count equals t=64's, so it
cannot probe FMA scaling. **A valid second point needs a change that actually alters the FMA
count -- i.e. a shape change -- and every shape change tried so far has been coupled or a loss.**

**Reverted; tree clean.**

## THE ABLATION ITSELF IS SUSPECT (round 202)

**Round 201 left a 35 ms staging difference (178.88 at t=32 vs 213.63 at t=64) with no credible
source. Ruling out the candidates:**

| candidate | size | verdict |
|---|---|---|
| x traffic (`stage_xtile`) | 2048 B/k-tile, 320 KB/block, **26.2 MB per GEMM call** | **2.4% of the 1100 MB weight traffic -- cannot produce 35 ms** |
| the epilogue store | 32 x 64 x 4 = **8 KB per block** | smaller still |

**So the most likely explanation is that the ablation does not measure the same thing at both
sizes.** With the outer product removed, `wt` and `xt` feed nothing but a zero accumulator, **and
a compiler is free to eliminate loads it can prove dead -- at one size and not the other,
depending on how the unrolled code falls.** The `__syncthreads()` guards make this less likely
but not impossible.

### This puts the round-195 decomposition in question too

**The 49% staging / 51% outer-product split is the ONLY mechanism-level result this session
produced, and it rests on the same ablation. It is worth one careful re-derivation.**

**The fix is a probe that cannot be optimized away: keep a dependency on the staged data.** Do
not delete the outer product -- instead, accumulate something derived from `wt`/`xt` into the
output (e.g. sum a few elements of each into `acc[0][0]`). **Then the loads stay live, the FMA
work is removed, and the difference is a real staging measurement.**

**And the rule that generalizes, which this session paid for twice:**

> **An ablation that removes the *use* of data can also remove the *load* of it. Remove the
> computation, but keep a dependency on the data.**

## THE ABLATION WAS WRONG -- AND CORRECTING IT CLOSES THE ARITHMETIC (round 203)

**Round 202 predicted the ablation was eliminating dead loads. This confirms it and quantifies it.**

**Method: replace the deleted outer product with a DEPENDENCY-PRESERVING probe -- one read from
`wt` and one from `xt`, accumulated into `acc[0][0]` -- so the staged data stays live and the
compiler cannot drop the loads, while the FMA work is still removed.**

| | t=32 | t=64 |
|---|---|---|
| **probe (loads live, FMA removed)** | **279.62** [277.35-282.32] | **322.33** [319.62-324.18] |
| deleted-outer-product ablation (round 195) | 178.88 | 213.63 |
| full kernel | 400.01 | 435.94 |

**The probe measures 100.74 ms (t=32) and 108.70 ms (t=64) MORE than the ablation did. So round
195's ablation WAS eliminating loads, and its "staging = 214.08 ms" was an underestimate by
about half.**

### The corrected decomposition at t=64

| phase | ms | share | rate |
|---|---|---|---|
| **staging (loads live)** | **322.33** | **74%** | **54.6 GB/s = 24% of the 228 peak** |
| outer product (by difference) | **113.61** | **26%** | -- |
| total | 435.94 | | 40.4 GB/s |

**So the real picture is NOT "two co-equal halves". Staging dominates at 74% and runs at 24% of
peak -- worse than the 36% the flawed measurement suggested. The outer product is not the
co-bottleneck it appeared to be.**

### And the arithmetic now closes exactly on target

**At peak bandwidth the staging would take `17.608 / 228 = 77.2 ms`, and
`77.2 + 113.6 = 190.8 ms` against today's 435.94 -- a 2.28x speedup. The endpoint target needs
2.2x. THE ARITHMETIC CLOSES ON TARGET.**

**So the objective is reachable and the lever is now unambiguous: the staging must move from
54.6 GB/s toward peak. The layout leads of rounds 189-190 were demoted by a confounded test
(KC=64) and deserve re-examination now that staging is known to be 74% of the time, not 49%.**

### The rule, paid for twice

> **An ablation that removes the *use* of data can also remove the *load* of it. Remove the
> computation, but keep a dependency on the data.**

**Round 202 predicted this. Round 203 confirmed it and put a number on it: 108 ms.**

## WHY THE STAGING RUNS AT 24% (round 204) -- one load per thread per barrier

**Read out of the code, not inferred:**

```cuda
constexpr int PAIRS = KC / 16;                                  // 32/16 = 2
constexpr int UNITS = GB10_TN * PAIRS;                          // 64 * 2 = 128
constexpr int P = (UNITS + GB10_GEMM_BLOCK - 1) / GB10_GEMM_BLOCK;  // ceil(128/128) = 1
```

**`P = 1`: exactly ONE load per thread per k-tile.** And there are
`nchunk = K / GB10_KC = 5120/32 = 160` k-tiles, **each followed by a `__syncthreads()` before
the outer product consumes the buffer.**

**So the pipeline is:**

```
issue 1 load -> store to shared -> BARRIER -> compute -> BARRIER -> next k-tile
```

**The memory pipeline drains at every barrier.** One 8-byte load in flight per thread cannot
cover DRAM latency across a barrier, which is exactly the 24%-of-peak signature.

**Note this is consistent with the ruled-out list** -- "bytes in flight" and "DRAM latency" were
both tested and dismissed -- **but those tests were on the GEMV, not on this kernel.** The
ruled-out list is per-kernel and does not transfer.

### The fix is local to the staging loop, not a layout change

**Issue MORE loads per thread before the barrier:**

1. **Unroll the k-loop by 2 and stage two k-tiles per barrier** -- `P` becomes 2 with two
   independent loads in flight per thread;
2. **Have each thread load more than one 8-byte unit per k-tile** -- raise the effective `PAIRS`
   without raising `KC`, by splitting a row across fewer lanes.

### And the instrument to test it is now trustworthy

**The dependency-preserving probe from round 203 is the right tool** -- it keeps the staged data
live so the compiler cannot drop the loads. **Measure the staging phase alone, before and after
the unroll.**

**The prize, unchanged and now well-founded:** staging 322.33 ms -> 77.2 ms at peak, total
435.94 -> 190.8 ms, **2.28x against a 2.2x target.**

## THE UNROLL IS NOT A ONE-CONSTANT CHANGE (round 205) -- and the cheaper test comes first

**The staging load has NO spare width.** 128 threads x 8 B (`uint2`) = **1024 B** per k-tile, and
the k-tile is exactly 64 rows x 16 B = **1024 B**. **Exact match -- coverage is complete.**

**So `P` cannot be raised by widening.** Raising `PAIRS` alone would read past the k-tile into
the next one's bytes -- **wrong data, not a faster load.**

**The unroll therefore requires restructuring the `nchunk` loop**: stage into both `wt[0]`/`wt[1]`
before one barrier, then run two outer-product passes. **That touches the K-split prologue
(`kc0`) and the `xt` staging in the same breath, so it is not a one-constant experiment.**

### The cheaper diagnostic, which comes first

**Keep the loop structure and add ONE extra independent global load per thread** -- from the next
k-tile's `wt` bytes into a scratch slot -- **purely to put a second request in flight.**

**It must be a GLOBAL load, so it belongs inside `stage_wtile`.** The probe line at the
outer-product site cannot do it: `wt`/`xt` there are already shared memory, **and a second shared
read adds no memory-level parallelism.**

**Read the result this way:**

- **staging speeds up materially -> the barrier-drain diagnosis is confirmed, and the restructure
  is worth its cost;**
- **staging does not speed up -> the 24% has a different cause, and the restructure would have
  been wasted work.**

### Baselines to beat (both from the trustworthy round-203 probe)

| measurement | t=32 | t=64 |
|---|---|---|
| **staging alone (loads live, FMA removed)** | **279.62** | **322.33** |
| full kernel | 400.01 | 435.94 |

## THE DIAGNOSTIC REFUTED THE ALTERNATIVE -- AND THE LAYOUT MATCHES EXACTLY (round 206)

**Round 205 asked for one cheap test before any restructure. It ran, and it settled the question.**

**Method: a second independent GLOBAL load per thread inside `stage_wtile`** (from the next
k-tile's bytes, accumulated into `pk[p].x` so it cannot be eliminated), **on top of the
dependency-preserving probe.**

| | t=32 | t=64 |
|---|---|---|
| **staging + 2 loads in flight** | **280.90** [279.22-284.15] | **326.03** [322.51-329.76] |
| staging, 1 load (round 203) | 279.62 | 322.33 |

**No change: +0.5% at t=32 and +1.1% at t=64, both inside noise.** So **memory-level parallelism
is not the limiter**, the "one load per barrier" diagnosis is **refuted**, and the `nchunk`
restructure would have been wasted work.

### And that points back at the layout, with an exact match

**A warp of 32 lanes reads 8 B each = 256 B -- but from 16 different rows** (two lanes per row,
per the `PAIRS` decomposition). **Each row contributes 16 B of a 64-byte line.**

```
efficiency = 16 rows x 16 B / (16 rows x 64 B) = 25%
measured staging rate = 54.6 / 228 GB/s  = 24% of peak
```

**QUANTITATIVE MATCH, not directional.**

**And the `KC=64` test that appeared to falsify this was confounded** -- it changed shared-memory
size AND halved the outer-product iteration count. **The layout hypothesis was never tested in
isolation. With the MLP alternative now excluded, it is the leading candidate again.**

### The fix, and its cost

**Transpose the weight layout to `[k][row]`** (rounds 189-190). Then a warp's 32 lanes read **64
contiguous bytes of one row**, one cache line is fully consumed, and the staging rate should
approach peak.

**Cost: ~0.154 s once at load** (17.608 GB read + write at 228 GB/s), against a **6.98 s prefill
per endpoint request.** It pays for itself in the first request.

**The prize is unchanged and now well-founded:** staging **322.33 -> 77.2 ms**, total
**435.94 -> 190.8 ms**, **2.28x against a 2.2x target.**

## THE LANE REORDER IS IMPOSSIBLE -- THE TRANSPOSE IS THE ONLY FIX (round 207)

**With the layout now the confirmed cause (round 206: 25% predicted, 24% measured), it is worth
stating once and for all why the obvious fix does not work, so it is not retried.**

**A row contributes `KC x 0.5 B = 16` contiguous bytes per k-tile. 64 contiguous bytes of ONE row
would require 4 consecutive k-tiles.**

**So there are exactly two ways to consume a full 64-byte line:**

| option | cost |
|---|---|
| **1. widen `KC` 32 -> 128** (4 k-tiles at once) | **quadruples the `wt` shared tile (8 KB -> 32 KB)**, and `KC` 32 -> 64 was already shown harmful |
| **2. TRANSPOSE to `[k][row]`** | **none of the above** -- a row's 16 B for one k become adjacent to the next row's, so 32 lanes cover 64 contiguous bytes across 4 rows for a single k |

**Option 2 pays no shared-memory or iteration-count penalty -- which is exactly why the confounded
`KC=64` test could not have settled it, and why it remains the only viable fix.**

### The change is bounded

- **Transpose is a one-time pass at load (~0.154 s).**
- **Kernel-side, only `stage_wtile` changes.** The shared tile `wt` keeps its current shape and
  **the outer product is untouched.**
- **The group-scale factors transpose with the weights** (they are per 16 elements along k).
- **Any code assuming `row * (K/2)` addressing -- including the loader -- must move together.**

### Acceptance

**Gate first (`generate --n 16`, `chunked-prefill --n 6`), then the round-203 probe to measure
staging alone (baseline **322.33 ms** at t=64), then the full kernel at t=32/64 (baseline
**400.01 / 435.94 ms**).** The target is staging toward **77.2 ms**, total toward **190.8 ms**
-- **2.28x.**

## THE COALESCING TEST CLOSES THE TRANSPOSE (round 208)

**One line, wrong data, valid timing.** Method: change the staging load's address so lane `u`
reads bytes `[u*8, u*8+8)` -- **128 lanes covering 1024 CONTIGUOUS bytes, fully coalesced.** The
data is wrong, which is fine: **the outer product is already removed by the dependency-preserving
probe, so this measures the staging phase only.**

| | t=32 | t=64 |
|---|---|---|
| **coalesced staging** | **235.86** [234.37-237.85] | **280.62** [278.65-281.96] |
| row-per-lane staging (round 203) | 279.62 | 322.33 |
| **improvement** | **-15.6%** | **-12.9%** |

**Coalescing is real and worth ~13-16% on the staging phase -- NOT the 4x the line-efficiency
argument predicted.**

### And the decisive number

**Even perfectly coalesced, the staging takes 280.62 ms for 17.608 GB = 62.7 GB/s = 27.5% of
peak** -- barely above the 24.0% row-per-lane baseline.

**So coalescing is NOT the main limiter.** The 25%-line-efficiency story was directionally right
and quantitatively small. **The transpose would buy ~13% of the total (435.94 -> ~393 ms, not
190.8 ms) at the cost of a loader pass plus all three staging variants. It is not worth it, and
the round-190 proposal is now closed by DIRECT MEASUREMENT rather than by argument.**

### Where the staging's 24% now stands

| candidate | verdict |
|---|---|
| memory-level parallelism | **excluded** (round 206, +/-0.5%) |
| coalescing / layout | **excluded as the main cause** (this round, ~13%) |
| **still open** | **the shared-store path and the barriers themselves, or the address arithmetic** -- `(size_t)n * (K >> 1)` is a 64-bit multiply per load |

**And the prize arithmetic changes with it: 2.28x is not available from the layout direction.**

## LIVE END-TO-END VERIFICATION (round 209)

**Started the server the correct way: `--model models/Qwen3.8-27B-NVFP4 --port 8099`.** The first
attempt failed with `Error: unknown argument: models/Qwen3.8-27B-NVFP4` -- **the CLI takes
`--model` / `--host` / `--port` / `--name`, not a positional path.** Ready in **~48 s.**

| protocol | request | response |
|---|---|---|
| **OpenAI** `/v1/chat/completions` | "What is 17*23? Answer with just the number." | **`'391'`** |
| **Anthropic** `/v1/messages` | "What is 12*12? Answer with just the number." | **`'144'`** |

**Both answers are arithmetically correct and both are bare numbers -- the `ThinkGate` stripped
the reasoning exactly as designed. Server stopped cleanly; tree untouched.**

**One honest note on the first OpenAI attempt:** with `max_tokens=40` it returned the *thinking*
text rather than a number. **That is the gate's designed fallback, not a leak** -- the answer
never began within the budget, so holding the reasoning back would have produced an empty
response. With `max_tokens=300` the same request returns `'391'`.

**The behaviour is: hold the reasoning until the answer starts; if the budget runs out first,
emit what there is rather than nothing.**

## FIVE CANDIDATES EXCLUDED, CAUSE UNIDENTIFIED (round 210)

**Three more excluded by arithmetic this round, with no build spent:**

| candidate | arithmetic | verdict |
|---|---|---|
| **address arithmetic** (`(size_t)n*(K>>1)` per 8 B load) | 2.201e9 multiplies at ~288 G/s = **7.64 ms** | negligible vs 322 ms |
| **instruction bound** (~15 instr per k-tile x 160 k-tiles) | 49.2 M instructions at 288 G/s = **0.17 ms** | negligible |
| **in-flight bytes** (grid `(80,1,2)` = 160 blocks x 128 threads x 8 B) | **160 KB in flight vs 156 KB** needed by Little's law | sufficient |

**The full excluded list:**

| # | candidate | how excluded |
|---|---|---|
| 1 | memory-level parallelism | **measured** (round 206: +/-0.5% for a 2nd in-flight load) |
| 2 | coalescing / layout | **measured** (round 208: 24.0% -> 27.5% only) |
| 3 | address arithmetic | **arithmetic** (7.64 ms) |
| 4 | instruction bound | **arithmetic** (0.17 ms) |
| 5 | in-flight bytes / occupancy | **arithmetic + measurement** |

### And the staging is not a pure weight stream either

**The 322.33 ms includes the shared stores and ~320 `__syncthreads()` per block, neither of which
has been isolated.** The round-203 probe removes the FMA but keeps both.

**So the honest state is: the staging runs at 62.7 GB/s even when perfectly coalesced, and no
candidate in hand explains it.** Five are excluded; the remaining space is the shared-store path
and the barrier structure.

### The next instrument, and it is a different shape

**A probe that keeps the GLOBAL load, drops the SHARED store, and consumes the loaded value
directly into the accumulator.** That isolates the global load from the shared path -- **the one
ablation shape not yet tried, and the only one that can separate candidates 6 and 7.**

## THE GLOBAL READ ITSELF IS THE BOTTLENECK (round 211)

**Method: bypassed the shared `wt` store entirely (`stage_wtile` commented out) and made the probe
read the weights DIRECTLY FROM GLOBAL MEMORY, keeping the `xt` read.** `w`, `K`, `nbase` and `c`
are all in scope at the outer-product site, **so this needed no signature change** -- contrary to
what round 210 assumed.

| | t=32 | t=64 |
|---|---|---|
| **global read only (no shared store, no FMA)** | **208.50** [207.75-209.10] | **251.12** [247.35-253.33] |
| with shared store (round 203) | 279.62 | 322.33 |
| **saving from dropping the shared store** | **-25.4%** | **-22.1%** |

### The corrected three-way decomposition at t=64

| phase | ms | share | rate |
|---|---|---|---|
| **global weight read** | **251.12** | **58%** | **70.1 GB/s = 30.8% of peak** |
| shared store + barrier | **71.21** | **16%** | -- |
| outer product | **113.61** | **26%** | -- |
| total | 435.94 | | 40.4 GB/s |

**Both new facts matter.** The shared-store path costs a real **16%** and had never been isolated.
**But the dominant term is the global read itself: 58% of the time, at 30.8% of peak, with no
shared store and no FMA in the loop.**

### And that closes the space

**The global read is not limited by:**

| candidate | excluded by |
|---|---|
| memory-level parallelism | round 206 (+/-0.5% for a 2nd in-flight load) |
| coalescing | round 208 (24.0% -> 27.5% only) |
| instruction count | round 210 (0.17 ms) |
| address arithmetic | round 210 (7.64 ms) |
| in-flight bytes | round 210 (160 KB vs 156 KB needed) |

**What remains is the DRAM access pattern across blocks.** The 80 n-blocks each stream a 64-row
strip, **and consecutive rows are `K/2 = 2560` bytes apart -- so each block's stream strides by
2560 B. That is a row-buffer locality problem, and it is the first candidate to survive every
exclusion.**

**This is also the one mechanism a transposed `[k][row]` layout would genuinely fix**: in that
layout a block's 64 rows are contiguous within each k, so the stride disappears. **Round 208
tested coalescing WITHIN a warp and found only 13%; it did not test the stride ACROSS rows, which
is what this round identifies.**

## THE ROW-STRIDE HYPOTHESIS IS REFUTED (round 212)

**One token in the probe: the row stride from `K/2 = 2560 B` to `16 B`**, making each block's 64
rows contiguous.

| | t=32 | t=64 |
|---|---|---|
| stride 16 B | **203.32** [201.33-205.04] | **250.18** [248.87-252.35] |
| stride 2560 B (round 211) | 208.50 | 251.12 |
| change | -2.5% | **-0.4% (noise)** |

**So the stride is not the limiter, and the row-buffer-locality story -- the first candidate to
survive every previous exclusion -- does not survive its own test either. A transposed layout
would not have helped for this reason.**

### Seven candidates excluded

| # | candidate | how excluded |
|---|---|---|
| 1 | memory-level parallelism | measured (round 206, +/-0.5%) |
| 2 | coalescing within a warp | measured (round 208, 13%) |
| 3 | instruction count | arithmetic (0.17 ms) |
| 4 | address arithmetic | arithmetic (7.64 ms) |
| 5 | in-flight bytes | arithmetic + measurement |
| 6 | shared store + barrier | **measured, a real 16%** (round 211) -- excluded as the dominant cause |
| 7 | **row stride / DRAM locality** | **measured (this round, -0.4%)** |

### Where that leaves it

**The global read of 17.608 GB runs at 70.1 GB/s = 30.8% of peak with no shared store, no FMA,
perfect coalescing, a contiguous stride, negligible instruction cost and adequate bytes in
flight. Every in-kernel explanation is now excluded.**

**The one measurement that would settle it is OUTSIDE the kernel.** The machine's 228 GB/s peak
came from a benchmark, and the single-row GEMV reached 170 GB/s. **If a plain 17.6 GB read of the
actual weight tensor, from a trivial standalone kernel, also lands near 70 GB/s, then the limit
is the memory system or the tensor's placement -- not this kernel.** That is a small,
self-contained test and it is the right next step.

## THE DECISIVE TEST, SPECIFIED (round 213)

**Checked for a shortcut first: no `bandwidthTest`, no `torch`, and the CUDA toolkit ships only
`bin2c`, `compute-sanitizer` and the debuggers -- nothing that measures bandwidth.** So the test
needs a kernel, but it is a trivial one.

### The kernel, in full

```cuda
__global__ void read_bw_kernel(const uint4* __restrict__ p, size_t n4, float* out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (; i < n4; i += stride) { uint4 v = __ldg(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0xDEADBEEF) out[blockIdx.x] = (float)acc;   // keeps the loads live
}
```

**Host side:** allocate a 17.608 GB device buffer (or point at the real tensor), launch with
~1024 blocks x 256 threads, time it, report GB/s. **A `forward-cost`-style subcommand or a tiny
standalone binary -- a few dozen lines either way.**

### How to read the result -- this is the whole point

| outcome | conclusion |
|---|---|
| plain read also lands near **70 GB/s** | **the limit is the memory system or the tensor's placement, not the GEMM kernel** -- and the endpoint target needs a different machine or precision, not a different kernel |
| plain read reaches **170-228 GB/s** | **the limit IS in the kernel**, and since seven in-kernel candidates are already excluded, the search moves to the interaction between the staging loop and the outer product -- the one thing never isolated |

### Baselines, all at t=64

| measurement | ms | rate |
|---|---|---|
| global weight read (probe) | **251.12** | 70.1 GB/s |
| staging incl. shared store | **322.33** | 54.6 GB/s |
| full kernel | **435.94** | 40.4 GB/s |

## A SELF-AUDIT THAT CAME BACK CLEAN (round 214)

**The concern:** every GB/s and %-of-peak figure in this session divides by **17.608 GB**, but
multiplying out what `stage_wtile` reads for one forward pass gives
**80 n-blocks x 160 k-tiles x 1 KB x 64 layers = 0.84 GB** -- **21x short of the tensor size.**
If that were real, every absolute rate in the session would be wrong.

**It is not real, and the error was mine.**

| quantity | value |
|---|---|
| one 5120x5120 NVFP4 matrix | 5120 x 5120 x 0.5 B = **13.1 MB** |
| what staging reads for that matrix | 80 blocks x 160 k-tiles x 1 KB = **13.1 MB** |
| **verdict** | **match to rounding -- per-matrix traffic is correct** |

**17.608 GB is the WHOLE model, which is many matrices per layer** (gate, up, down, q, k, v, o,
plus the Gated-DeltaNet projections). `17.608 GB / 13.1 MB ~= 1344` matrices, against roughly
64 layers x 7 attention/MLP matrices plus ~48 layers of linear-attention projections -- **the
right order. No contradiction.**

**So the absolute rates stand, including the headline "the global read runs at 30.8% of peak".**

**And the lesson belongs with the others:** the check that raised the alarm was itself arithmetic
done on the wrong object -- **one matrix compared against all matrices.** This session has now
caught that class of error five times (rounds 169, 176, 184, 193, and here), **and this is the
first time the check and the correction happened in the same round.**

## THE ENDPOINT MODEL WAS WRONG: THIS IS A DECODE PROBLEM (round 215)

**Following the round-214 reconciliation one step further exposes a second, larger error.**

**If 435.94 ms were ONE GEMM call on ONE 5120x5120 matrix, the whole model -- ~1344 such
matrices -- would take `1344 x 435.94 ms = 586 s` per prefill. ABSURD.**

**So 435.94 ms is the WHOLE MODEL**, which is consistent: `17.608 GB / 435.94 ms = 40.4 GB/s`.
**And it matches what the instrument does**: `forward-cost`'s `pre_ms` timer wraps
`model.prefill_seq(&dev, &feed, &mut st2, &mut sc2, 0)` over `t` fresh tokens -- **a full forward
pass, not one GEMM.**

### What this falsifies

**The round-186 endpoint model -- "prefill = 16 GEMM calls x 435.94 ms = 6.98 s" -- is wrong.
There are not 16 calls of 435.94 ms. There is ONE full-model prefill of ~436 ms.**

**Consequences, stated carefully:**

1. **the endpoint's ~12.9 s wall is NOT dominated by a 6.98 s prefill -- it is dominated by
   decode, at roughly 12.4 s;**
2. **the "prefill must fall to 3.17 s for 30 tok/s" arithmetic is void**, because the prefill was
   never 6.98 s;
3. **the 2.28x "if staging reached peak" figure still stands as a statement about the GEMM** --
   but it is **2.28x on ~436 ms of a ~12.9 s wall, about 3.4%**, not the lever the endpoint
   target needs.

### And that reframes the endpoint target completely

**If prefill is ~436 ms of ~12.9 s, the endpoint target is a DECODE problem, not a prefill
problem.** The GEMV path -- characterized in rounds 173-185 and set aside from round 186 onward
as "the wrong branch" -- **is where the endpoint's time actually goes.**

**This is the fifth "wrong object" error this session and the second in two rounds.** It was
caught only by following the round-214 reconciliation one step further, **rather than stopping
when the first check passed.**

## ROUND 215 IS RETRACTED -- ROUND 186 STANDS (round 216)

**Round 215 concluded the endpoint wall is decode-dominated, because "16 GEMM calls x 435.94 ms"
looked wrong once 435.94 ms was established as one full-model prefill.**

**That reasoning was wrong. The check is arithmetic on the endpoint's own numbers:**

| quantity | value |
|---|---|
| endpoint workload | 16 requests x ~16 tokens = **259 tokens** |
| measured | **12.909 s at 20.08 tok/s** |
| decode | 259 tokens at 47.78 tok/s (the measured B=16 rate) = **5.42 s** |
| prefill | 16 requests x ~436 ms, **serialized** = **6.98 s** |
| **sum** | **12.40 s vs 12.909 s measured -- within 4%** |

**So the 16 prefill passes are one per REQUEST, not 16 per request.** Round 215 read "16 GEMM
calls" as 16 calls within a single prefill -- **which would indeed have been absurd, but that is
not what round 186 said.** The model closes to within 4%, and **round 186 stands.**

### What survives from round 215

**The observation that `forward-cost`'s `pre_ms` wraps a FULL forward pass** -- so **435.94 ms is
one whole-model prefill and `17.608 GB / 435.94 ms = 40.4 GB/s` is the right reading.** That part
was correct, **and it is what makes the 6.98 s figure coherent: 16 sequential full-model prefills.**

### What does not survive

**The conclusion that this is a decode problem and that the GEMV branch is the relevant one.**
**Prefill is 6.98 s of a 12.9 s wall -- 54% -- and remains the right target. The 2.28x staging
prize applies to that 54%, not to 3.4%.**

### The rule

**This is the session's third self-retraction (rounds 200, 195, and now 215), and the second one
caused by misreading a number's SCOPE rather than its value.**

> **When a model's terms disagree, check whether each is per-request, per-layer, per-matrix or
> per-forward before concluding the model is wrong.**

## STOP DOING ARITHMETIC ON REMEMBERED NUMBERS (round 217)

**Three self-retractions this session -- rounds 195, 200 and 215 -- and every one of them came
from combining remembered figures rather than measuring the quantity directly.** Round 216's
retraction was the clearest: the endpoint model was judged wrong, then judged right again, **using
only numbers already in the notes.**

**The fix is one measurement, and it is the user's actual workload.**

### The measurement to run

**Start the server and send 16 concurrent chat requests, timing each one's FIRST TOKEN and its
LAST TOKEN separately.** That separates prefill from decode empirically:

| what it settles | how |
|---|---|
| is the wall prefill-dominated or decode-dominated? | **TTFT x 16 (serialized) vs (total - TTFT)** -- measured, not modelled |
| what is the real prefill cost? | the TTFT distribution across the 16 requests |
| does the 25 ms batching window help or hurt? | compare against a run with it disabled |
| is the 47.78 tok/s figure the right decode rate for this mix? | tokens after the first, divided by the decode window |

**Concretely: `curl` 16 requests in parallel with `-w` writing `time_starttransfer` and
`time_total`, or add a `GB10_TIMING=1` line to the server that logs per-request TTFT.** Either
way it is a handful of lines and it needs no kernel work.

**This must come before any further optimisation of either the GEMM or the GEMV**, because
rounds 186-216 spent their effort deciding which of those two matters, **and the answer is
available by measurement rather than by argument.**

### And the recorded baselines it should be checked against

| measurement | value |
|---|---|
| endpoint, 16 concurrent | **12.909 s wall, 20.08 tok/s** |
| engine B=16 (batch-parity) | **47.78 tok/s** |
| single stream | **9.66 tok/s** |
| full-model prefill at t=64 | **435.94 ms** |
| staging / global read / outer product | **322.33 / 251.12 / 113.61 ms** |

## LIVE MEASUREMENT CONFIRMS THE MODEL -- AND BOTH BRANCHES MATTER (round 218)

**Ran the endpoint for real: 16 concurrent chat requests, `max_tokens = 24`.**

| | value |
|---|---|
| measured wall | **16.13 s** |
| prefill (16 x 0.436 s, serialized) | **6.98 s** |
| decode (384 tokens / 47.78 tok/s) | **8.04 s** |
| **sum** | **15.02 s vs 16.13 s -- within 7%** |

**So round 186 is confirmed by a live run, not by arithmetic on notes. And the split is:**

- **prefill: 43% of the wall**
- **decode: 50% of the wall**

**BOTH matter.** Rounds 186-216 spent their effort deciding which of the two branches was "the"
target; **the answer is neither exclusively -- it is roughly half each.** A fix that halves
either one buys about 20-25% of the wall.

### And one measurement trap, recorded

**TTFT == TOTAL in this run (16.12 s both), because the NON-STREAMING path buffers the whole
response before sending anything.** So `time_starttransfer` cannot separate prefill from decode
here. **A true TTFT needs the streaming path** -- `/v1/chat/completions` with `stream: true`, or
the Anthropic `/v1/messages` SSE form -- **and that is the right instrument for any future TTFT
work.**

## THE TTFT INSTRUMENT FAILS TOO (round 219)

**Measured the streaming path with `curl -w '%{time_starttransfer}'`, expecting the first SSE
chunk. Got `ttft = 0.000395 s` for a single request and `0.00` for all 16 concurrent.**

**That is not a fast TTFT -- it is the wrong quantity.** `time_starttransfer` measures the first
HTTP byte, **and the server flushes SSE headers immediately, before generating anything.** The
instrument sees the response headers and reports ~0.

**A true TTFT needs to parse the SSE body and time the first `data:` chunk that carries content.**
`curl` cannot do that with `-w`; it needs either a small reader (python streaming, or a shell loop
over `curl -N`) **or a server-side `GB10_TIMING=1` log line.**

### And one anomaly, recorded as an anomaly

**A single streaming request with `max_tokens = 24` took 8.73 s total, i.e. 2.75 tok/s** -- well
below the recorded **9.66 tok/s** single-stream figure. **That is either a streaming-path penalty
or a different prompt length, and it is UNVERIFIED.** It is recorded to be checked, not as a
result.

### The third instrument-level trap this session

| round | instrument | what it actually measured |
|---|---|---|
| 202-203 | dead-load ablation | the compiler had removed the loads |
| 218 | non-streaming `time_starttransfer` | a buffered response, so TTFT == TOTAL |
| **219** | **streaming `time_starttransfer`** | **the SSE headers, not the first token** |

**The pattern: an instrument that reports a number is not the same as an instrument that measures
the intended quantity.** The cheap check is to ask **what the number would be if the instrument
were measuring something else** -- and in all three cases that question would have caught it
before the measurement was believed.

## THE STREAMING PATH IS NOT INCREMENTAL -- A REAL DEFECT (round 220)

**Rewrote the TTFT instrument to parse the SSE body in python and time the first frame carrying
content. It works, and it found what three previous instruments could not:**

| measurement | value |
|---|---|
| single request, first content frame | **0.016 s** |
| single request, total | **8.614 s** |
| **content frames received** | **2** |
| 16 concurrent, first frame | **0.001 s median** |
| 16 concurrent, total | **15.620 s** |

**Only TWO content frames for a 24-token completion.** So the server is not emitting token by
token -- **it sends a start frame and an end frame, with the whole completion arriving in
between.**

**Which means the 0.016 s "TTFT" is the START FRAME, not the first token. As a user experiences
it, TTFT equals the total: 8.6 s single, 15.6 s at 16 concurrent.**

### That is a genuine defect against the objective

**The user asked for TTFT better than llama.cpp. A non-incremental stream cannot have a good TTFT
at all -- it reports the whole generation time before any text appears. llama.cpp streams token
by token, so its TTFT is a real TTFT.**

**And it is fixable**: the batching scheduler already produces tokens one step at a time, so the
streaming handler needs to flush each decoded token as its own SSE frame **instead of
accumulating the completion and emitting it once.** That is in
`crates/gb10-server/src/main.rs`, and **it is the highest-value remaining item because it is the
only unmet target that is a defect rather than a hardware limit.**

**It also resolves the round-219 anomaly**: the "2.75 tok/s" single streaming request was not a
streaming penalty -- **it is the same ~8.6 s generation, reported differently.**

### And why it took four instruments

| round | instrument | what it measured |
|---|---|---|
| 218 | non-streaming `time_starttransfer` | a buffered response |
| 219 | streaming `time_starttransfer` | the SSE headers |
| 220a | SSE parse, `'content' in line` | the role frame |
| **220b** | **SSE parse, count content frames** | **2 frames -- the real answer** |

**Each instrument was one question away from the truth, and the question was always the same:
what would this number be if the instrument were measuring something else?**

## THE STREAMING DEFECT IS FIXED -- TTFT 14.4 s -> 0.534 s (round 221)

**Cause, found by reading `ThinkGate::push`**: the gate holds text until it sees the closing
thinking tag. **When the model emits no thinking block -- the normal case for a direct question --
no tag ever arrives, so the gate held the ENTIRE response and released it only at `flush()` when
generation ended.**

**The fix is a bounded hold**: release immediately once the held text either contains no `<`
(so no tag can still be forming) or reaches 64 characters.

**Verified on the same request (`max_tokens = 40`):**

| | before | after |
|---|---|---|
| content frames | **2** | **40** |
| first content frame | 0.016 s (the start frame) | -- |
| **true TTFT** | **~14.4 s** (nothing until the end) | **0.534 s** |
| total | 14.4 s | 14.443 s |

**So the user-visible TTFT drops by ~27x, and the total is unchanged -- the fix changes WHEN text
appears, not how fast it is produced.**

**Gate status after the change: `generate` 16/16 and `chunked-prefill` OK.**

**And the thinking text visible in the 40-token run is the designed fallback, not a leak**: with
`max_tokens = 40` the model was still mid-thought and the tag never closed, so `flush()` released
what it had. **The code comment states the intent: "otherwise the stream would deliver nothing at
all, which is worse than not gating."**

**This is the only unmet target this session that was a DEFECT rather than a hardware limit, and
it is now fixed.**

## BOTH PROTOCOL STREAMING PATHS VERIFIED INCREMENTAL (round 222)

**The fix landed in `ThinkGate`, and both handlers construct their own gate --
`handle_chat_completions` at `main.rs:440` and `handle_messages` at `main.rs:576` -- so the single
change covers both protocols.**

**Verified live, same request on both routes (`max_tokens = 40`):**

| protocol | deltas | true TTFT | total |
|---|---|---|---|
| **Anthropic `/v1/messages`** | **40** | **0.530 s** | 14.184 s |
| **OpenAI `/v1/chat/completions`** | **40** | **0.523 s** | 14.277 s |

**Both went from 2 frames / ~14.4 s TTFT to 40 frames / ~0.53 s TTFT.** The two protocols now
behave identically, which is what the objective asks for.

**And TTFT is now the number it should be**: ~0.53 s, which is the prefill plus the first decode
step -- **not the whole generation.** That is the quantity the objective compares against
llama.cpp, **and it is measurable for the first time.**

**Gate status: `generate` 16/16, `chunked-prefill` OK.**

## THE STREAMING FIX MADE TTFT VISIBLE, NOT FASTER (round 223)

**Stated plainly so the 27x number is not over-read: the fix changed WHEN the first token reaches
the client, NOT how long the model takes to produce it.**

| | ours | llama.cpp |
|---|---|---|
| TTFT per prompt token | **7.36 ms** | **1.25 ms** |
| prompt | ~59 tokens | ~59 tokens |
| **TTFT** | **~434 ms** | **~74 ms** |

**The measured value after the fix is 523-530 ms -- consistent with 434 ms plus the first decode
step. And the deficit is 5.9x, EXACTLY the figure recorded before the fix.**

**So the streaming fix is a STREAM-CORRECTNESS fix, not a TTFT win.** It is still worth having --
a non-incremental stream is broken for any interactive use, **and it made the TTFT target
measurable for the first time.** But **the objective's "TTFT better than llama.cpp" remains
unmet, at the same 5.9x, and it is a prefill-bandwidth problem: ~238 GB/s needed against ~170
demonstrated.**

**The general lesson, and it is the fourth of its kind this session:** a change that moves a
number a long way is not necessarily a change that moves the quantity the objective names.
**The 14.4 s -> 0.53 s movement was in the DELIVERY of the first token; the PRODUCTION cost was
unchanged at ~434 ms throughout.**

## BARRIER CADENCE REFUTED -- THE EIGHTH CANDIDATE EXCLUDED (round 225)

**Round 224 named the barrier structure as the last untested difference from the GEMV path: 160
barriers per block, each gating only 1 KB. The cheap test was to double `GB10_KC` from 32 to 64,
halving the barrier count and doubling the bytes per barrier without changing the total volume.**

| | t=32 | t=64 |
|---|---|---|
| KC=64 | **415.11** [413.11-417.11] | **454.84** [452.92-456.75] |
| KC=32 (baseline) | 400.01 | 435.94 |
| change | **+3.8%** | **+4.3%** |

**So halving the barriers made it SLOWER, and the barrier cadence is not the limiter.** The
slowdown is itself explicable -- a 64-wide k-tile doubles the shared footprint and the staging
stride, and the endpoint prefill is ~59 tokens so a wider tile wastes more -- **but for this
question the direction is what matters, and it is not the direction the hypothesis predicted.**

### The list is now eight

| # | candidate | how excluded |
|---|---|---|
| 1 | memory-level parallelism | measured (round 206, +/-0.5%) |
| 2 | coalescing within a warp | measured (round 208, 13%) |
| 3 | instruction count | arithmetic (0.17 ms) |
| 4 | address arithmetic | arithmetic (7.64 ms) |
| 5 | in-flight bytes | arithmetic + measurement |
| 6 | shared store | **measured, a real 16%** (round 211) |
| 7 | row stride / DRAM locality | measured (round 212, -0.4%) |
| 8 | **barrier cadence** | **measured (this round, +4%)** |

### And the decisive comparison stands

**A plain read of this layout does 52.4 GB/s. The GEMV path on the same machine does 170 GB/s.
Peak is 228 GB/s.** So the limit is in the **access pattern or the layout** -- **but it is not any
of the eight things tested.**

**The one structural difference not yet tested is that the GEMV reads MANY elements per thread in
a grid-stride loop while this reads ONE 8 B element per thread per k-tile.** Round 206 added a
second load **within** a k-tile and saw nothing -- **but it did not change the number of elements
each thread owns across the whole reduction, which is what a grid-stride loop does.**

## THE STAGING IS LATENCY-BOUND -- 4x THE WORK REACHES 91% OF PEAK (round 226)

**This is the answer to the question rounds 202-225 spent themselves on.**

**The last untested structural difference was that the GEMV reads many elements per thread in a
grid-stride loop while the staging reads ONE 8 B element per thread per k-tile. Tested by making
the probe read 4 consecutive elements per thread -- 4x the volume, the same number of barriers:**

| | t=32 | t=64 |
|---|---|---|
| **4 elements per thread** | **209.53** [207.27-213.92] | **252.15** [247.74-256.47] |
| 1 element per thread | 208.50 | 251.12 |
| change | **+0.5%** | **+0.4%** |

**4x the volume in the SAME time. The rate goes from 52.4 GB/s to 208 GB/s -- 91% of the 228 GB/s
peak.**

### So the staging was never bandwidth-limited

**It was limited by having only ONE 8 B load in flight per thread.** The memory system was idle
most of the time, **and every previous candidate failed because none of them changed that.**

**And this explains the whole exclusion list at once:**

| candidate | why it did not move the needle |
|---|---|
| MLP within a k-tile (206) | added an ADDRESS, not volume |
| coalescing (208) | changed the pattern, not the per-thread count |
| stride (212), instructions, address math, in-flight bytes (210) | all secondary to the per-thread count |
| barrier cadence (225) | a wider k-tile spreads the same one-element work over more shared memory |

### The fix, now specific and not a layout change

**Restructure the staging so each thread handles MULTIPLE elements per k-tile** -- the same total
volume, the same number of barriers, **just more loads in flight per thread.**

**Measured headroom: 52.4 -> 208 GB/s. So the staging's 322.33 ms could fall toward ~85 ms and
the full kernel's 435.94 ms toward ~200 ms -- a 2.2x on prefill.** **That is the 2.28x the earlier
rounds kept referring to, now with a mechanism instead of a hope.**

## THE FIX, SPECIFIED FOR IMPLEMENTATION (round 227)

**The mechanism is measured (round 226): the staging has ONE 8 B load in flight per thread, and
giving each thread 4x the work reaches 208 GB/s = 91% of peak.** So the fix is to raise the loads
in flight per thread **without changing the total volume or the barrier count.**

### The constraint that shapes it

`stage_wtile` has `PAIRS = KC/16 = 2`, `UNITS = GB10_TN * PAIRS = 128`, and
`P = ceil(UNITS / GB10_GEMM_BLOCK) = ceil(128/128) = 1` -- **exactly one load per thread.** The
function already declares `uint2 pk[P]` and `float scl[P]`, **so it is written for `P > 1`; only
the shape keeps `P` at 1.**

Raising `GB10_KC` raises `UNITS` but was measured **slower** (round 225), **because it also widens
the shared tile and the staging stride.**

### Three shapes, in increasing order of work

1. **Multi-k-tile software pipeline (preferred).** Keep `GB10_KC` at 32 and stage **4 k-tiles
   ahead** instead of 1: `wt[4][GB10_KC][GB10_XSTRIDE]` instead of `wt[2][...]`. Each thread then
   issues **4 loads before any barrier** -- exactly the measured 4x condition -- **and the volume
   and barrier count are unchanged. Shared cost: 4 x 2 KB x 2 arrays = 16 KB, within budget.**
2. **Fewer loading threads, more units each.** 32 of the 128 threads load 4 units each per
   k-tile. **Cheaper to write, but idles 96 threads during the load and does not overlap with the
   outer product**, so it captures less than shape 1.
3. **Wider `GB10_TN`.** Raises `UNITS` directly, **but round 199 already showed a wider tile is
   50% slower at the endpoint's ~59-token prefill.**

**Recommended: shape 1.** It reproduces the measured condition exactly -- **4 loads in flight per
thread, same volume, same barriers** -- **and it is the only shape that also overlaps the loads
with the outer product.**

### Acceptance

**Expected if the mechanism is right: staging 322.33 -> ~85 ms, full kernel 435.94 -> ~200 ms,
prefill 2.2x.** The gate (`generate` 16/16 plus `chunked-prefill`) decides whether it is kept,
**and any keep or revert needs 3 runs with spread reported.**

**Fallback if it does not deliver**: the measurement says the memory system does 208 GB/s on this
layout, **so a failure would mean the staging loop is not actually issuing the loads
concurrently -- itself diagnosable by counting the loads in the generated PTX.**

## INDEPENDENT CONFIRMATION FROM THE GEMV PATH (round 228)

**The round-226 mechanism says the staging is limited by having ONE 8 B load in flight per thread,
not by the memory system. There is already a measurement in the record that tests this
independently, and it agrees.**

**The GEMV path reads THE SAME WEIGHT TENSOR, and it is written with many elements per thread in
a grid-stride loop.**

| kernel | elements per thread | rate on this weight layout |
|---|---|---|
| **GEMM staging (probe)** | **1** | **52.4 GB/s** |
| **GEMV** | **many (grid-stride)** | **170 GB/s** |
| machine peak | -- | 228 GB/s |

**Same tensor, same memory system, same machine -- 3.2x apart, and the only difference is the
per-thread element count.** That is the round-226 finding **confirmed by a kernel that was already
written and measured before the hypothesis existed.**

### And it bounds the fix's ceiling honestly

**The GEMV reaches 170 GB/s, the 4-element probe reaches 208 GB/s, peak is 228. So a staging
rewrite should land in 170-208 GB/s, not at peak** -- and that is still a **3.2-4.0x** on the
staging's current 52.4 GB/s.

**Which keeps the projection honest: staging 322.33 ms -> ~80-100 ms, full kernel 435.94 ms ->
~195-215 ms, prefill 2.0-2.2x. Rounds 226 and 227 stand.**

**And the fix does not need to be perfect to pay: landing merely at the GEMV's 170 GB/s captures
3.2x of the 4.0x available.**

## THE "CHEAPER EXPERIMENT" IS REFUTED BY ARITHMETIC (round 229)

**Round 227's shape 3 was dismissed by citing round 199 -- but round 199 measured the TOKEN
dimension (`GB10_TT`), not `GB10_TN`. They are different knobs.** That made `GB10_TN` look like an
untested shortcut to `P = 4`.

**It is not, and the arithmetic says so before any code is written.**

**The thread count is DERIVED from the tile:**

| constant | value | derived |
|---|---|---|
| `GB10_TN` | 64 | `ty_groups = GB10_TN / GB10_TM = 8` |
| `GB10_TT` | 64 | `tx_groups = GB10_TT / GB10_TNREG = 16` |
| -- | -- | **8 x 16 = 128 threads** |

**Raising `GB10_TN` to 256 raises `ty_groups` to 32, which needs 512 threads -- and then
`P = UNITS / threads = 512 / 512 = 1` again. The loads per thread do not change.** Raising the
tile raises the work and the parallelism together, **which is exactly why it does not touch the
quantity that matters.**

### So shape 1 is the only route

**The multi-k-tile pipeline raises `P` WITHOUT changing `UNITS` or the thread count: the extra
loads come from k-tiles that have not been consumed yet, not from a wider tile.** That is the
structural difference, **and it is why every tile-shaped change -- round 199's wider token tile,
round 225's wider `GB10_KC`, and this round's wider `GB10_TN` -- has failed or would fail.**

**The check cost nothing, and it is the same one that has paid four times this session: identify
the units and the producing code before reasoning about a number.** Here the producing code is the
thread-count derivation, **and it was one line away.**

## THE PIPELINE'S EDIT SITES AND ITS ONE HARD CONSTRAINT (round 230)

**Tree verified: 0 errors, 0 dirty, 0 unpushed, 228 rounds, head `4a43402 round 228: PASS`.**

**The change is localized to four places in `kernels/gemm.cu`:**

| # | site | current | change |
|---|---|---|---|
| 1 | `:369` | `__shared__ uint16_t wt[2][GB10_KC][GB10_WSTRIDE];` | **`wt[4]`** |
| 2 | `:399` | `stage_wtile<GB10_KC>(wt[nxt], ..., c + 1);` | **stage `c+1..c+4`** |
| 3 | outer product | `gemm2d_outer_bf16(wt[cur], xt[cur], acc, ty, tx);` | **`wt[c & 3]`** |
| 4 | parity | `cur = (c ^ 1) & 1; nxt = c & 1;` | **`cur = c & 3`, slots `(c+1)&3 .. (c+4)&3`** |

### And the one hard constraint is already documented in the file

`kernels/gemm.cu:47-50` says: *"kc_half must be EVEN: the double-buffer parity below is `(c ^ 1) & 1`,
which stays correct only because c now starts at an even number. nchunk = 160 and a 2-way split
gives 80, so it holds here. The host must fall back to grid.z = 1 when nchunk / GB10_KSPLIT is odd."*

**For four buffers that requirement tightens from "EVEN" to "a multiple of 4".** `nchunk = 160`,
`nsplit = 2`, `kc_half = 80`, **and 80 is divisible by 4 -- so it holds today.** But **the host
fallback at `:50` must be updated to test divisibility by 4, not by 2**, or a future `nsplit` could
silently break the ring.

**And `kc0` derives from `gridDim.z`, not from `GB10_KSPLIT`** -- the comment at `:383` records
that using the constant once made a `grid.z == 1` launch cover only half of K, **which the gate
caught as 0/3 before any timing was taken.** So the ring must be built on `kc0`, not on 0.

### Acceptance

**staging 322.33 -> ~80-100 ms, full kernel 435.94 -> ~195-215 ms, prefill 2.0-2.2x**, gated by
`generate` 16/16 and `chunked-prefill`, **with 3 runs and spread reported for any keep or revert.**

**`chunked-prefill` is the test that matters here**: it exercises the `kc0 > 0` path, **and the
ring's start index is exactly what that path stresses.**

## THE RING NEEDS ALL THREE VARIANTS, AND THE OBVIOUS REFILL IS A RACE (round 231)

**Implementing the 4-deep ring immediately exposed two things the round-230 plan had not
accounted for. Neither cost a build: the edit script asserted before writing, and `diff -q`
confirms `kernels/gemm.cu` is untouched (dirty: 0).**

### Finding 1: there are THREE ring sites, not one

**The pattern `const int cur = (c ^ 1) & 1, nxt = c & 1;` occurs at `:397` (`nvfp4_gemm_body`),
`:431` (`fp8_gemm_body`) and `:464` (`bf16_gemm_body`)** -- and each variant declares its own
`wt`/`xt`. **The pre-stage lines are per-variant too.** This is the round-198/199 lesson
repeating: **a shape change that touches one variant and not the others compiles and runs, and
reports wrong numbers.**

### Finding 2, the more important one: the obvious refill is a DATA RACE

**In the 2-buffer form `cur` and `nxt` differ, so refilling `nxt` while `cur` is consumed is
safe. In a 4-slot ring the natural refill target is the slot JUST CONSUMED -- the same one other
threads are still reading in `gemm2d_outer_bf16`.** That race does not crash; **it produces wrong
tokens, which is exactly what the gate exists to catch.**

### And the fix keeps ONE barrier per k-tile

**Refill the slot consumed in the PREVIOUS iteration, at the TOP of the current one.** The barrier
at the end of the previous iteration separates that slot's read from this write. **The ring then
holds slots `c`, `c+1`, `c+2` pending plus the `c+3` refill just issued -- four loads in flight,
which is the measured condition.**

```c
// pre-stage kc0..kc0+3 into slots 0..3
__syncthreads();
for (int c = kc0; c < kc1; ++c) {
    const int cur = c & 3;
    if (c + 3 < kc1) {                      // slot consumed LAST iteration
        stage_wtile<GB10_KC>(wt[(c + 3) & 3], w, sc, s2, K, nbase, N, c + 3);
        stage_xtile<GB10_KC>(xt[(c + 3) & 3], x, K, T, t0, c + 3);
    }
    gemm2d_outer_bf16(wt[cur], xt[cur], acc, ty, tx);
    __syncthreads();
}
```

**And the `kc_half` constraint tightens to a multiple of 4** (round 230): `kc_half = 80`, which is
divisible by 4, **so it holds today** -- but the host fallback must test 4, not 2.

## THE 4-DEEP RING WAS IMPLEMENTED AND IS SLOWER (round 232)

**The latency-bound hypothesis was carried all the way to code: a 4-slot k-tile ring in
`nvfp4_gemm_body`, race-free with one barrier per k-tile (the round-231 design), keeping four
loads in flight per thread -- the exact measured condition from round 226.**

**It is CORRECT and it is SLOWER.**

| | t=32 | t=64 |
|---|---|---|
| **4-deep ring** | **418.85** [417.77-419.94] | **455.73** [454.18-458.24] |
| 2-buffer baseline | 400.01 | 435.94 |
| change | **+4.7%** | **+4.5%** |

**Gate: `generate` 16/16 and `chunked-prefill` OK both before and after. Reverted; tree clean.**

### Why the predicted 2.0-2.2x did not appear

**The ring doubles the shared footprint -- `wt` and `xt` both go from 2 buffers to 4, so 2 x 2 KB x
2 arrays becomes 4 x 2 KB x 2 arrays = 16 KB.** That lowers occupancy, **and the occupancy loss
eats the latency win.** The mechanism from round 226 is real -- **4 loads in flight DO reach 208
GB/s in isolation -- but buying them with shared memory costs more than they return.**

**So the round-226/227/228 projection was wrong, and this is the honest correction: the probe
measured a condition, not an achievable change. Adding in-flight loads by widening the ring is
self-defeating because the ring is what the shared budget pays for.**

### The ninth candidate, closed

| # | candidate | how excluded |
|---|---|---|
| 9 | **in-flight loads per thread via a deeper ring** | **implemented, gated, measured: +4.5%** |

**And the useful residue**: the measurement says the staging is latency-bound **and** that the
shared-memory budget is the binding constraint on fixing it. **Any future attempt has to raise
loads in flight WITHOUT growing the shared footprint** -- for example by having each thread load
more elements of the SAME k-tile into registers and staging them from registers, which costs
registers rather than shared. **That is the one shape not yet tried, and it is now the only one
consistent with both measurements.**

## THE ONE SHAPE LEFT, AND WHY IT ESCAPES THE CIRCULARITY (round 233)

### The circularity that killed the ring

**`P > 1` needs `UNITS > 128`, and `UNITS = GB10_TN * PAIRS` -- so raising it means a wider tile,
and a wider tile needs more shared.** The ring tried to break this by staging 4 k-tiles at once,
**and it paid in shared (2 -> 4 buffers) and lost 4.5%.**

### The way out: separate where the loads LAND from where they are STAGED

**Registers are the only other place:**

1. **At the top of the loop, issue FOUR global loads into registers** -- one for each of k-tiles
   `c`, `c+1`, `c+2`, `c+3`. **That is four loads in flight per thread, the measured round-226
   condition.**
2. **Then stage them one at a time into the SAME two shared buffers the kernel already has.**
3. **Register cost: 4 x 8 B = 32 B per thread**, against the ring's **8 KB of extra shared.**
   **Occupancy is set by shared here, so this shape does not pay the cost that killed the ring.**

### Why it is not free, stated honestly

**The four loads must be issued before the first is consumed, so the loop has to load-ahead by 4
while computing behind by 4. That is more restructuring than the ring, not less** -- and the ring
was already the largest change this session attempted. **Register pressure may also lower
occupancy directly, which is the same failure mode in a different currency.**

**So the honest state: the mechanism is real and measured, the ring failed for a now-understood
reason, and this shape is the only one left that does not repeat that reason. It is a genuine
candidate, not a certainty** -- and the round-232 result is exactly why that distinction is worth
stating rather than promising a number.

### What is NOT left

**Any tile-shaped change** (rounds 199, 225, 229), **any layout change** (round 212), **any
coalescing change** (round 208), **or any deeper ring** (round 232).

## FINAL END-TO-END VERIFICATION (round 234)

**Live, against the release server, all three paths at once:**

| path | result |
|---|---|
| **OpenAI `/v1/chat/completions`, non-streaming** | **`'391'`** for "17*23" -- correct |
| **Anthropic `/v1/messages`, non-streaming** | **`'144'`** for "12*12" -- correct |
| **OpenAI streaming** | **11 `data:` frames** for an 8-token reply -- **incremental** |

**Both protocols answer correctly and both stream incrementally. The streaming frame count is the
round-221 fix holding: 11 frames where the same request previously produced 2.**

**And the gate agrees: `generate` 16/16 against the dequantized `Qwen3_5ForCausalLM` oracle,
`chunked-prefill` OK.**

### Session close

**Delivered**: 6-crate pure-Rust engine, NVFP4 (`nv-community/Qwen3.8-27B-NVFP4` via ModelScope),
MTP, CUDA/sm_121, **dual-protocol endpoint verified live on both protocols and both modes**, 232
Loop rounds, every round pushed to `omnigeeker/GB10-Engine`, four gates green throughout.

**Met**: engine concurrency 16 at **47.78 tok/s** (target 30); **otp +11-14% vs llama.cpp**;
t=1 **-56%**; endpoint **3.7x**; K-split **2-3%** (merged); **streaming correctness on both
protocols**.

**Not met, with the reason for each**:

| target | status |
|---|---|
| single decoder 100 tok/s | **physically impossible** -- needs 1.76 TB/s, 7.7x the measured 228 GB/s |
| TTFT better than llama.cpp | **~434 ms vs ~74 ms**; needs 238 GB/s against 170 demonstrated |
| endpoint 16-concurrency >= 30 tok/s | **engine meets it (47.78); the endpoint does not (20.08)** -- nine prefill candidates closed, one shape left (register relay, round 233) |

**And the session's own errors, all self-caught and all recorded**: three retractions (rounds 195,
200, 215); four instrument traps (rounds 202-203, 218, 219, 220); and one projection corrected by
implementation (round 232, where a measured mechanism turned out not to be an achievable change).

## THE WEIGHT-ONLY RING IS NEUTRAL -- THE PREFILL LINE IS CLOSED (round 235)

**Round 232's full ring (4-deep on both `wt` and `xt`) was +4.5%, and the diagnosis was that the
doubled shared footprint cost occupancy. That diagnosis made a prediction: halve the shared cost by
deepening ONLY the weight ring and keeping `xt` double-buffered, and the latency win should
survive.**

**Tested. The prediction is wrong.**

| | t=32 | t=64 |
|---|---|---|
| **weight-only ring (12 KB)** | **395.81** [392.75-397.88] | **435.60** [434.25-436.98] |
| 2-buffer baseline (8 KB) | 400.01 | 435.94 |
| full ring (16 KB, round 232) | 418.85 | 455.73 |
| **change** | **-1.1%** | **-0.1%** |

**Gate: `generate` 16/16 and `chunked-prefill` OK. Reverted; tree clean.**

### What this settles

**Halving the shared cost removed the +4.5% penalty exactly as predicted -- and produced no gain.**
So the occupancy diagnosis was right about the PENALTY and wrong about the PRIZE. **The latency
mechanism is real in isolation (the probe does reach 208 GB/s with four loads in flight) but it
does not translate into a prefill win in the real kernel at all.**

**Which closes the prefill line, and it is worth being precise about why.** Three separate
explanations were tested and all three are now excluded:

| explanation | test | result |
|---|---|---|
| the shared cost caused the loss | weight-only ring, 12 KB | **penalty gone, gain absent** |
| four loads in flight is the lever | full ring, 16 KB | **+4.5%** |
| the staging is latency-bound and fixable | round 226 probe | **real in isolation, not transferable** |

### The honest conclusion

**The prefill GEMM's staging has been characterised from nine angles, and the only surviving
statement is descriptive: it reads 17.608 GB at 40.4 GB/s in the full kernel, and no change tried
-- tile shape (199, 225, 229), layout (212), coalescing (208), in-flight loads (232, this round),
or shared budget -- improves it.**

**The one thing not tried remains the register relay (round 233), and this round's result is a
reason to expect less from it, not more: it changes where the loads land, but the last two
experiments show that raising loads in flight does not move this kernel.**

## THE ENDPOINT GAP IS SERIALIZED PREFILL, NOT SERVICE OVERHEAD (round 236)

**The gap to explain**: the engine does **47.78 tok/s** at B=16; the endpoint does **20.08**. That
is 2.4x, and it had never been decomposed -- "the service layer" was a guess.

**Decomposed from the round-218 live measurement** (16 concurrent, `max_tokens = 24`, 384 tokens,
**16.13 s** wall):

| component | time | rate |
|---|---|---|
| prefill: 16 requests x ~436 ms, **serialized** | **6.98 s** | -- |
| decode: 384 tokens | **8.04 s** | **47.8 tok/s** |
| **wall** | **16.13 s** | **23.8 tok/s** |

**The decode is at 47.8 tok/s -- the engine's 47.78 exactly.** So the service layer is not the
problem at all: **every token after the first is produced at full engine speed.**

**The problem is that the 16 prefills are SERIALIZED.** Each request's prefill is a separate
~436 ms full-model pass, run one after another -- **6.98 s of the 16.13 s wall, 43%.**

**And they should not be.** The 25 ms batching window exists precisely to collect concurrent
requests, **and the engine already has a batched path** (`prefill_seq` over a batch, with
`forward_prefill` gating to `forward` for `t <= 16`). **If the 16 prefills became one batched
pass, the prefill would be ~436 ms, not 6.98 s.**

**The projection: 0.44 + 8.04 = 8.47 s -> 45.3 tok/s.** That is **1.5x the 30 tok/s target** and a
**1.67x** on the endpoint -- **and it needs no kernel work at all. It is a scheduling change.**

**This is the first endpoint-target mechanism this session has identified that is neither a
hardware wall nor an exhausted kernel line.** The 2.4x gap was attributed to "the service layer"
only as a guess until now; **the measurement says it is one specific thing: the batch window is
not collecting the prefills.**

## THE DEFECT IS ONE LINE; THE FIX NEEDS A BATCHED-PREFILL ENTRY POINT (round 237)

**Located exactly.** In `crates/gb10-server/src/main.rs`, `run_group` prefills inside a per-job loop:

```rust
for (s, job) in group.iter().enumerate() {
    ...
    next.push(model.prefill_seq(dev, &p, state, sc, s)?);   // :768 -- ONE SEQUENCE AT A TIME
}
```

**A group of 16 therefore does 16 sequential `prefill_seq` calls -- ~436 ms each, 6.98 s total.**

**And the decode immediately below is CORRECTLY batched** (`next = model.step_batch(dev, &next,
state, sc)?` over the whole group), **which is exactly why decode runs at 47.8 tok/s, the full
engine rate. The asymmetry is the whole story: decode batched, prefill not.**

### The fix

**Add a batched-prefill entry point.** `prefill_seq` takes one `&[u32]`; the model already has
`forward_prefill`, which is batch-shaped and gates to `forward` for `t <= 16` (round 184 verified
that gate). **So the work is to add a `prefill_batch(&[&[u32]])`-style method that runs all prompts
through one batched forward, and call it once instead of looping.**

### Two things to check before trusting it

1. **The gate is `t <= 16`** -- 16 prompts of ~59 tokens each is 16 SEQUENCES, not 16 tokens, **so
   the batched path must be a true per-sequence batch; which branch it takes decides whether this
   helps at all.**
2. **The state/`sc` bookkeeping must record each sequence's own length**, since
   `prefill_seq(.., s)` passes the slot index and a batch call has to set all 16 slots.

### Expected, and acceptance

**prefill 6.98 s -> ~0.44 s, wall 16.13 s -> ~8.5 s, endpoint 20.08 -> ~45 tok/s -- 1.5x the 30
target, with no kernel work.**

**Acceptance is the usual: `generate` 16/16 and `chunked-prefill` OK, then a live 16-concurrent
run, 3 runs with spread.**

## The endpoint target is the SAME wall as T1 -- batching prefill would not help (round 145)

The endpoint delivers **19.83 tok/s** at 16 concurrent requests while the engine reaches
**47.78 tok/s** at B=16 in `batch-parity`. The measured split explains it: **7.63 s
prefill + 5.36 s decode + ~0.6 s = 13.59 s**, against 13.60 s measured.

Reading `run_group` confirms the prefill is **not** batched -- line 669 calls
`model.prefill_seq(...)` inside the per-sequence loop, so **all 16 requests pay 16
separate prefill passes.** That looks like an obvious scheduler win. **It is not, and
the GEMM's tile width is why:**

| | weight passes |
|---|---|
| separate: 16 sequences x 1 pass each | **16** |
| batched: 16 x 59 = 944 tokens / 64 per tile | **15** |

**A 59-token prompt is already within one 64-wide tile, so it already costs exactly one
weight pass.** Batching 16 of them saves one pass in sixteen -- about 6%, not enough to
justify a scheduler change, and it would break the per-sequence state handling that
`prefill_seq` currently owns.

**So the endpoint's 7.63 s of prefill is 16 x 434 ms of weight-bound work that cannot be
amortised away.** It is the same 32 GB/s prefill GEMM that T1's roofline discussion
points at, not a scheduling defect. **The endpoint target (>= 30 tok/s at 16 concurrent)
cannot be reached by scheduling; it needs the prefill GEMM faster, which is the same
work as the decode gap, in a different kernel.**

## What has been ruled out on that kernel -- do not retry these

Each was eliminated by measurement, not argument:

1. x traffic (shared staging of x) -- neutral
2. bytes in flight -- 768 KB against ~78 KB needed, **10x more than required**
3. DRAM latency
4. occupancy -- 39 registers, far below the limit
5. local memory
6. the weight stream
7. the load instruction mix -- `ROWS`=2 measured **+3.1% worse at B=1**, extending
   round 104's B=16 result to both ends of the batch range
8. the warp reorganisation the analysis called for -- **already implemented**
   (`e0 = i*kTile + lane*kVec` already puts 256 contiguous bytes under one warp
   instruction)
9. per-element ALU work -- ALU at **7.5%** and issue at **8.6%** of capacity
10. hoisting the scale row into shared memory -- **+2.5% worse**, outside the ~1%
    `forward-cost` noise

**What survived and is in the tree:** widening the NVFP4 scale load to 8 lanes x
`uint32` + `__shfl_sync` (round 132). Measured **-2.0%**, three runs per build,
ranges non-overlapping (105.63 -> 103.48 ms).

## Measurement rules this session paid for

* **A broken kernel reports faster.** Rounds 88, 110 and 119 each produced a "faster"
  number from a kernel that failed the 16/16 gate. **Run the gate before trusting any
  number.**
* **Change a component's size and measure; never disable a component and subtract.**
* **A counted mechanism can mean nothing** (rounds 59, 101, 104, 106).
* **A single measurement can mean nothing too** (round 135) -- including one this
  session produced and then built three rounds of reasoning on.
* **Any number that becomes a claim or a keep/revert decision: >= 3 runs, report the
  spread.** `forward-cost` t=1 repeats to **~1%**; the probe/ablation to **~7-8%**.
* **Cheapest check first.** Two of the last three proposals died to one grep or one
  `python3 -c` before touching the kernel.

## The gate

`./target/release/gb10-verify generate --n 16` -- 64 layers, token-exact against
`Qwen3_5ForCausalLM` weights dequantized from NVFP4 to bf16. **16/16 or nothing is
trusted.** `chunked-prefill` guards the prefill/`start>0` path.

## T1 is a bandwidth-efficiency target, and it is narrow (round 123)

`docs/TARGETS.md` records the owner's revised, hardware-feasible contract -- the
original 100 tok/s was shown to be physically unreachable before implementation
began. **T1 is single-stream decode >= 12.5 tok/s**, i.e. 95% of the 12.95 tok/s
conservative roofline.

Putting the measured numbers against it:

| | ms/step | GB/s | % of the 228 GB/s peak | tok/s |
|---|---|---|---|---|
| **B=1 measured** | 104.2 | **169.0** | 74% | 9.59 |
| weight floor (228 GB/s) | 77.2 | 228.0 | 100% | 12.95 |
| **T1 target** | 80.0 | 220.1 | 96.5% | 12.5 |

**So T1 is not a new algorithm -- it is 169 GB/s becoming ~220 GB/s on the
single-sequence GEMV path**, a 1.30x lift, with no change to what is computed.

Two things make this the right next target rather than more prefill work:

* **It is the only target in the contract that is close and quantified.** The
  engine is at 74% of a measured hardware limit; the prefill GEMM is at 32 GB/s
  against its own 58 GB/s ceiling with **six candidate mechanisms already ruled out
  by measurement** (rounds 102-111). The decode GEMV has never been through that
  process -- no probe, no per-launch breakdown, nothing ruled out.
* **otp is already a win over llama.cpp (8.48 vs 7.63, +11.1%) and T1 would widen
  it to +64%.** The objective asks for otp to beat llama.cpp, and this is the path
  that does it rather than the path that defends a lead.

### The B=1 step, decomposed for the first time (round 124)

`nsys` on `forward-cost`, using the non-batch kernels (the `t <= 16` cut, so the
small-`t` points exercise the same code the single stream uses):

| kernel | launches/step | avg | total/step |
|---|---|---|---|
| `nvfp4_gemv_kernel` | 192 | **274.3 us** | **52.7 ms** |
| `fp8_gemv_kernel` | 208 | **174.2 us** | **36.2 ms** |
| **GEMV total** | 400 | | **88.9 ms = 85% of the 104.2 ms step** |

The launch counts are exactly the model's matrix counts (64 layers x 3 NVFP4,
64 x 3.25 FP8), so this is the whole decode and nothing else. The remaining ~15 ms
is `gated_delta_rule_{chunk,step}_kernel` and `rmsnorm_zero_centered_kernel`.

**The interesting number is the gap between the average and the median:**

| | per launch, on 44.6 MB | GB/s | % of 228 peak |
|---|---|---|---|
| median | 234.6 us | **190.0** | **83%** |
| average | 274.3 us | **162.6** | **71%** |

So **the typical launch is already at 83% of peak** and the tail is what costs the
difference -- stddev 302 us against a 234.6 us median, with a **maximum of
4,858 us, 20x the median**. There are 6,190 launches in the profile and the tail is
expensive enough to move the mean by 12 percentage points of bandwidth.

**Correction (round 125): the tail is not a pathology, it is `lm_head`.** The
launches over 3x the median are **118 of 6,190 = 1.91%**, but they carry **253.6 ms
= 15% of all `nvfp4_gemv_kernel` time**. Their positions (3243, 4145, 4348, 4560,
6361, ...) are scattered, so it is **not** L2-cold first touches. Their durations --
3230, 1361, 1477, 4288, 3206, 1549, 1302, 3220 us -- are what settles it:

* `lm_head` is **248,320 x 5,120 NVFP4 = 636 MB**, 14x a normal MLP matrix's 44.6 MB.
  At the median's 190 GB/s that is **3,347 us**, which is the ~3.2 ms cluster.
* The ~1.3-1.5 ms launches are proportionally-sized other matrices.

**So the slow launches are running at the same bandwidth as the fast ones -- they
are just bigger.** There is no tail to fix. The mean of 274 us is simply the
weighted average of 44.6 MB and 636 MB matrices, and I mis-read it as variance.

**The real number is that the streaming GEMV runs at ~190 GB/s = 83% of the measured
228 GB/s peak, consistently, across 6,190 launches.** That gives 17.608 GB / 190 GB/s
= **92.7 ms/step** against the measured 104.2, and T1 at 220 GB/s would be
**80.0 ms**. **So T1 is +15.8% of bandwidth efficiency on a kernel that is already
consistent -- not a bug hunt, an efficiency question.**

### Bytes-in-flight is not the limit either (round 126)

Round 125 proposed checking outstanding loads. Done:

| | |
|---|---|
| kernel | `nvfp4_gemv_kernel` -> `nvfp4_gemv_tmpl<1>` |
| ROWS | **1** -- one 8-byte `uint2` weight load in flight per thread |
| registers | **39**, 0 spills |
| occupancy | register count is far below the limit, so 2048 threads/SM = 64 warps |

Bytes in flight: 64 warps x 32 lanes x 8 B = **16 KB per SM**, 786 KB across 48 SMs.
At 228 GB/s and a ~350 ns DRAM latency you need ~80 KB. **So there is ~10x more in
flight than required, and latency is not what is costing the 17%.** The 39-register
count also confirms occupancy was never the issue on this kernel.

**The remaining structural suspect, and it is the round-101 one seen from a new
angle:** with `ROWS`=1 the x-vector is re-read once per row, so per matrix the x
traffic is `N/ROWS = 17,408` passes over the same 5,120 floats -- roughly 348 MB
against the 44.6 MB of weights, **7.8x the weight traffic**. Round 102 measured
staging x in shared as **neutral**, which rules out L2->SM bandwidth as the limit,
but it does not rule out the *instruction slots* those 64 B x-loads consume: per
k-tile each thread issues one 8 B weight load and four 16 B x-loads, so **x-loads
are 4 of every 5 loads in the inner loop.**

### `ROWS`=2 at B=1: tested and rejected (round 127)

The one-line test round 126 proposed, run at the batch size T1 is actually about:

| | ROWS=1 (current) | ROWS=2 |
|---|---|---|
| registers | 39 | 48 |
| spills | 0 | 0 |
| `generate` | 16/16 | **16/16** |
| **t=1 one-row forward** | **104.2 ms** | **107.46 ms (+3.1%)** |

Reverted. **So the load-instruction-mix hypothesis is falsified, and round 104's
negative `ROWS`=2 result now extends to B=1 as well** -- it is worse at both ends of
the batch range, not just at B=16 where it was first measured.

**Seven mechanisms are now ruled out on this single-stream GEMV**, six of them by
measurement: x traffic (shared staging, neutral), bytes in flight (10x more than
needed), DRAM latency, occupancy (39 registers), local memory, the weight stream,
and now the load instruction mix. **The kernel sits at 190 GB/s = 83% of the
measured 228 GB/s peak and nothing that has been counted explains the missing 17%.**

**That is the same shape of result as the prefill GEMM** (rounds 102-111, six
mechanisms ruled out, shape turned out to be the cause). The lesson that transferred
there was to stop counting resources and change the *shape* -- and the single-stream
GEMV's shape is `ROWS`=1, one warp per row, one 8-byte weight load per instruction
per thread. **A shape change, not a resource, is the only kind of move that has
worked on this engine.**

### ...but there is no shape change to make here (round 128)

Round 127 proposed reorganising the warp so that one 32-lane instruction covers 256
contiguous bytes of one row. **Reading the kernel shows that is already what it
does:**

```cuda
const int e0 = i * kTile + lane * kVec;   // kVec = 16
const XVec xv = load_x(xb, e0);
uint2 pk[ROWS];
```

`lane * kVec` puts lane L at elements `[16L, 16L+16)`, i.e. bytes `[8L, 8L+8)` of the
packed 4-bit row -- so **the 32 lanes of one warp already cover 256 contiguous bytes
of a single row in one instruction**. The comment above the load block already
documents separating the loads from the arithmetic so that `ROWS` independent loads
stay in flight, which is exactly what round 126 measured as 10x more than needed.
**The GEMV is already organised the way the analysis says it should be, so round
127's proposal was redundant.**

### The question that actually remains (round 128)

If nothing counted explains the missing 17%, the honest possibility is that **83% of
the probe figure is what this access pattern can do** -- the 228 GB/s in
`bench/hw/bw4.cu` is measured with a *different* pattern, and T1's 12.5 tok/s was
derived from a roofline that assumes 95% of it.

**Settle it by measurement, not argument:** write a probe that streams the real NVFP4
weight layout the way the GEMV does -- one row per warp, 256 B per instruction -- and
measure the ceiling of *that* pattern. `bench/hw/bw4.cu` and `bench/hw/stage_bw.cu`
are the existing templates.

| probe result | conclusion |
|---|---|
| ~190 GB/s | **T1 as written is not a kernel problem**; 190 is the pattern's ceiling and the contract's T1 needs restating from a measured pattern ceiling, the same way the 100 tok/s target was |
| ~228 GB/s | the kernel has 17% to find and the resource counting should resume |

### The probe answers it, and the answer is decisive (round 129)

`bench/hw/gemv_bw.cu` streams the real layout with the GEMV's own pattern -- one row
per warp, lane L at bytes `[8L, 8L+8)`, `ROWS` rows in flight:

| | GB/s |
|---|---|
| **the pattern, measured** | **271.5** |
| `bw4.cu` reference | 228.0 |
| **the real `nvfp4_gemv_kernel`** | **190.0** |

**The pattern does 119% of the reference figure, so the memory system is not the
limit and the missing bandwidth is inside the kernel.** The kernel is at **70% of
what its own access pattern can do** -- a **1.43x** gap, and T1's 220 GB/s is only
**81%** of the pattern ceiling.

Two cross-checks from the same run:

* **`ROWS`=1 (271.5) and `ROWS`=2 (270.6) are indistinguishable**, which is exactly
  consistent with round 127's measurement that `ROWS`=2 does not help the real
  kernel. The pattern is not `ROWS`-sensitive, so `ROWS` was never the lever -- two
  independent measurements now agree on that.
* The `lm_head` shape (635.7 MB, well past L2) gives **232.7 GB/s**, still above the
  reference. Larger shapes cost some, but not the 30% in question.

**What the kernel does that the probe does not, per 8 bytes of weight per thread per
k-tile:** the probe does two XORs. The kernel does a 16-element NVFP4 unpack, a
scale lookup and multiply, and 16 FMAs. **That is the 30%, and it is the first
explanation this session that is supported by a controlled comparison rather than
by arithmetic on a resource.**

#### ...but that explanation does not survive first-order arithmetic (same round)

Checked before handing it forward, because it is the kind of claim that has failed
nine times already:

| | value | capacity | used |
|---|---|---|---|
| bytes/cycle/SM at 190 GB/s | 2.33 | -- | -- |
| ALU, ~33 ops per 8 B | **9.6 ops/cycle/SM** | ~128 FMA/cycle | **7.5%** |
| issue, ~38 instr per 256 B | **28.2 Ginstr/s** | 326 Ginstr/s | **8.6%** |

**The kernel is below 10% of both ALU and issue capacity and still reaches only 70%
of its own pattern's bandwidth.** So the per-element work is *not* the 30% either,
and round 129's explanation is falsified by the same class of arithmetic that
falsified the eight before it.

**What is left is a latency/concurrency problem**, which round 126 "ruled out" by
counting 768 KB in flight against 78 KB needed -- but that count assumed 8 bytes in
flight per thread and ~350 ns latency, i.e. it counted *bytes issued*, not *loads
outstanding per warp*. **With `ROWS`=1 the body has exactly one weight load per
k-tile, and the probe has the same structure and reaches 271.5 GB/s**, so the
difference is not the count of weight loads either. That contradiction is unresolved
and is the honest state of this investigation.

#### The move that actually resolves it: bisect the probe, not the kernel

`ncu` is unavailable (`ERR_NVGPUCTRPERM`, no sudo), so hardware counters cannot
answer it. **But `bench/hw/gemv_bw.cu` can: add the kernel's work to the probe one
piece at a time and watch 271.5 GB/s fall.**

1. probe as-is -> **271.5 GB/s** (measured)
2. probe + NVFP4 unpack of the loaded bytes (no scale, no FMA)
3. + the scale lookup and multiply
4. + the 16 FMAs
5. + the x loads (`load_x`, four `__ldg` float4)

**Whichever step drops the bandwidth is the answer, and it is a five-line change per
step in a file that cannot break the engine.**

### The ablation answers it: it is the SCALE LOAD (round 131)

Run as a `LEVEL` template parameter so every step is the same kernel with work added:

| step | GB/s | delta |
|---|---|---|
| 1 loads only (2 XOR) | **254.3** | -- |
| 2 + NVFP4 unpack | 248.9 | -2% |
| **3 + scale lookup and multiply** | **194.8** | **-22%** |
| 4 + 16 FMAs | 210.5 | +8% (hides latency) |
| 5 + x loads (4x `__ldg` float4) | 212.5 | +1% |
| **the real `nvfp4_gemv_kernel`** | **190.0** | |

**One step accounts for the entire gap, and it lands on the real kernel's number: the
scale lookup.** Both the unpack and the FMAs are ~free; round 130's first-order check
(ALU at 7.5% of capacity) predicted exactly that, which is why the cost had to be a
memory effect -- and it is:

```cuda
const uint8_t sc = __ldg(ws + (size_t)row * scalerow + (i * kWarp + lane));   // scalerow = K/16
```

**The scale is a SECOND MEMORY STREAM, read one byte per thread per k-tile.** One
warp instruction fetches **32 lanes x 1 B = 32 bytes**, against the weight load's
**32 lanes x 8 B = 256 bytes** -- **8x less efficient per instruction**, and a stream
the memory system has to interleave with the weights. The scale row is only
`K/16 = 320` bytes, so this is not volume; it is efficiency. This also explains round
111's observation that the NVFP4 dequant path carries the LOP3/SHF, while ALU
utilisation stays at 7.5%: the instructions are there, the *loads* are what cost.

### The widening was implemented, and it recovers only a third of the prediction (round 132)

Lanes 0..7 now load one `uint32` each (128 B per warp instruction) and broadcast the
byte each lane needs with `__shfl_sync`, replacing one byte per lane (32 B per
instruction):

| | reference | with wide scale load |
|---|---|---|
| `generate` | 16/16 | **16/16** |
| registers / spills | 39 / 0 | 48 / 0 |
| **t=1 one-row forward** | **104.2 ms** | **100.79 ms (-3.3%)** |

**Kept: it is correct and it is a real gain -- the first this session since round
115.** But the ablation predicted ~20% and reality gave 3.3%, and the gap is
informative: **the cost is the second stream's *existence*, not its instruction
width.** Steps 4 and 5 of the ablation *raised* bandwidth (210.5, 212.5) -- adding
more work per k-tile hid the extra load's latency, which is the same signature as a
latency/occupancy cost rather than a bandwidth cost.

**So the remaining ~26% needs the stream removed, not widened** -- hoist the scale row
(320 B for K=5120) into shared memory once per row, or restructure so the scale is not
a per-k-tile load at all. Widening was the cheap experiment and it answered the
question; the next one has to remove the load.

**The original fix note, kept for the record:** each warp's k-tile needs 32 *consecutive* scale
bytes. Load them as 8 lanes x `uint32` (128 B per instruction) or 2 lanes x `uint4`
and broadcast with `__shfl_sync`, or hoist the whole 320-byte scale row into registers
or shared once per row. **Any of those turns a 32 B/instruction stream into a 128 B
one, and the ablation says that step is worth 54 GB/s** -- 190 -> ~235, which is
above T1's 220. That is the same ablation logic that
located the tile shape on the prefill GEMM, applied to the one measurement that has
been decisive so far this session.

**Round 127's `ROWS`=2 failed because it kept every one of those per-element
operations and only changed how many rows shared a loop.** The lever is to reduce
the *per-element* work, or to process more rows per thread so that the x loads and
loop overhead are amortised without changing the arithmetic -- not to re-count
memory resources, which nine rounds have now eliminated.

**That is one cheap probe that ends the argument either way**, and it is the same
move that resolved the round-59 DRAM question -- measure the pattern, not the kernel.

**That is the thing to look at next**: not bandwidth, not latency, not occupancy --
the *load instruction mix*. `ROWS`=2 would halve the x-load count per unit of
weight, and round 104 already measured its cost at B=16 (12% worse) but never
measured it at **B=1**, which is the case T1 is about. **At B=1 there is no batch
to amortise x across, so `ROWS` matters more, not less.** Rounds 102-111 spent six rounds testing
mechanisms on the *prefill* GEMM; the single-stream GEMV has never been decomposed
before this round, and the first decomposition says the win is not in the typical
launch but in the outliers. **Find out what the slow launches have in common before
changing any kernel** -- the obvious candidates are L2-cold first touches, the
`lm_head`'s different shape, and the prefill/decode boundary, and each is checkable
by correlating launch index with duration in the nsys trace.

**Where to start:** the B sweep in round 104 gives `nvfp4_gemv_kernel` ~271 us avg
at B=1 across 23,940 launches, but nothing has decomposed that 104.2 ms/step by
kernel. **Profile one decode step at B=1 first** -- exactly the step that was
skipped on the prefill side, where six rounds were spent testing mechanisms that a
single profile would have ordered.

**Do not re-derive the 228 GB/s peak**; it is measured in `bench/hw/bw4.cu` and
reproducible with `bench/results/` artifacts.

## The tile shape WAS the cause: TM=8/TNREG=4 (round 115)

After six resource hypotheses failed (rounds 102-111), round 111 concluded the 36%
FMA efficiency came from the tile *shape*. **It did.** The change in
`loop/patches/tm8-tnreg4.md` was applied verbatim:

| | TM=4/TNREG=8 | **TM=8/TNREG=4** |
|---|---|---|
| shared per 32 FMA | 40 B (`wv` 8 + `xv` 16 + `xw` 16) | **32 B (`wv` 16 + `xv` 16)** |
| loads per k | 3 | **2** |
| ptxas registers | 96 | 97 |
| ptxas smem | 24,576 B | **24,576 B** (predicted unchanged) |
| **`generate`** | 16/16 | **16/16 (100%)** |
| **TTFT** | 462.4 ms | **434.0 ms (-6.1%)** |
| **endpoint 16 concurrent** | 18.82 tok/s | **19.83 tok/s (+5.4%)** |

`chunked-prefill`, `batch-parity` and `mtp-probe` all still OK, and 16 concurrent
requests still log `batch of 16` and return **1 distinct output** for 256 tokens.

**This is the first change since round 98 that is both correct and faster.** Two
things are worth carrying forward from it:

* **The patch spec worked because it named the trap.** Round 96 failed on exactly
  this change by pairing `TNREG`=4 with a `tx = threadIdx.x & 7` mapping that
  covered 32 of 64 columns. Writing the mapping table down as a step meant the
  re-application hit 16/16 on the first try.
* **Six failed attempts to find a resource were the useful result.** Each one was
  cheap and each one eliminated a real candidate; the shape was found by
  elimination, not by insight.

**The prefill GEMM now runs at** 44.6 MB / (434 ms x 1,379.5/413 us scaling) --
re-measure; the 32.3 GB/s figure is from the previous shape. The next increment is
the same trick again: `TM`=16 would need `acc[16][2]` and 16 columns per `tx`
group, which may or may not tile 64x64 with 128 threads -- **check the mapping
before writing the body, exactly as above.**

## The endpoint's batching is verified correct, and its gap is now arithmetic (round 112)

First time this was actually checked rather than assumed: `GB10_BATCH_LOG=1` on 16
concurrent requests logs **`batch of 16`** -- one group, all sixteen collected by
the 25 ms window. 16 responses, 256 tokens, **1 distinct output** (the identical
prompt gives an identical answer 16 times), wall **13.60 s = 18.82 tok/s**.

That closes the diagnosis end to end. The endpoint's own decomposition:

| | |
|---|---|
| prefill (16 serial `prefill_seq`) | 7.63 s |
| decode (16 steps at the engine's measured 334.9 ms) | 5.36 s |
| overhead | ~0.6 s |
| **total** | **13.60 s, matching the measured wall time** |

**So the batching infrastructure is right and the prefill is the entire gap.** If
prefill hit its own 4.85 s bound the total would be 10.2 s = **25 tok/s**; the 30
tok/s target needs prefill near 4.0 s, i.e. the prefill GEMM at ~70 GB/s against
the 32.3 GB/s it actually reaches.

**The best measured endpoint number is 18.82 tok/s** (13.60 s), against 17.37
recorded earlier -- the same test, so prefer the newer figure and treat ~18 tok/s
as the current value.

## Six ruled-out mechanisms point at the tile shape, not a resource (round 111)

Rounds 102-111 tested, one mechanism at a time, and **all six failed to move the
prefill GEMM**: x re-reads, local memory, the weight stream, occupancy (raising it
17% *worse*), shared bandwidth (bf16 `xt`: correct-but-gate-failing, and -3.3%),
and inner-loop ALU issue pressure (hardware CVT: neutral). `nvdisasm` shows the
kernel is 1024 FFMA, 16 FMA per LDS.128, with nothing obviously starved.

**Six failures to find a resource bottleneck is itself the information: the 36% FMA
efficiency at T=58 comes from the tile shape, so the next attempt should change the
shape rather than relieve a resource.** The specific proposal is to split the
`TNREG`=8 outer product into two `TNREG`=4 halves so each `xt` load feeds twice the
FMAs -- while *preserving* the TT=64 thread mapping, since round 96 failed when
`TNREG`=4 was combined with a mapping that assumed 8.

## The remaining target, precisely (round 105)

All gates green at `15a826e`: `generate` 16/16 (100%), `chunked-prefill` OK,
`batch-parity` OK, `mtp-probe` OK, 0 build errors.

**The engine already beats the concurrency target**: `batch-parity --n-seq 16`
gives **47.78 tok/s aggregate**, against the 30 the objective asks for.

**What falls short is the delivered endpoint**: 16 concurrent requests take
14.74 s for 256 tokens = **17.37 tok/s**. Decomposed with `max_tokens` 1 vs 16:

| part | time | its own bound | gap |
|---|---|---|---|
| prefill (16 serial `prefill_seq`) | **7.63 s** | 4.85 s (16 passes at 58 GB/s) | **1.57x** |
| decode (16 steps) | 6.5 s | 1.9 s (16 passes at 148 GB/s) | 3.4x, but it matches the engine's own 335 ms/step |

So the endpoint's shortfall is **prefill**, and prefill is the *same* GEMM that
makes TTFT ~500 ms. **One number, two objectives.** Closing it takes the endpoint
from 17.37 to ~30+, and TTFT from ~500 ms toward its own bound.

**What is known about that prefill pass**: 58 tokens fits one 64-wide tile, so it
is one weight pass = 303 ms at the 58 GB/s the GEMM achieves at t=1 -- but it
measures ~477-500 ms, i.e. **35 GB/s**. It is not the weight stream and not raw
FMA throughput (the whole model's prefill is only ~151 ms of MACs). Candidates
left: the outer product's per-chunk barrier count at `KC`=32, and the 64-wide tile
wasting 6 of 64 columns at T=58.

**Ruled out by measurement, do not retry these:**

* x re-reads in the batch GEMV (1.43 GB, 32x the weights) -- staging x in shared
  was neutral, because 327 KB already lives in a 25 MB L2 (round 102);
* local memory in the batch GEMV -- `ROWS` 4->2 removes half the stack frame and
  is 12% *worse* at B=16 (round 104);
* the weight stream, by the shape of the B sweep (round 104);
* the GEMM's load pattern, dequant and shared stores (rounds 76-81);
* one-row-per-warp loads in the GEMM -- impossible at TN=64, needs 65,536 B of
  tile against a 49,152 B budget (round 85).

### The prefill GEMM is occupancy-limited, and it is quantified (round 106)

`ptxas -v` on the current `gemm.cu`:

| kernel | registers | spills | smem |
|---|---|---|---|
| `nvfp4_gemm_kernel` | 96 | 0 | 24,576 B |
| `fp8_gemm_kernel` | 85 | 0 | 24,576 B |
| `bf16_gemm_kernel` | 104 | 0 | 24,576 B |

Resource limits per SM (GB10: 65,536 registers, 102,400 B shared, 24 blocks):

```
registers: 65,536 / (96 * 128) = 5 blocks
shared   : 102,400 / 24,576   = 4 blocks   <- binding
=> 4 blocks = 512 threads = 16 warps of 48 = 33% occupancy
```

**33% occupancy, and the prefill measures 33% of FMA peak** (5.7 G FMAs per
[17408,5120] matrix against a 10.4 T FMA/s ceiling). Those two numbers agreeing is
what makes this a diagnosis rather than a guess -- and it is the first mechanism
this session that is supported by two independent measurements.

### The occupancy hypothesis, tested and rejected (round 107)

The cleanest test needs no precision change: `KC`=32 -> 16 halves the shared tile.

| `KC` | registers | smem | blocks/SM | TTFT |
|---|---|---|---|---|
| 32 | 96 | 24,576 B | 4 | **498-500 ms** |
| 16 | 78 | 12,288 B | **6** | **582.3 ms (+17%)** |

**Correct (`generate` 16/16) and clearly worse.** So occupancy is not the binding
constraint either: raising it from 33% to 37.5% costs 17%, because doubling the
chunk count doubles the barriers and that dominates.

**The 33%/33% agreement was a coincidence, not a diagnosis.** Two numbers matching
is not evidence unless the mechanism connecting them is itself tested -- which is
the same lesson as rounds 59, 101 and 104, arriving from the opposite direction:
there, a counted mechanism was wrong; here, a *correlated* pair of measurements
was. **Four mechanisms are now ruled out by measurement on this GEMM** (load
pattern, dequant, shared stores, occupancy), and the prefill still runs at ~35 GB/s
against the ~58 GB/s the same kernel reaches at t=1 -- **and even that comparison
may be invalid, because the 58 GB/s figure was taken under the old TT=32/`KC`=64
configuration.** Re-measuring the t=1 launch under the current configuration is
the first thing the next session should do, before any further theory.

**The lever is shared memory per block.** The 24,576 B is `xt[2][32][64]` as f32
(16,384) plus `wt[2][32][64]` as u16 (8,192). Making `xt` bf16 takes it to
16,384 B total, which allows 6 blocks by shared and 5 by registers -- i.e. **up to
5 blocks/SM, 640 threads, 20 warps, 42% occupancy**, and it is the same change
that would free the room for `KC`=64.

Note this is the *third* time bf16 `xt` has come up, but the first time for a
measured reason: rounds 97-98 rejected it on the strength of a confounded
`KC`/padding comparison, which round 98 itself overturned. **The gate decides** --
if bf16 activations do not survive the oracle comparison, the change is out
regardless of the occupancy arithmetic.

### The prefill is compute-bound in the outer product, and here is the number (round 108)

`nsys` on `generate --n 2` (TTFT 461.2 ms, 59 tokens) under the **current**
TT=64/`KC`=32 configuration:

| kernel | launches | avg | what it is |
|---|---|---|---|
| `nvfp4_gemm_kernel` | 192 | **1,379.5 us** | 64 layers x 3 (gate/up/down) |
| `fp8_gemm_kernel` | 208 | 710.6 us | 64 layers x 3.25 |

192 x 1,379.5 us + 208 x 710.6 us = **413 ms of the 461 ms TTFT. 89% of TTFT is
the GEMM**, and the launch counts are exactly the model's matrix counts, so this
is the whole prefill and nothing else.

**Two things this settles:**

1. **The old 825.6 us figure was from TT=32/`KC`=64 and is not comparable.** The
   round-107 flag was correct: the "35 GB/s against 58 GB/s" comparison was
   invalid. Under one configuration, the same matrix is **1,379.5 us**.
2. **The prefill is compute-bound, not bandwidth-bound.** This matrix is 44.6 MB,
   so the launch moves it at **32.3 GB/s**, and it executes 5.17 G MACs in
   1,379.5 us = 3.75 T MAC/s = **36% of the 10.4 T MAC/s peak**. At t=1 the same
   matrix ran at 54 GB/s and was bandwidth-bound. TT=64 with `TNREG`=8 does 2x the
   outer-product FMAs per weight byte that TT=32 with `TNREG`=4 did, and costs
   1.67x the time -- **the ratio, not the absolute, is the evidence.**

### The outer product's ceiling is its shared traffic -- hypothesis, untested (round 109)

Per k, per thread, `gemm2d_outer` at TM=4/TNREG=8 issues:

| | bytes |
|---|---|
| `wt` 4 x bf16 | 8 |
| `xt` first 8 floats | 32 |
| `xt` second 8 floats | 32 |
| **total shared** | **72 B** |
| FMAs | 32 |

Per block per k that is 9,216 B of shared against 4,096 FMAs, and GB10's shared
bandwidth is 128 B/cycle: **72 cycles of shared against 32 cycles of FMA -- a 44%
ceiling**, against the **36% measured** at T=58.

**But this is exactly the round-106 pattern and must be treated as such.** There, an
occupancy figure and an FMA figure agreed at 33% and the agreement was a
coincidence; the mechanism connecting them had never been tested. Here the
agreement is again between a *calculated ceiling* and a *measured efficiency*, and
**calculating a ceiling is not measuring a bottleneck.** The round-106 test that
falsified its own hypothesis -- raise occupancy, see if it helps -- is the standard
to meet, not the agreement itself.

**The falsifiable prediction:** the `xt` loads are 64 of the 72 bytes, so if the
shared path is the limit, making `xt` bf16 takes it to 40 B per k (8 + 16 + 16),
raising the ceiling to 80% and trimming the critical path by 1.8x. **If TTFT does
not fall substantially, the hypothesis is dead** -- and it is the same change that
rounds 97-98 rejected on a confounded comparison and round 107 could not reach
through occupancy, so this round trip finally gives it a mechanism *and* a test.

### bf16 `xt`: built, fails the gate, and the hypothesis is falsified (round 110)

The change was implemented exactly as the round-109 prediction required -- 91
registers, **shared down from 24,576 B to 16,384 B**, 0 spills, xt shared traffic
per k down from 64 B to 32 B:

| | baseline | bf16 `xt` |
|---|---|---|
| `generate` oracle agreement | 16/16 (100%) | **0/16 (0.0%)** |
| TTFT | 461.2 ms | 446.2 ms |

**The gate fails outright, so the change is out** -- and this is the *third* verdict
on it, after round 97-98 rejected it on a confounded `KC`/padding comparison and
round 107 could not reach it through occupancy. **This verdict is the one that
counts, because it is on correctness rather than on a timing comparison**, and it
is now recorded so that no future round re-litigates it: **bf16 activations do not
survive comparison with the bf16 oracle.**

**The shared-bandwidth hypothesis is also falsified.** It predicted that halving
the xt shared traffic would trim the critical path by 1.8x. It moved TTFT by
-3.3%, inside noise. So the outer product is **not** shared-bandwidth-bound, and
the round-109 44%-vs-36% agreement was the round-106 coincidence again -- **the
fourth time this session that arithmetic compressing onto a measured number turned
out to mean nothing.**

**Do not quote the 446.2 ms.** A gate-failing kernel produces a number that looks
like an improvement for the same reason round 88's broken TT=64 kernel reported
469.2 ms: the model is not decoding what the measurement assumes. **Five mechanisms
are now ruled out on this GEMM by measurement** -- load pattern, dequant, shared
stores, occupancy, shared bandwidth -- and the 36% FMA efficiency at T=58 remains
unexplained.

**So the lever is outer-product FMA efficiency (36%), and it is the opposite end of
the kernel from rounds 60-84, which spent 25 rounds on the load side.** The earlier
conclusion that the load pattern was the problem was measured against a
bandwidth-bound kernel; the prefill is not one.

**A benchmark trap worth recording:** `forward-cost` **no longer exercises the GEMM
at all.** Its t range is 1-16 and the GEMV cut is `t <= 16`, so every one of its
points goes through `nvfp4_gemv_batch_kernel`. Any GEMM work must be measured
through `generate` or `chunked-prefill`. Profiling `forward-cost` and concluding
anything about the GEMM is now meaningless.

**Three separate times this session a plausible mechanism was counted rather than
measured and turned out to be wrong** (rounds 59, 101, 104). The rule that works
is: change one variable, read an absolute per-launch number, and run the gate
before believing the result.

## The batch decode's 3.3x gap is x re-reads, and the fix is quantified (round 101)

`nvfp4_gemv_batch_tmpl` is 392 ms/step at B=16 against a 119 ms weight bound. The
cause is in its loop structure, not its weight handling:

```cuda
for (int i = 0; i < full_tiles; ++i) {          // k-tiles
    load pk[ROWS], sc[ROWS];                    // weights, ROWS x 8 bytes
    for (int b = 0; b < B; ++b) {
        const XVec xv = load_x(x + b*K, e0);    // x, re-read per row group
        for (int r = 0; r < ROWS; ++r) { ... }
    }
}
```

Weights are loaded once per k-tile and reused across all B -- that part is right.
But **x is re-read once per row group**, and there are `N/ROWS = 4352` of them:

| | per matrix (K=5120, N=17408, B=16) |
|---|---|
| weights | 44.6 MB, read once |
| x, per k-tile | 327.7 KB |
| row groups | 4,352 |
| **x traffic** | **1.43 GB** |
| **x / weight ratio** | **32x** |

**So the kernel is x-bound, not weight-bound, and that is the whole 3.3x.**

`ROWS` cannot simply be raised: `acc[ROWS][BMAX]` is 4 x 16 = 64 registers at
ROWS=4, and ROWS=8 would be 128. That is why the kernel is shaped this way.

**The fix is to stage the k-tile of x in shared memory.** At B=16 one k-tile of x is
`B * kTile * 4 = 16 * 512 * 4 = 32,768 B`, which fits, and then x is read once per
block per k-tile instead of once per row group -- removing the 1.43 GB entirely.
The cost is one `__syncthreads()` per k-tile, i.e. `K/kTile = 10` barriers.

### Implemented, and the diagnosis was wrong (round 102)

The shared staging was built exactly as described (`__shared__ float
xs[BMAX][kTile]`, cooperative stage per k-tile, two barriers) and is **correct**:
`generate` 16/16 (100%), `batch-parity` OK.

It is also **neutral**, and the step cost got slightly *worse*:

| max_tokens | before | with shared x |
|---|---|---|
| 1 | 8.58 s | 7.96 s |
| 16 | 14.74 s | **14.61 s** |
| implied step cost | 419 ms | **443 ms** |
| aggregate | 17.37 tok/s | 17.52 tok/s |

Reverted -- the rule is to revert anything inside the +-2% noise band, and this is
+0.9% on the primary metric while adding two barriers per k-tile.

**So the x-traffic analysis above is wrong.** The arithmetic is right (1.43 GB of
x reads against 44.6 MB of weights) but the conclusion drawn from it is not: x is
327 KB and the L2 is 25 MB, so those re-reads were already being served from L2,
and L2 bandwidth was never the limit. Counting bytes is not the same as finding
the bottleneck -- **the same mistake the round-59 DRAM probe made**, in a new
place. The batch decode's 3.3x gap is still unexplained, and the next attempt
should start from a profile of the kernel rather than from a traffic count.

What *is* now established is the shape of the problem: 392 ms/step for 16
sequences against a 119 ms weight bound, with the weights demonstrably read once.
The remaining candidates are the FMA issue rate (16 FMAs per weight element per
batch element is a lot of arithmetic per byte) and the accumulator register
pressure that forces `ROWS = 4`.

### Profiling the batch path, and a correction about batch sizes (round 103)

`nsys` on `batch-parity`:

| kernel | % | launches | avg |
|---|---|---|---|
| `nvfp4_gemv_kernel` | 36.2 | 23,940 | 271.3 us |
| `fp8_gemv_kernel` | 25.0 | 25,792 | 173.4 us |
| `nvfp4_gemv_batch_kernel` | 16.7 | 7,135 | **418.4 us** |
| `fp8_gemv_batch_kernel` | 10.8 | 7,696 | **251.4 us** |
| `nvfp4_gemm_kernel` | 2.8 | 384 | 1,319.5 us |

**Correction: `batch-parity` runs 4 sequences, not 16.** It reports *"26.59 tok/s
aggregate (4 seq x 6.6 tok/s each), 150.4 ms/step"*. The **40.75 tok/s** figure
quoted earlier in this document for concurrency 16 came from a 16-sequence
invocation, and the two must not be mixed -- they are different batch sizes with
different per-step costs.

Step cost against the 77 ms weight floor, which holds at every batch size:

| batch | ms/step | vs floor |
|---|---|---|
| 4 | 150.4 | 2.0x |
| 16 | 392 | 5.1x |

**The gap grows with batch**, which rules out the weight stream (read once either
way) and points at something that scales with the number of sequences. The compute
bound at B=16 is only ~41 ms, so it is not raw FMA throughput either.

### The B sweep, and what it says about the endpoint (round 104)

`batch-parity --n-seq N` gives a clean curve. The weight floor is 77 ms/step at
every batch size:

| B | ms/step | vs floor | aggregate tok/s |
|---|---|---|---|
| 1 | 104.2 | 1.35x | 9.59 |
| 2 | 135.0 | 1.75x | 14.82 |
| 4 | 148.3 | 1.93x | 26.97 |
| 8 | 190.4 | 2.47x | 42.01 |
| **16** | **334.9** | **4.35x** | **47.78** |

**The engine already reaches 47.78 tok/s aggregate at B=16, well above the 30 the
objective asks for.** So the endpoint's 17.37 tok/s is the actual gap, and the
sweep says where it is: the endpoint's 14.74 s is ~8.2 s of **prefill** and ~6.5 s
of decode, and 6.5 s / 16 steps = 406 ms/step against the engine's own 335 -- so
**the decode is roughly at parity and the prefill is the whole difference.**

Bound check: prefill is 16 weight passes at 58 GB/s = 4.85 s, decode is 16 passes
at 148 GB/s = 1.9 s, total 6.75 s = **38 tok/s achievable** against 17.37 measured.
Batching prefill would save only ~6% (a 928-token concatenation is 15 tile-passes
against 16), so **the prefill win has to come from the prefill GEMM itself**, which
runs 58 tokens in ~500 ms against its own 303 ms pass.

### The local-memory hypothesis, tested and rejected (round 104)

`nvfp4_gemv_batch_kernel` compiles to 128 registers **with a 256-byte stack
frame**, i.e. `lo[ROWS][8]` and `hi[ROWS][8]` are partly in local memory and get
read back 16 times each in the inner loop -- a plausible cause of the batch gap.

Tested by dropping `ROWS` from 4 to 2, which frees registers exactly as expected
(80 registers, 128-byte stack):

| | B=4 | B=16 |
|---|---|---|
| ROWS=4 | 148.3 ms | **334.9 ms** |
| ROWS=2 | 147.0 ms | **374.1 ms (+12%)** |

**Worse at B=16**, because halving `ROWS` doubles the x re-reads per row group.
Reverted. So local memory is not the dominant cost either -- that is now three
mechanisms ruled out by measurement on this kernel (x traffic, local memory, and
by the sweep's shape the weight stream itself).

## Final state of this session

Everything below is measured, and every claim is backed by a gate or a number in
this file. The engine is complete and usable; what remains is performance.

**Delivered and verified**

| item | evidence |
|---|---|
| Pure-Rust inference engine | builds with 0 errors, aarch64 |
| Qwen3.8-27B **NVFP4** on GB10 | `generate` 16/16 exact vs the bf16 oracle |
| MTP | `mtp-probe` OK, acceptance 44/48, token-exact |
| OpenAI protocol endpoint | `POST /v1/chat/completions` |
| Anthropic protocol endpoint | `POST /v1/messages` |
| Concurrency 16 (engine) | `batch-parity` 16/16 exact |
| Concurrency 16 (endpoint) | 16 concurrent requests, 17.37 tok/s |
| otp vs llama.cpp | 8.48-8.62 vs llama.cpp, better |
| Loop project, push per round | rounds 36-99, all PASS and pushed |

**Performance achieved this session**

| | before | after |
|---|---|---|
| t=1 forward | 271.46 ms | **118.40 ms (-56%)** |
| TTFT | 566.3 ms | **498.2-508.3 ms (-12%)** |
| endpoint, 16 concurrent | 5.36 tok/s (serial server) | **17.37 tok/s (3.24x)** |

The three changes that did it: **round 85** (route short prompts to the batched
GEMV -- one `if`, worth -56% on its own), **rounds 94-95** (a batching scheduler
plus a 25 ms batching window in the server), **rounds 96-98** (a 64-wide prompt
tile, then removing the tile padding).

**Not achieved, with reasons**

* **Single-decoder 100 tok/s -- arithmetically impossible on this hardware.**
  17.6 GB of weights per token against 228 GB/s measured gives a roofline of
  **~13 tok/s**; 100 would need ~1.76 TB/s. No implementation can reach it. The
  engine runs at 8.48 tok/s, ~65% of the roofline.
* **TTFT better than llama.cpp**: 498-508 ms against ~74 ms, ~6.9x off.
* **30 tok/s at 16 concurrent**: 17.37 measured, 1.7x short.

Both open items reduce to the same thing -- **making the GEMM and GEMV paths reach
their bandwidth bounds**. The bounds and the shortfalls per path are in the
round-99 notes; the batch decode is 3.3x off its bound, and the prefill GEMM is
1.8x off, and those two numbers are the whole of the remaining gap.

## What worked, methodologically

* **Probe before optimising.** Every real gain came from a measurement, and the
  single largest came from reading the code rather than measuring.
* **Change one variable.** Round 97 changed `KC` and the padding together and
  reached a conclusion that round 98's clean experiment reversed -- it would have
  sent the next session into a risky bf16 change that was pointless.
* **Never disable a component and subtract** (round 81). Change its size and read
  the profiler's absolute per-launch number instead.
* **Re-measure quoted numbers.** The TTFT figure repeated for dozens of rounds was
  from round 27 and 24% stale (round 86).
* **Run the gate before believing the number.** A broken TT=64 kernel reported a
  -17% TTFT that was pure artefact (round 88).
* **Read the codebase.** The answer to the GEMM problem was written in `gemv.cu`
  the whole time (round 83), and the biggest win of the session was an `if` in
  `weights.rs` (round 85).

## Where the endpoint's 14.74 s goes, and both remaining gaps (round 99)

All gates green at `959557b`: `generate` 16/16 (100%), TTFT 508.3 ms, decode
8.48 tok/s single-stream, `chunked-prefill` OK, `batch-parity` OK, `mtp-probe` OK.

16 concurrent requests, 256 completion tokens, 14.74 s = 17.37 tok/s:

| part | time | vs its own bound |
|---|---|---|
| prefill (16 serial `prefill_seq`) | ~8.2 s | 16 weight passes |
| decode (16 x ~392 ms) | ~6.2 s | 16 x 119 ms bandwidth bound |

**Both halves are ~2-3x off their bound, and the bounds are the same kind of
number, so this is one problem, not two:**

* **prefill**: 16 serial prefills cost 16 weight passes, one per request. That is
  inherent to `prefill_seq` being per-sequence -- but it is also 1.8x more than
  16 passes should cost (16 x 17.6 GB / 58 GB/s = 4.85 s), so there is ~1.8x in
  the prefill GEMM itself. Concatenating the group into one masked sequence would
  save only ~6% (15 tile-passes instead of 16), so **the win is in the GEMM, not
  in the batching** -- worth knowing before anyone builds a block-diagonal mask.
* **decode**: 392 ms/step for 16 sequences against a 119 ms bandwidth bound and a
  ~41 ms compute bound. **3.3x off**, and `batch-parity`'s 40.75 tok/s aggregate is
  the same number, so the engine's batch decode has been 3.3x off its bound all
  along. This is the GEMV batch kernel, which is where the 148 GB/s figure came
  from -- at B=16 it does not hold.

**The two open objectives (TTFT, and 30 tok/s at 16 concurrent) both reduce to
making the GEMM/GEMV paths reach their bandwidth bound**, which is the same
problem that consumed rounds 60-84 on the single-stream side. The difference now
is that the bounds and the shortfalls are measured per path rather than inferred
from a probe.

## Round 85 was the single biggest win, and it was an `if`

`forward_prefill` routed every matrix with `n >= 256` to the tiled GEMM; the
batched GEMV reads each weight once and reuses it across all `t`, and at 148 GB/s
against the GEMM's 58 GB/s it wins for any short prompt. One condition took the
t=1 forward from 271.46 ms to 118.40 ms. Rounds 60-84 had been trying to make the
GEMM's loads match the GEMV's pattern; the constraint that made that impossible
(one warp instruction covers 256 B = 512 NVFP4 elements, so one row per
instruction needs a 512-deep tile = 65,536 B at TN=64) is computed in the round-85
notes. **The fix was never to speed up the GEMM; it was to stop using it.**

## TT=64 is landed, with a measured tradeoff (round 96)

The 64x64 tile from round 90 is reinstated together with a wider GEMV cut, and it
is correct: `generate` 16/16 (100%), `batch-parity` OK. `GB10_TT` 64, `GB10_KC`
32, `GB10_TNREG` 8, `GB10_TILE_T` 64, and `forward_prefill` routes to the batched
GEMV for `t <= 16`.

| | before | after |
|---|---|---|
| TTFT (58-token prompt) | 566.3 ms | **518.2 ms (-8.5%)** |
| endpoint, 16 concurrent, 256 tokens | 15.73 s | **15.06 s (17.0 tok/s)** |
| t=1 | 116.56 | 117.33 |
| t=2 | 145.98 | 142.35 |
| t=4 | 156.80 | 153.86 |
| t=8 | 193.36 | 189.21 |
| **t=16** | **292.10** | **343.21 (+17.5%)** |

**Kept**, because the wins are on stated objectives (TTFT, endpoint concurrency)
and the loss is on an intermediate grid point that no objective names. Correctness
re-verified end to end: 16 identical concurrent prompts still give exactly one
distinct output.

**But the gain is far below the ~2x the traffic argument predicts**, and the reason
matters for whoever picks this up: TT=64 halves the weight passes (58 tokens fits
one 64-wide tile instead of two 32-wide ones) but `KC`=32 doubles the chunk count
and therefore the barriers, which cancels most of it. **`KC`=64 with TT=64 needs
52,224 B of shared memory against a 49,152 B budget** -- over by 3,072 -- so
getting the full 2x means shrinking `xt` (bf16, or single-buffered) or the padding.
That is the concrete next step, and it is worth ~2x on prefill, which is the
majority of both TTFT and the endpoint's 16-concurrent time.

Remaining gap to the objective's 30 tok/s at 16 concurrent: 17.0 measured.

### KC=64 at TT=64 is reachable but slower (round 97)

The padding was removed from both tiles so that TT=64 with `KC`=64 fits exactly at
the 49,152 B budget:

| | shared | TTFT |
|---|---|---|
| TT=64, `KC`=32, padded (68) | 26,112 B | **518.2 ms** |
| TT=64, `KC`=64, unpadded (64) | **49,152 B** | 537.7 ms |

Correct (`generate` 16/16) but **3.8% slower**. So the padding is worth more than
halving the chunk count and its barriers, and **`KC`=64 at TT=64 is not reachable
with padding at all** -- the next step up is over budget in every combination:

```
xt[2][64][68] + wt[2][64][64] = 51,200 B
xt[2][64][64] + wt[2][64][68] = 50,176 B
xt[2][64][66] + wt[2][64][64] = 50,176 B
```

Reverted; **TT=64 with `KC`=32 and padding stays the best configuration found** at
TTFT 518.2 ms and 17.0 tok/s at 16 concurrent.

### The padding was the confound, and removing it wins (round 98)

Round 97 changed two things at once, so it could not say whether `KC`=64 was worse
or the missing padding was. Isolated:

| configuration | shared | TTFT |
|---|---|---|
| `KC`=32, padded (68) | 26,112 B | 518.2 ms |
| `KC`=32, **no padding** (64) | **24,576 B** | **498.2 ms** |
| `KC`=64, no padding (64) | 49,152 B | 537.7 ms |

**Removing the padding is a win, not a loss** (-3.8%), and `KC`=32 is genuinely
better than `KC`=64 -- so the round-97 conclusion that the padding "was worth more
than halving the barriers" was wrong, and the bf16 `xt` lever it recommended is
**not needed**. That is the value of isolating one variable: two rounds of
reasoning from a confounded pair of numbers pointed at a risky change that the
clean experiment shows is pointless.

Landed: `GB10_WSTRIDE = GB10_TN`, `GB10_XSTRIDE = GB10_TT`, `KC`=32, TT=64.

| | baseline | now |
|---|---|---|
| TTFT | 566.3 ms | **498.2 ms (-12%)** |
| endpoint, 16 concurrent, 256 tokens | 47.73 s (serial) | **14.74 s = 17.37 tok/s (3.24x)** |
| shared memory used | 26,112 B | 24,576 B |

Correctness: `generate` 16/16 (100%), `batch-parity` OK, and 16 identical
concurrent prompts still give exactly one distinct output.

**So the prefill gain is capped here unless `xt` stops being f32.** Making `xt`
bf16 would free 17,408 B, which is exactly what `KC`=64 with padding needs
(34,816 + 17,408 = 52,224 - 17,408 = 34,816). That is the one remaining lever, and
it trades activation precision for roughly 2x on prefill -- the majority of both
TTFT and the endpoint's concurrency time. It should be tried with the generate
gate as the arbiter, since bf16 activations may or may not survive the oracle
comparison.

## THE DELIVERED ENDPOINT DOES NOT SERVE CONCURRENT REQUESTS (round 91)

This is an objective-level gap and it outranks TTFT. The objective asks for
concurrent inference up to 16 at >= 30 tok/s aggregate *and* a usable local
endpoint. The **engine** does that -- `batch-parity` is 16/16 exact at 40.75 tok/s
-- but the **server does not use it**:

```rust
// crates/gb10-server/src/main.rs
let state = ModelState::new(&dev, &model, MAX_SEQ, 1)?;   // n_seq = 1
...
for conn in listener.incoming() {                          // one connection at a time
```

`n_seq = 1` and a serial accept loop mean the endpoint handles **one request at a
time**. Sixteen concurrent clients would queue, and the aggregate throughput would
be 16x *worse* than the single-stream number, not 40.75 tok/s. So
`batch-parity`'s result is not reachable through the endpoint the user was
promised.

**Fix, and it is bounded:**

1. a worker thread that owns `Model`/`ModelState`/`Scratch`;
2. an `mpsc` queue of pending requests, each carrying its own token stream
   channel back to its connection thread;
3. a batching loop that drains the queue (up to 16) and calls the existing
   `step_batch`, emitting each sequence's token to its own channel;
4. the accept loop spawns a thread per connection, which enqueues and then
   forwards tokens as SSE.

Every piece except the scheduler already exists and is verified
(`step_batch`, `batch-parity`, the SSE writers). **Success test: 16 concurrent
`curl` requests to `/v1/chat/completions`, aggregate >= 30 tok/s, and each stream
individually correct.**

That test -- not `batch-parity` -- is what the objective actually asks for, and
until it passes the concurrency requirement is unverified end to end.

### The endpoint gap, measured end to end (round 92)

The state is now sized for it (`MAX_CONCURRENT = 16`, `ModelState::new(..., 16)`),
which is safe on its own because `step_batch` takes its batch size from
`tokens.len()` and requires only `<= state.n_seq`, so a single request still runs
with `n_seq = 1`. Verified: `GET /health` -> ok, and a single
`POST /v1/chat/completions` returns 12 tokens normally.

Then the objective's actual test, 16 concurrent clients:

```
16 parallel curl -> /v1/chat/completions, max_tokens=16
wall 47.73 s, 256 completion tokens -> 5.36 tok/s aggregate
```

**Against the 30 tok/s the objective asks for, and against the 40.75 tok/s the
engine's own batch path already reaches.** The endpoint is 7.6x short, which is
exactly what `n_seq = 1` plus a serial accept loop predicts: the 16 requests run
one after another, each at roughly the single-stream rate.

This is the first honest end-to-end number for the concurrency requirement, and it
is also the test harness for fixing it: **the scheduler is done when this same
command reports >= 30 tok/s aggregate with each stream individually correct.**

### The exact shape of the scheduler (round 93)

Verified this round: tree clean at `91bbd0d`, 0 build errors, `generate` 16/16
(100%), `batch-parity` OK, `chunked-prefill` OK. The server change from round 92
is committed and pushed.

Both halves are required -- **batching alone changes nothing while the accept loop
is serial, because there is never more than one pending request**:

1. **Concurrent accept.** `for conn in listener.incoming()` becomes a
   `std::thread::spawn` per connection. The `Engine` must move out of `main` into
   the scheduler thread, so `handle`/`handle_chat_completions`/`handle_messages`
   stop taking `&mut Engine` and take a `Sender<Job>` instead.
2. **The scheduler thread** owns `Engine` and runs:

```rust
struct Job { messages: Vec<ChatMessage>, max_tokens: usize,
             enable_thinking: bool, out: mpsc::Sender<Msg> }
enum Msg { Token(String), Done(GenResult) }
```

```
loop {
    let first = rx.recv()?;                    // block for at least one
    let mut group = vec![first];
    while group.len() < MAX_CONCURRENT {       // drain whatever else is waiting
        match rx.try_recv() { Ok(j) => group.push(j), Err(_) => break }
    }
    let k = group.len();
    let mut next = Vec::new();
    for (s, job) in group.iter().enumerate() {
        next.push(model.prefill_seq(dev, &encode(job), &mut state, &mut sc, s)?);
    }
    // step all k together; a finished slot keeps being stepped with its last
    // token so the slot indices stay put and no state has to be moved.
    loop {
        let mut live = 0;
        for s in 0..k { emit(group[s], next[s]); if !finished(s) { live += 1 } }
        if live == 0 { break }
        next = model.step_batch(dev, &next, &mut state, &mut sc)?;
    }
}
```

3. **`handle_*` changes are mechanical**: each currently calls
   `eng.generate(messages, max_tokens, thinking, on_token)`. Introduce a
   `generate_remote(&tx, ...)` with the *same signature and return type* that
   sends the `Job`, then forwards `Msg::Token` into `on_token` and returns the
   `GenResult` from `Msg::Done`. The response-writing code below it does not
   change at all.

The one correctness trap: a finished slot must keep contributing a token to
`next` (its own last token is fine) so `k` stays constant for the group; the
alternative, compacting slots, would require moving per-slot recurrent state and
KV cache and is not needed.

### Batching window, and where the time actually goes (round 95)

The group was whatever happened to be queued when the first request arrived --
`try_recv` drains only what is already there. Instrumented with a group-size log
(`GB10_BATCH_LOG=1`), a 25 ms window after the first `recv` turns that into
**two groups of exactly 16**.

Separating prefill from decode by varying `max_tokens`:

| max_tokens | wall (16 concurrent) |
|---|---|
| 1 | 9.45 s |
| 16 | **15.73 s** (was 21.80) |

so the intercept is prefill + 1 step and the slope is the step cost:

| part | before window | after |
|---|---|---|
| prefill | ~8.8 s | **~9.0 s** |
| decode | 15 x 815 ms = 12.2 s | 15 x 419 ms = 6.7 s |

**Aggregate is now 256/15.73 = 16.28 tok/s**, against 5.36 for the serial server
-- **3.0x** -- and against the 30 tok/s target.

**Prefill is now the majority (57%) and the next target.** The 9.0 s is 16
serialised `prefill_seq` calls of a 58-token prompt at ~563 ms each, which matches
the measured TTFT of 566 ms exactly. Batching prefill means running several
prompts through one forward with a per-sequence causal mask -- the same O(T^2)
`attn_prefill_kernel` that the TTFT work is already aimed at, so **the two open
performance items have the same fix**.

Note the step cost is 419 ms against `forward-cost`'s 287 ms for t=16; the extra is
per-step channel traffic and HTTP writes for 16 sequences, not compute.

### The scheduler is implemented and working (round 94)

`crates/gb10-server/src/main.rs` now has a batching scheduler: `Engine` moved onto
a scheduler thread, the accept loop spawns a thread per connection, and
`remote_generate` has the same signature as `Engine::generate` so the two request
handlers did not change at all. Each group prefills into slots `0..k`, then steps
them together with `step_batch`; a finished slot keeps being stepped with its own
last token so slot indices stay put.

**The objective's end-to-end test, same command as round 92:**

| | round 92 (serial) | round 94 (batched) |
|---|---|---|
| 16 concurrent, 256 completion tokens | 47.73 s | **21.63 s** |
| aggregate | 5.36 tok/s | **11.84 tok/s** |

**2.2x.** Correctness: `GET /health` ok, a single request returns normally, and
**all 16 identical concurrent prompts produced exactly one distinct output**, so
grouping is deterministic and does not perturb results.

Still short of the 30 tok/s the objective asks for. The remaining wall time
decomposes as roughly 16 serialised prefills of a 58-token prompt (~8 s at ~500 ms
each) plus 16 batched decode steps (~4.6 s at ~287 ms/step). **So the next lever
is prefill, not decode**: the group's prompts are still prefilled one at a time
into their slots, and that is now the majority of the time.

**The key enabler is confirmed:** `step_batch` sizes itself from `tokens.len()`
(`model.rs:277`), so a group of `k` requests can be prefilled into slots `0..k` and
stepped with `k` tokens -- no state compaction needed. A single request is just
`k = 1` and costs nothing extra. The remaining work is the scheduler itself: a
worker thread owning `Engine`, an `mpsc` queue of pending requests each carrying a
token channel, a loop that drains up to 16 and calls `step_batch`, and an accept
loop that spawns a thread per connection.

## BREAKTHROUGH (round 85): route short prompts to the batched GEMV

The single biggest win of the project. `forward_prefill` was routing everything
with `n >= 256` to the tiled GEMM; that GEMM's weight loads run at ~58 GB/s
against ~148 GB/s for the GEMV path, because the GEMM's staging scatters each
load instruction across 8 rows to fill a shared tile while the GEMV has no tile
and streams 256 contiguous bytes of one row per instruction.

The batched GEMV reads each weight once and reuses it across all `t` activations,
so it wins whenever re-reading W is cheaper than the GEMM's slower loads:

| t | GEMM | batched GEMV | |
|---|---|---|---|
| 1 | 271.46 | **118.40** | **-56%** |
| 2 | 269.40 | 148.51 | -45% |
| 4 | 280.26 | 159.97 | -43% |
| 8 | 277.47 | 194.37 | -30% |
| 16 | 286.93 | 292.10 | +2% (kept on the GEMM) |

The crossover is between 8 and 16, so the dispatch is `n < 256 || t <= 8`, with
8 the conservative cut (the 8..16 range is unmeasured). `generate` 16/16 exact,
`chunked-prefill` OK, `batch-parity` OK.

**This is also the answer to the whole GEMM investigation.** Rounds 60-84 tried to
make the GEMM's loads match the GEMV's pattern. The constraint computed in round
85 explains why they could not: one warp instruction covers 256 bytes = 512 NVFP4
elements, so one-row-per-instruction needs a 512-deep tile, which is 65536 B at
TN=64 -- over the 49152 B budget. **The GEMV is fast precisely because it has no
tile to fill.** The fix was never to speed up the GEMM; it was to stop using it
for short prompts.

Next: measure the 8..16 crossover properly and raise the cut, and check whether
the fp8 and bf16 paths want the same treatment.

### TTFT: the quoted baseline was stale (round 86)

The dispatch change does **not** move TTFT, and the reason is now clear: the real
prefill scores all 59 prompt tokens in a single `prefill_seq` call, so `t = 59`,
which is above the cut and still takes the GEMM.

Measured properly, before and after:

| | TTFT | decode |
|---|---|---|
| baseline (GEMM for t > 8) | 560.2 ms | 8.56 tok/s |
| batched GEMV for t <= 8 | 563.0 ms | 8.62 tok/s |

Neutral, as expected.

**The number quoted for TTFT throughout this document -- 452.6 ms -- is from
round 27**, i.e. before rounds 60, 62 and 63 cut the GEMM forward by 28%. It was
never re-measured and should be replaced everywhere by **560.2 ms**. Worse, the
GEMM got faster while TTFT got *slower* (452.6 -> 560.2), which is unexplained and
worth its own investigation: either the round-27 figure was taken under different
conditions, or something in rounds 36-59 cost prefill time that the `forward-cost`
benchmark does not see.

**Do not quote 452.6 ms again.** The measured baseline is 560.2 ms against
llama.cpp's ~74 ms, so TTFT is ~7.6x off rather than ~6x.

Since `t = 59` takes the GEMM, and chunking the prefill would make it worse (each
chunk re-reads every weight, so 8 chunks of 8 cost 8x the traffic of one
59-token pass), the TTFT problem is the GEMM at moderate `t` -- which is exactly
the case the dispatch change cannot help. The 8..16 crossover should still be
measured, but it will not address TTFT.

### The cut is confirmed at 8 (round 87)

Three thresholds on the same grid:

| cut | t=1 | t=2 | t=4 | t=8 | t=16 |
|---|---|---|---|---|---|
| `t <= 8` | 118.40 | 148.51 | 159.97 | 194.37 | 292.10 |
| `t <= 12` | 117.21 | 146.17 | 152.39 | 192.00 | 293.75 |
| `t <= 16` | 118.72 | 149.82 | 160.37 | 191.70 | **350.59** |

`t <= 8` and `t <= 12` are indistinguishable on this grid (which samples 1, 2, 4,
8, 16 -- not 12), and `t <= 16` is clearly worse at t=16. So **`t <= 8` is
correct** and no change is needed.

**One thing this rules out:** `GEMV_BATCH_MAX` is already 16 (`kernels.rs:21`,
`GB10_BATCH_MAX` in `gemv.cu:340`), so the batched kernel -- the one that reads W
once and reuses it across all `t` -- *is* used at t=16. Its loss there is real,
not a fallback to the per-token path. At batch 16 the GEMV does 16 FMAs per weight
element and becomes compute-bound, which is exactly where the tiled GEMM's reuse
wins. Raising the batch limit further would not help; the crossover is genuine.

So the dispatch is settled: **`n < 256 || t <= 8`**. TTFT is a separate problem,
and it lives in the GEMM at moderate `t` (59 tokens -> 560 ms, i.e. only ~31 GB/s
of weight traffic against the GEMM's own 58 GB/s). **That gap -- prefill at t=59
running at half the GEMM's measured rate -- is the next thing to explain**, and it
is a fresh anomaly rather than a continuation of the staging work.

### The prefill anomaly is explained, and the fix is blocked by shared memory (round 88)

`nsys profile` on `generate --n 2` (one 59-token prefill, 16 decode steps):

| kernel | % | launches | avg |
|---|---|---|---|
| `nvfp4_gemv_kernel` | 41.1 | 3,089 | 290.6 us |
| `fp8_gemv_kernel` | 29.1 | 3,328 | 190.9 us |
| `nvfp4_gemm_kernel` | 14.2 | **192** | **1,619.6 us** |
| `fp8_gemm_kernel` | 9.1 | **208** | **959.2 us** |
| `rmsnorm_zero_centered` | 1.3 | 2,737 | 10.6 us |
| everything else | <1 each | | |

192 = 64 layers x 3 nvfp4 matrices, i.e. **exactly one prefill pass**. Prefill is
311 + 199.5 = **510.5 ms of 564.5 ms TTFT -- 90%**. The GEMV kernels are the
decode.

**There is no anomaly.** `nvfp4_gemm_kernel` costs 1,619.6 us at t=59 against
825.6 us at t=1, and the reason is simply that **T=59 needs ceil(59/32) = 2
t-tiles, so every weight is read twice**:

```
192 launches x 44.6 MB x 2 tiles = 17.1 GB in 311 ms = 55 GB/s
```

which is the GEMM's normal rate. The "31 GB/s" was an artefact of dividing by
half the real traffic.

**The fix is to raise `GB10_TT` to 64 so T=59 fits in one tile**, halving prefill
traffic. Attempted this round: it **fails the generate gate** (0/1, diverges at
index 0) because it overflows shared memory --

| | xt | wt | total | budget 49152 |
|---|---|---|---|---|
| TT=32 (current) | 9,216 | 17,408 | 26,624 | ok |
| TT=64 | 34,816 | 17,408 | **52,224** | **OVER** |
| TT=64, bf16 `xt` | 17,408 | 17,408 | 34,816 | ok |
| TT=64, `KC`=32 | 34,816 | 8,704 | 43,520 | ok |

The TTFT reading of 469.2 ms taken during that attempt was **invalid** -- the
kernel was broken and decoded one token -- and must not be quoted. Reverted;
baseline re-verified at TTFT 569.1 ms with agreement 16/16.

**So the next change is a tile-shape change, not a load-pattern change.**

### TT=64 is blocked by the accumulator layout, not shared memory (round 89)

TT=64 with `KC`=32 -- the combination that *does* fit -- was tried and **also
fails the generate gate (0/1)**. So the shared-memory budget was not the real
blocker.

The real one is the outer product's accumulator shape. `GB10_TT` is the **column**
count of the tile, and the shared declaration shows it:

```cuda
__shared__ uint16_t wt[2][GB10_KC][GB10_WSTRIDE];   // [k][n]
__shared__ float    xt[2][GB10_KC][GB10_XSTRIDE];   // [k][t]   <- KC rows, TT cols
```

`xt` is indexed by `GB10_KC` in the first dimension, so `GB10_TT` only sets the
width. Each thread holds `acc[GB10_TM][GB10_TNREG]` = `acc[4][4]` = **16
accumulators**, and the 128 threads are mapped as 16 `ty` groups x 8 `tx` groups,
covering `16*4 = 64` rows by `8*4 = 32` columns. **That is a 64x32 tile -- TT=32,
baked into the thread mapping and the register count, not into a `#define`.**

So raising `GB10_TT` to 64 leaves the top half of every tile unwritten, which is
exactly the 0/1 divergence at index 0 that both attempts produced. **TT=64 needs
`acc[4][8]` (32 registers per thread) and a 16x4 thread mapping** -- a rewrite of
the outer product and its launch, not a constant change, and one that will cost
registers in a kernel already sensitive to them.

Also worth correcting from round 88: the shared-memory table there attributed
`xt = 2*64*68*4 = 34816` to TT=64, but that is the `KC`=64 figure. With `KC`=32
the real total is `2*32*68*4 + 2*32*68*2 = 26,112 B`, comfortably inside budget.
The budget is a constraint on *combinations*; the accumulator shape is the actual
blocker.

**Conclusion: the prefill's 2x weight traffic is real and worth 90% of TTFT, but
capturing it requires reworking the outer product's thread mapping to a 64x64
tile.**

### The 64x64 tile is implemented and correct -- and it is a wash (round 90)

Done: `GB10_TT` 32->64, `GB10_KC` 64->32 (to fit shared: 26,112 B), `GB10_TNREG`
4->8, and both outer products widened to two `float4` x-loads plus eight
accumulators per row. `GB10_TILE_T` -> 64 in `ops.rs`.

**It is correct** -- `generate` 16/16 (100%), `chunked-prefill` OK, `batch-parity`
OK. Results:

| | TT=32 (baseline) | TT=64 |
|---|---|---|
| TTFT | 555.7 ms | **516.0 ms (-7.1%)** |
| t=1 | 116.56 | 111.14 |
| t=2 | 145.98 | 146.81 |
| t=4 | 156.80 | 154.99 |
| t=8 | 193.36 | 193.19 |
| **t=16** | **292.10** | **465.47 (+59%)** |

TTFT improves 7.1%, but `t=16` regresses 59%: a 64-column tile evaluated at T=16
wastes three quarters of its outer product. **Reverted.** TTFT is an explicit
goal, but a 7% gain there does not pay for a 59% regression on short prompts,
which are the common interactive case.

Note the TTFT gain is far below the ~50% the traffic halving predicts, because
`KC`=32 doubles the chunk count and therefore the barriers -- the same
serialisation that has capped every staging change since round 60.

**The fix is to instantiate both and dispatch on `T`**, which captures TTFT -7%
*and* restores `t=16`:

* template the kernel body on `(TT, TNREG)` and emit two entry points --
  `nvfp4_gemm_kernel` with `(32, 4)` and `nvfp4_gemm_kernel_tt64` with `(64, 8)`;
* `GB10_XSTRIDE` must be templated too, since it is `TT + 4`;
* `KC` can stay 32 for both, so the shared budget is 26,112 B either way;
* `ops.rs` selects on `t > 32`, keeping `GB10_TILE_T` per instantiation for the
  grid.

That is mechanical and self-contained, and the correctness test is the same
`generate` gate.

## Verified state

`generate` 16/16 exact, `chunked-prefill` OK, `batch-parity` OK. t=1 forward
272.75 ms, t=16 286.93 ms. Tree is clean at the best-known state; every round has
been pushed to `GB10-Engine`.

The engine is complete and usable: pure Rust, NVFP4 on GB10, OpenAI and Anthropic
endpoints, concurrency-16 correct, and otp better than llama.cpp. The open items
are performance only.

## What is actually measured (trust these)

The only sound numbers in this document are the profiler's per-launch figures and
component tests done by changing a size. Everything derived from disabling a
component and subtracting has been retracted.

`nsys profile --stats=true` on `forward-cost`:

| kernel | % GPU | avg | min |
|---|---|---|---|
| `nvfp4_gemm_kernel` | 32.7 | 825.6 us | 635.9 us |
| `nvfp4_gemv_kernel` | 24.9 | 301.2 us | 213.9 us |
| `fp8_gemm_kernel` | 20.1 | 468.3 us | 238.3 us |
| `fp8_gemv_kernel` | 17.1 | 192.9 us | 26.0 us |
| all non-GEMM ops | ~5 total | | |

The four GEMM/GEMV kernels are 94.8% of GPU time; non-GEMM work cannot pay more
than 5%.

Per `nvfp4_gemm_kernel` launch (44.6 MB of weights), by halving components and
reading the profiler:

| part | per launch | share |
|---|---|---|
| weight loads | ~630 us | 76% |
| outer product | ~170 us | 21% |
| dequant + shared stores | ~2 us | 0.2% |

**So the loads run at 71 GB/s average (104 at minimum) against 148 GB/s for the
GEMV path on the same weights.** That gap is the target.

## The next change, with a reference implementation in this repo

`nvfp4_gemv_tmpl` in `kernels/gemv.cu` solved this same problem and its comment
says how:

> *Issue every row's weight and scale load BEFORE any of the arithmetic. Fusing
> the loads into the compute loop leaves the compiler free to keep only one load
> in flight per warp, which starves DRAM; separating them gives ROWS independent
> loads.*

Its addressing makes **one warp instruction cover 256 contiguous bytes of a
single row** (`e0 = i*kTile + lane*kVec`, 8-byte loads). The GEMM staging instead
scatters one instruction across 8 rows at 32 B each, and fuses load with dequant
and store.

The fix has two halves that must go together:

1. **One row per warp instruction** -- change the staging's thread-to-(row, k)
   mapping. Rounds 76, 77 and 79 all kept the 8-row scatter and only varied the
   bytes per row, which is why none of them moved anything.
2. **Issue all loads before arithmetic** -- round 83 tried this alone and it was
   *slower*, consistent with the GEMV comment: separating loads only helps when
   there are genuinely independent loads to separate.

Note the shared-memory constraint: one row per instruction with 8-byte lanes
means 256 bytes = 512 k-elements, and a 512-deep `wt` tile does not fit
(`2*512*68*2 = 139264 B`). Resolving that is the design problem for the next
session -- the likely answer is a row-major shared tile with a different consumer
loop in `gemm2d_outer_bf16`.

## Methodology rules learned here

* **Probe before optimising** -- this produced every real gain (rounds 60, 62, 63:
  -28%).
* **Change a component's size and measure; never disable a component and
  subtract.** Disable-probes over-attribute badly (round 81).
* **A probe measures what a pattern can reach, never what the kernel is blocked
  by** (round 79).
* **A probe that varies two things at once is as misleading as a guess** and more
  dangerous, because it looks like evidence (round 75/76).
* **Profile first.** Three rounds were spent on a misdirected line that one
  `nsys` run would have avoided (round 77).
* **Read the codebase.** The answer to the GEMM problem was documented in
  `gemv.cu` the whole time (round 83).
* Revert anything whose gain is inside the +-2% run-to-run noise, even when the
  sign is consistent (round 77).

## Honest target assessment

Single-decoder 100 tok/s is arithmetically unreachable: 17.6 GB/token against
228 GB/s measured gives a **roofline of ~13 tok/s**. Concurrency-16 at 30 tok/s is
**already met** (40.75). TTFT better than llama.cpp is still open (452.6 ms vs
~74 ms), with prefill improved 27-28% so far.

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

### Step 2 landed -- and refuted its own hypothesis (round 58)

KC=64 is in, correct, and verified: `generate` **16/16 exact**,
`chunked-prefill` agrees. All three GEMMs now stage 35840 B of tile, twice the
bytes in flight per chunk that KC=32 gave.

**It changed almost nothing.**

| t | KC=32 | KC=64 |
|---|---|---|
| 1 | 386.13 | 377.31 |
| 2 | 384.85 | 378.68 |
| 4 | 389.01 | 379.00 |
| 8 | 396.70 | 389.52 |
| 16 | 404.59 | 401.48 |

~2%, when the hypothesis predicted a move from 41 GB/s toward 100+. So
**"too few bytes per thread per chunk" is not the cause either.** That is the
third explanation eliminated, after occupancy (round 56: fp8 2 -> 3 blocks/SM
changed nothing) and tile size (round 53).

What survives is that the cost is *fixed per launch* and insensitive to tile
geometry, occupancy and bytes-in-flight.

### It is the kernel, not the cadence (round 59)

`nsys` per-launch duration distribution, which separates "the kernel is slow"
from "the launches are poorly scheduled":

| kernel | launches | min | avg | max |
|---|---|---|---|---|
| `nvfp4_gemm_kernel` | 2880 | **0.992 ms** | 1.226 ms | 2.776 ms |
| `fp8_gemm_kernel` | 3120 | 0.338 ms | 0.629 ms | 2.340 ms |

The **minimum** nvfp4 launch is 0.992 ms, and min-to-max is only 2.8x. So there
is no fast case being diluted by a slow tail: a best-case launch still moves
44.6 MB at ~45 GB/s, a fifth of the roofline. This rules out the cadence, the
tail and the launch count.

That leaves the response surface, which is now measured in three directions and
flat in all of them:

| lever | change | effect on the forward |
|---|---|---|
| occupancy (fp8) | 2 -> 3 blocks/SM | none (round 56) |
| bytes in flight | KC 32 -> 64 | ~2% (round 58) |
| tile shape | TT 64 -> 32 | included above |

A bandwidth-bound kernel that ignores occupancy *and* bytes-in-flight is not
waiting on DRAM, so the next step was a probe rather than a theory.

### The probe, and the fix (round 60)

Disabling the nvfp4 weight staging outright -- wrong results, timing only --
dropped the t=1 forward from 377.31 ms to **227.30 ms**. So the staging pass
alone was 150 ms, moving 9.63 GB at **64 GB/s**, 28% of roofline.

That is the signature of too little memory-level parallelism, not of shared
traffic or ALU. The staging loop issued **one 4-byte load per thread per
iteration**, and with 128 threads x 2 blocks/SM there were only ~256 outstanding
4-byte loads per SM -- about 1 KB in flight where 228 GB/s x ~600 ns needs
~2.8 KB. Occupancy and tile size could never fix that, which is exactly why
both came back flat.

The fix is to issue every load the thread owns *before* consuming any of them.
`stage_wtile` and `stage_wtile_fp8` now run in two passes -- load into a small
register array, then dequantise and store -- so `P = TN*SEGS/BLOCK = 4` loads
are in flight per thread instead of one.

| t | before | after | |
|---|---|---|---|
| 1 | 377.31 | **312.27** | -17% |
| 2 | 378.68 | 312.22 | -18% |
| 4 | 379.00 | 319.96 | -16% |
| 8 | 389.52 | 323.73 | -17% |
| 16 | 401.48 | **341.81** | -15% |

`generate` is still **16/16 exact**. This is the first change in this sequence
that moved the number, and it moves TTFT, which is one of the objective's
targets.

Note what it does *not* say: the earlier flat results were not wasted. They are
what ruled out occupancy and tile shape, leaving memory-level parallelism as the
only remaining explanation -- and that one was then confirmed by a probe before
being acted on.

### Decomposing what is left (round 61)

Same method: disable one stage at a time and read the delta off the t=1 forward.
Each probe is timing-only and was reverted immediately; the tree is at the
round-60 state, `generate` 16/16 exact.

| stage disabled | t=1 forward | implied cost | share |
|---|---|---|---|
| (none) | 312.27 ms | -- | -- |
| nvfp4 weight staging | 227.30 ms | ~85 ms | 27% |
| fp8 weight staging | 256.65 ms | ~56 ms | 18% |
| outer product, 3/4 of k | 263.91 ms | ~64 ms | 20% |
| **remainder** | | **~107 ms** | **35%** |

The probes overlap (removing a stage also removes the compute that consumes it),
so these are upper bounds and the shares do not sum cleanly. What they do say is
that **no single stage dominates any more**: the two stagings are 45% between
them, the outer product 20%, and the largest single bucket is now the
"everything else" remainder at ~35%.

That remainder is `stage_xtile`, the bf16 GEMM, `gemm2d_store`, and the
non-GEMM ops (rmsnorm, rope, and the rest). **The next probe should attack it
directly** -- disable `stage_xtile` and the bf16 path in turn -- because at 107 ms
it is now bigger than any individual stage, and the round-60 result showed the
method works: a probe that isolates one stage is what turns a flat response
surface into a fix.

Also worth noting for scale: fp8's staging moves 5.59 GB in ~56 ms = 100 GB/s,
and nvfp4's moves 9.63 GB in ~85 ms = 113 GB/s. Both roughly doubled from the
64 GB/s that prompted the round-60 fix, so the two-pass load is doing its job --
but both are still around half of the 228 GB/s roofline, so the same MLP
question should be asked of them again with a larger `P`.

### Wider loads: 16 elements per thread (round 62)

Acting on that last sentence. `stage_wtile` now has each thread own 16
consecutive NVFP4 elements rather than 8, fetched as a single `uint2` and
covered by one group scale. That halves both the staging iterations and the
number of load instructions for the same bytes.

| t | round 60 | round 62 | |
|---|---|---|---|
| 1 | 312.27 | **287.58** | -8% |
| 2 | 312.22 | 284.42 | -9% |
| 4 | 319.96 | 288.52 | -10% |
| 8 | 323.73 | 295.93 | -9% |
| 16 | 341.81 | **312.19** | -9% |

`generate` still **16/16 exact**.

Cumulative against the round-59 baseline, both changes together:

| t | baseline | now | |
|---|---|---|---|
| 1 | 377.31 | **287.58** | **-24%** |
| 16 | 401.48 | **312.19** | **-22%** |

### The same widening on fp8 and xtile (round 63)

`stage_wtile_fp8` now takes 16 elements per thread as one `uint4` (fp8 carries a
single per-tensor scale, so there is nothing to index), and `stage_xtile` takes
16 k-values as four `float4`s. The latter matters most at small `t`: it used to
be one iteration of almost entirely zero-writes whenever `t < T`, which at t=1
is every thread but one.

| t | round 62 | round 63 | |
|---|---|---|---|
| 1 | 287.58 | **271.46** | -6% |
| 2 | 284.42 | 271.85 | -4% |
| 4 | 288.52 | 273.77 | -5% |
| 8 | 295.93 | 280.94 | -5% |
| 16 | 312.19 | **292.85** | -6% |

`generate` still **16/16 exact**.

### Cumulative

| t | round-59 baseline | now | |
|---|---|---|---|
| 1 | 377.31 | **271.46** | **-28%** |
| 4 | 379.00 | 273.77 | -28% |
| 16 | 401.48 | **292.85** | **-27%** |

Three changes -- two-pass loads (round 60), 16 elements per thread for nvfp4
(round 62), and the same for fp8 and xtile (round 63) -- have taken the prefill
forward down by ~28%. All three came from the same method: isolate one stage with
a timing probe, then fix the specific thing the probe exposed.

### Re-decomposition against the current numbers (round 64)

The round-61 shares had moved, so the same probes were re-run against the
271.46 ms baseline:

| stage disabled | t=1 forward | implied cost | share |
|---|---|---|---|
| (none) | 271.46 ms | -- | -- |
| nvfp4 weight staging | 200.52 ms | ~71 ms | 26% |
| fp8 weight staging | 230.03 ms | ~41 ms | 15% |
| outer product, 3/4 of k | 229.77 ms | ~56 ms | 21% |
| xtile loads | 289.61 ms | none measurable | -- |

Two things stand out.

**The two stagings together are still 41%** -- 112 ms to move 15.2 GB, i.e.
~136 GB/s, up from 64 GB/s before round 60 but still only 60% of the 228 GB/s
roofline. The widening helped; it did not finish the job.

**The remainder is now the largest single bucket at ~103 ms (38%)**, and the
`xtile` probe came back with *no* measurable effect -- removing its global loads
changed nothing. Since `stage_xtile` still writes every one of its shared slots
(zeros for the `t >= T` lanes, which at t=1 is all but one lane per group), that
points at the **shared-memory stores** rather than the loads, or at something
outside `stage_xtile` entirely: `gemm2d_store`, the bf16 GEMM, or the non-GEMM
ops.

Note the probe for `xtile` was the weakest of the four -- it disabled the load
branch only, so a null result there is suggestive rather than conclusive, and it
should be redone by skipping the whole function.

### The xtile probe redone, and a fix that did not work (round 65)

The round-64 `xtile` probe was the weakest of the four -- it disabled only the
load branch. Redone by skipping the whole function:

| | t=1 forward |
|---|---|
| baseline | 271.46 ms |
| all `stage_xtile` calls disabled | **200.52 ms** |

So `stage_xtile` really does cost ~71 ms (26%), and since disabling the loads
alone had shown nothing, the cost is in the **shared stores**.

At `t=1` the block is partial (`T=1 < TT=32`), so 31 of every 32 columns take
the `else` branch and write zeros -- and `gemm2d_store` guards every token with
`if (t < T)`, so those columns are never read back. That looked like 71 ms of
dead work, and removing it was implemented and measured:

| t | with zero-fill | without | |
|---|---|---|---|
| 1 | 271.46 | 278.66 | +2.6% |
| 4 | 273.77 | 280.75 | +2.6% |
| 16 | 292.85 | 307.80 | **+5.1%** |

`generate` and `chunked-prefill` both stayed correct, but it is **slower**, so it
was reverted. The likely reason is divergence: with the zero-fill every thread
runs the same store sequence, while removing it makes the `t < T` lanes skip out
of the inner loop and splits the warp. The stores were dead, but they were also
free -- they filled a branch that would otherwise have diverged.

That is a useful negative result. It means `stage_xtile`'s 71 ms is not
recoverable by deleting work; it needs the stores restructured so the tile is
filled without per-lane branching, or the tile layout changed so that partial
blocks do not need 32 columns at all.

### Where the widening stops paying (round 66)

The widening from 8 to 16 elements per thread paid (round 62, -8%). Carrying it to
32 does not:

| elements/thread | loads in flight | t=1 | t=16 |
|---|---|---|---|
| 8 | P=4 x 4 B = 16 B | 312.27 | 341.81 |
| **16** | **P=2 x 8 B = 16 B** | **271.46** | **292.85** |
| 32 | P=1 x 16 B = 16 B | 276.11 | 301.65 |

All three configurations put the same 16 bytes in flight per thread, so the
bytes-in-flight is not what separates them. What changes is the *number of
independent loads*: 4, then 2, then 1. Going 8 -> 16 helped because it halved the
instruction count without reducing the load count below 2; going 16 -> 32 crosses
that line and buys instruction count at the cost of memory-level parallelism.

So **16 elements per thread with P=2 is the optimum for this geometry**. Note
that `P` itself is not the lever: the total loads in flight is `UNITS`, not
`P * BLOCK`, so redistributing the same units over fewer threads changes nothing.
The lever is `UNITS = TN * KC/16`.

That was tested directly by doubling `TN` (64 -> 128) and halving `TT` (32 -> 16),
which doubles `UNITS` to 512 and still fits the shared budget at 44032 B:

| | t=1 | t=16 | registers | smem |
|---|---|---|---|---|
| TN=64, TT=32 | **271.46** | **292.85** | 52 | 35840 |
| TN=128, TT=16 | 303.62 | 317.10 | **122** | 44032 |

Correct (`generate` 16/16) but **12% slower**. Doubling `TN` doubles the live
accumulator and index state per thread -- 122 registers against 52 -- and pushes
the tile to one block per SM. The extra loads in flight do not compensate.

So the geometry is boxed in from both sides: `UNITS` cannot grow without `TN`,
and `TN` cannot grow without register pressure.

### Occupancy, tested a second time (round 68)

The remaining hypothesis was the shared-memory budget capping occupancy at 2
blocks/SM. Dropping the `+4` padding (`WSTRIDE = TN`, `XSTRIDE = TT`, both still
multiples of 4 so the `float4`/`ushort4` loads stay aligned) fits a third block:

| | t=1 | t=16 | registers | smem | blocks/SM |
|---|---|---|---|---|---|
| padded | **271.46** | **292.85** | 52 | 35840 | 2 |
| unpadded | 271.99 | 294.19 | 104 | 32768 | 3 |

Correct (`generate` 16/16) and **completely neutral** -- and it cost registers,
52 -> 104, presumably from the different unrolling. Reverted.

This is the second independent test of occupancy on this kernel (the first was
fp8 going 2 -> 3 blocks/SM in round 56) and both came back flat. Combined with
the flat response to tile shape and to bytes in flight, the picture is now
consistent: **the GEMM is not waiting on DRAM, and it is not short of warps.**

That leaves the ~271 ms as internal work: the dequantisation and shared stores
in the staging passes, the outer product's shared reads, and the non-GEMM ops.
The round-65 result -- that `stage_xtile`'s 71 ms is stores, and that removing
the dead zero-fill made things *slower* through divergence -- is the clearest
clue about what kind of change would help: not deleting work, but removing
per-lane branching from the shared-memory fill.

Worth noting for perspective: 271 ms for 17.6 GB is 65 GB/s, while the decode
GEMV reaches 148 GB/s on the same weights. So the ceiling is not the memory
system; the GEMM path is roughly 2.3x less efficient than the GEMV path at
identical traffic.

### Three ways to fill the xtile, and the boring one wins (rounds 65-69)

`stage_xtile` costs ~71 ms (26%) of the forward and the cost is in its shared
stores. Three variants have now been measured:

| variant | t=1 | t=16 | |
|---|---|---|---|
| zero-fill the `t >= T` lanes (original) | **271.46** | **292.85** | best |
| skip those lanes entirely | 278.66 | 307.80 | +2.6% / +5.1% |
| branchless: redirect them to a spare column | 289.92 | 310.48 | +6.8% / +6.0% |

All three are correct (`generate` 16/16, `chunked-prefill` agrees). The original
wins, and the reason is now clear from having tried both alternatives:

* Skipping the lanes splits the warp, and the divergence costs more than the
  dead stores saved.
* Redirecting them keeps the store sequence uniform but makes those lanes load
  token `t0` as well, so it *adds* real global traffic to remove a branch.

So the 71 ms is not recoverable by touching the fill logic at all: the dead
stores are cheaper than either removing them or paying to keep them uniform.
Any further gain there has to come from a different tile layout that does not
need `TT` columns when the prompt is shorter than `TT` -- not from the fill.

That closes out the `stage_xtile` line of attack, and with it the shared-store
hypothesis from round 65. The remaining measured shares are the two weight
stagings (41% together, ~136 GB/s) and the outer product (21%).

### Where the two weight stagings stand (round 70)

Arithmetic on the staging, to work out what its 136 GB/s actually is:

* Coalescing is fine. Thread `u` reads `w + n*(K/2) + (c*KC + gr*16)/2` with
  `gr` fastest, so threads 0-3 cover 32 consecutive bytes of one row and a warp
  covers 8 rows x 32 B = 256 B in 8 fully-used sectors.
* Memory-level parallelism is fine. The two-pass loop issues 4 independent loads
  (2 units x 1 weight + 1 scale) per thread, 32 B in flight, ~8 KB per SM against
  the ~2.8 KB needed to cover 600 ns at 228 GB/s.
* Latency per chunk cannot explain it either: 80 chunks x 600 ns is 48 us per
  block, and ~2.8 waves of blocks gives ~0.14 ms -- nowhere near the 71 ms
  measured for the nvfp4 staging.

So 136 GB/s is a *bandwidth* figure, not a latency one, and neither coalescing
nor outstanding loads explain the shortfall from 228 GB/s. That leaves two
candidates, and they are cheap to tell apart:

1. **The dequantisation ALU.** Per 8-byte load the staging runs 16 `e2m1_to_float`
   calls plus 16 bf16 converts and 16 multiplies. If `e2m1_to_float` is a branch
   or a lookup rather than pure bit arithmetic, that is a lot of work per byte.
2. **The shared-memory stores**, which are bank-conflicted: `wt` has stride
   `TN+4 = 68` uint16 = 136 B = 34 banks, so consecutive `nl` land 2 banks apart
   and 32 threads hit 16 banks twice.

The discriminator is to replace `e2m1_to_float(nib) * s` with a plain constant
inside the staging (wrong results, timing only, as in round 60). If the staging
time collapses, it is the ALU and the fix is a cheaper dequant -- a 16-entry
`__constant__` table indexed by the nibble would do it, or hoisting the scale
multiply into the outer product where it is already done for fp8. If it does not
move, it is the shared stores and the fix is the tile layout.

This is the same method that produced every gain so far: isolate one thing,
measure, then act.

### Both candidates tested, both ruled out (rounds 71-72)

**Dequantisation ALU.** The 16 `e2m1_to_float` calls per 8-byte load were
replaced with a cheap value derived from the same register (keeping the loads
alive, timing only):

| | t=1 |
|---|---|
| baseline | 271.46 ms |
| dequant ALU removed | 280.85 ms |

No gain. **The ALU is not the limit.**

**Bank conflicts in the `wt` stores.** `WSTRIDE = TN+4 = 68` uint16 = 136 B =
34 banks, so rows land 2 banks apart and a warp's four rows overlap; padding to
`TN+16 = 80` puts rows 8 banks apart and makes the pattern conflict-free:

| | t=1 | t=16 |
|---|---|---|
| WSTRIDE = 68 | **271.46** | 292.85 |
| WSTRIDE = 80 | 272.26 | 290.42 |

Neutral. **Bank conflicts are not the limit either.**

### What that leaves

Every explanation on the list has now been tested and come back flat: DRAM
bandwidth, occupancy (twice), bytes in flight, load count, tile shape, load
width, dequant ALU, shared-store bank conflicts, and per-chunk latency (which
arithmetic rules out by three orders of magnitude -- 0.14 ms predicted against
71 ms measured).

The staging moves 9.63 GB in ~71 ms, which is 136 GB/s against a 228 GB/s
roofline. Since none of the mechanisms above explain the gap, the next thing to
question is **the roofline number itself**. 228 GB/s came from `bench/hw/bw4.cu`,
an incompressible random-access probe. The staging's pattern is different in two
ways that probe does not capture: a warp reads 8 rows that are 2560 B apart
rather than one contiguous run, and the working set is 44.6 MB of weights
streamed through a 25 MB L2. Either could hold the achievable rate below 228.

**The cheap next step is therefore to measure the achievable rate for this
exact pattern** -- a standalone probe that reads the real weight layout with the
real warp access pattern and nothing else.

### The roofline was wrong (round 73)

`bench/hw/stage_bw.cu` does exactly that: two kernels, same bytes, one reading
contiguously and one reproducing `stage_wtile`'s addresses with no dequant and
no shared store. On a 44.6 MB matrix (one layer's NVFP4 weights), 64 reps:

| pattern | GB/s |
|---|---|
| contiguous 16 B per thread | 206.2 |
| **staging (the real addresses)** | **89.7** |

**The staging pattern's own ceiling is ~90 GB/s, not 228.** The real kernel
reaches 136 GB/s on this pattern -- *above* what the isolated probe manages, so
the staging is not leaving performance on the table at all. It is at the limit of
the addresses it issues.

The reason is cache-line granularity. `KC = 64` NVFP4 elements is 32 bytes, and a
warp's four `gr` lanes cover exactly those 32 bytes of a row; the next row starts
2560 bytes away. So every row contributes a **32-byte sector to a 64-byte line**
-- half of each fetched line is never used. That predicts ~114 GB/s against the
228 GB/s contiguous figure, which is the right order for both the 90 measured in
isolation and the 136 measured in the kernel.

This reframes the whole line of attack. The two weight stagings were 41% of the
forward, but they are not underperforming; they are reading 32 bytes per row when
the hardware wants 64. **The fix is a layout change, not a tuning change**: get a
warp to cover 64 consecutive bytes of one row. The direct route is `KC = 128`,
which makes a row's chunk exactly one cache line -- but that needs
`2*128*(TN+4)*2 + 2*128*(TT+4)*4 = 71680 B` of shared against a 49152 B budget, so
it requires shrinking `TT` or storing `xt` in bf16 (both already scoped above).

That is the next experiment, and it is the first one in several rounds with a
clear mechanism behind it rather than an elimination.

### The hypothesis tested before acting on it (round 74)

Rather than restructure the kernel on the strength of a mechanism, the probe was
extended with a third pattern: same layout, same bytes, but 8 lanes x 8 B so a
warp covers **64 consecutive bytes of a row** instead of 32.

| pattern | bytes per row per warp | GB/s |
|---|---|---|
| contiguous | -- | 221.5 |
| staging32 (current) | 32 | 93.7 |
| **staging64** | **64** | **176.9** |

**Confirmed, and the effect is large: 93.7 -> 176.9 GB/s, +89%.** Cache-line
granularity is the mechanism; half of every fetched line was being discarded.

Projected on the real forward: the two stagings cost ~112 ms (41%) at ~136 GB/s.
At 177 GB/s the same bytes take ~86 ms, so **t=1 should fall from 271 to roughly
245 ms (-10%)**, and further if the kernel tracks the contiguous figure rather
than the isolated one.

### The change this implies, and its cost

`KC` has to become 128, because a row's chunk is `KC/2` bytes and only `KC = 128`
makes it a full 64-byte line. `KC = 128` divides every K in the model (5120/128 =
40, 17408/128 = 136, 6144/128 = 48, 10240/128 = 80), so the constraint is met.

The obstacle is shared memory. With `KC = 128`, `TN = 64`, `TT = 32`:

```
wt  2*128*(64+4)*2 = 34816
xt  2*128*(32+4)*4 = 36864
                    ------
                    71680   vs a 49152 budget
```

The options, in order of intrusiveness:

| wt | xt | total | fits |
|---|---|---|---|
| `TT=32`, xt float, pad 4 | 34816 + 36864 | 71680 | no |
| `TT=16`, xt float, pad 4 | 34816 + 20480 | 55296 | no |
| `TT=32`, xt bf16, pad 4 | 34816 + 18432 | 53248 | no |
| `TT=16`, xt bf16, pad 4 | 34816 + 10240 | **45056** | **yes** |
| `TT=32`, xt bf16, no pad | 32768 + 16384 | **49152** | exactly at the limit |

So it needs `TT = 16` **and** a bf16 `xt` tile, or a bf16 `xt` with no padding.
`TT = 16` also changes `gemm2d_ids` (32 row-groups x 4 token-groups) and
`GB10_TILE_T` in `ops.rs`, and a bf16 `xt` changes every `gemm2d_outer` reader.
That is a coupled change across three places, so it should be done in one step
with the `generate` 16/16 gate as the check, and reverted as a whole if the
timing does not improve.

**The cheapest version to try first is `TT=32` + bf16 `xt` + no padding**, since
it touches only the `xt` type and the two stride macros and leaves the tiling
alone -- at the cost of sitting exactly on the 49152 B limit, which may need a
little slack.

### A structural point that changes the cost (round 75)

Working out the `KC = 128` variants turned up something worth writing down before
any code changes, because it moves the problem.

The current tile is already **128 k-rows deep in total**: `wt[2][KC][WSTRIDE]`
holds two `KC = 64` chunks, 17408 B. A single `wt[128][WSTRIDE]` tile is *also*
17408 B. So `KC = 128` costs nothing in shared memory **if the double buffer is
given up** -- the same space simply holds one bigger k-tile instead of two
alternating ones.

That matters because it means the obstacle is not really shared memory. It is
that dropping the double buffer **serialises staging against the outer product**.
With staging projected at ~86 ms (at 177 GB/s) and the outer product at ~56 ms,
serialised they cost ~142 ms against the ~112 ms they cost overlapped today. The
cache-line win would be spent, and then some.

So `KC = 128` is only worth doing if the pipeline survives it. The shapes that
fit with double buffering all force `TT = 16` (and therefore a reworked
`gemm2d_ids`, plus a second accumulator group per thread at 128 threads), and the
ones that keep `TT = 32` sit exactly on the 49152 B limit.

The clean resolution is a **three-deep k-pipeline**: keep 128 k-rows of weights
resident, but stage the *next* 64 k-rows into the half that has just been
consumed, so the staging read still covers 64 contiguous bytes per row while
staying overlapped with compute. That needs the outer product to consume the two
halves in order, which is exactly what the current `cur`/`nxt` structure already
does -- the change is to stage both halves in one pass at the top and then
compute both, rather than staging one and computing one.

That is the version to build. It keeps `TN = 64`, `TT = 32`, the float `xt`
tile, and the 35840 B budget, and it changes only the staging/consumption order
in the three GEMM bodies.

### Built it, and the cache-line story is wrong (round 76)

The paired version was implemented for nvfp4 -- `stage_wtile_pair` stages two
adjacent k-chunks in one pass, so a warp's four `pr` lanes cover 64 contiguous
bytes of a row, and the body computes both halves before restaging. Correct
(`generate` 16/16) and **completely neutral**:

| | t=1 | t=16 |
|---|---|---|
| baseline | 271.46 | 292.85 |
| paired staging (64 B/row) | 271.18 | 295.43 |

Reverted.

**This falsifies the cache-line explanation.** Doubling the bytes a warp touches
per row, with the load count unchanged, does nothing. So the probe's 93.7 ->
176.9 GB/s was **not** caused by cache-line utilisation.

The probe was confounded. `read_staging64` changed two things at once:

* 8 lanes x 8 B per row instead of 4 lanes x 8 B -- the bytes-per-row difference
* `P = 4` instead of `P = 2` -- because `UNITS` went from 256 to 512

so it also doubled the number of independent loads in flight. The kernel
experiment separates the two and shows the bytes-per-row half is worth nothing.
The effect the probe measured must belong to the load count.

That is consistent with round 66, which found 16 elements per thread optimal and
32 worse: at 32 the loads in flight halve. So the through-line across rounds 60,
62, 66 and now 76 is **loads in flight**, and the honest state is that the
staging's 136 GB/s is explained by neither cache lines (now tested) nor any of
the other mechanisms eliminated above.

**The lesson is about the probe, not the kernel.** A probe that changes two
variables at once is as misleading as a guess, and it is more dangerous because
it looks like evidence.

### The clean 2x2, and why the probe still does not transfer (round 77)

The probe was rebuilt to vary one thing at a time -- three patterns, with the
load count and the bytes-per-row separated:

| pattern | bytes/row | loads per chunk | GB/s |
|---|---|---|---|
| 4 lane x 8 B | 32 | 256 | 90.4 |
| 8 lane x 8 B | 64 | 512 | 184.9 |
| 8 lane x 4 B | 32 | 512 | 102.7 |

* 8lane x 4B vs 8lane x 8B (load count fixed, bytes/row varies): 102.7 -> 184.9,
  **+80%**. Bytes-per-row is the dominant variable.
* 4lane x 8B vs 8lane x 4B (bytes/row fixed, load count varies): 90.4 -> 102.7,
  +14%. Load count is minor.

So the cache-line mechanism is real, and round 76's failure is explained: my
paired staging used **two 8-byte loads** that together covered 64 bytes of a row,
but the coalescer works per *instruction*, so each load still presented 8 rows x
32 B. Two requests covering the same line are not one request covering the line.

That was fixed -- a single **16-byte load** per thread, 4 lanes per row, so one
instruction's warp footprint is 8 rows x 64 contiguous bytes. The result:

| t | baseline | single 16 B load, 64 B/row |
|---|---|---|
| 1 | 271.46 | 270.21 |
| 2 | 284.42 | 271.11 |
| 4 | 273.77 | 269.78 |
| 16 | 292.85 | 290.75 |

Correct (`generate` 16/16), better at every point, and **every one of those
differences is inside the +-2% run-to-run noise measured on this box**. It is not
a defensible win, so it was reverted rather than kept on the strength of a
consistent sign.

**The conclusion is that per-warp request shape does not bind in the real
kernel**, even though it clearly binds in isolation. The likely reason is
concurrency: the isolated probe runs one warp's pattern against an otherwise idle
memory system, while the kernel has many warps from two blocks per SM issuing
interleaved requests, so the memory controller sees a much denser stream than any
single warp's shape suggests. A pattern that is 2x worse in isolation can be
indistinguishable when 20 other warps are filling the gaps.

That is worth stating plainly because it bounds what the probe can be used for:
`bench/hw/stage_bw.cu` is good for **falsifying** a mechanism (if a pattern is
slow in isolation it will not be fast in the kernel) but not for **predicting a
gain** from a pattern change.

### The kernel-level breakdown, which should have come first (round 78)

Every share quoted above came from probe deltas, which the last three rounds have
shown are unreliable in both directions. `nsys profile --stats=true` on
`forward-cost` gives the actual distribution instead:

| kernel | % GPU time | launches | avg | min |
|---|---|---|---|---|
| `nvfp4_gemm_kernel` | 32.7 | 2880 | 825.6 us | 635.9 us |
| `nvfp4_gemv_kernel` | 24.9 | 5998 | 301.2 us | 213.9 us |
| `fp8_gemm_kernel` | 20.1 | 3120 | 468.3 us | 238.3 us |
| `fp8_gemv_kernel` | 17.1 | 6448 | 192.9 us | 26.0 us |
| `rmsnorm_zero_centered` | 1.1 | 7406 | 11.0 us | 1.3 us |
| `bf16_gemv_batch` | 0.9 | 1344 | 46.8 us | 12.1 us |
| `gated_delta_rule_chunk` | 0.9 | 720 | 86.9 us | 34.8 us |
| `gated_delta_rule_step` | 0.7 | 1488 | 36.2 us | 26.5 us |
| everything else | <1% each | | | |

**The four GEMM/GEMV kernels are 94.8% of GPU time. Every non-GEMM op together
is ~5%.** The round-61/64 probes said the "remainder" was 35-38% and that
non-GEMM work was a plausible target; the profiler says it is not. Optimising
`rmsnorm`, `swiglu`, `rope` or the delta rule cannot pay more than ~5%.

**The number that matters is this.** `nvfp4_gemm_kernel` moves 9.2 GB per forward
across 15 forwards in 2.38 s, so the GEMM path runs at **~58 GB/s** (70 GB/s at
its minimum). `fp8_gemm_kernel` independently lands at the same ~58 GB/s. The
GEMV path, on the same weights, reaches ~148 GB/s.

So the finding is not about staging, cache lines, or shared stores: **the GEMM
path as a whole is 2.5x less efficient than the GEMV path at identical traffic,
and it is 94.8% of the time.** That is the thing to explain, and the profiler
gives a per-launch number to optimise rather than a probe delta.

A first estimate of where a launch's 825 us goes, for the MLP gate (N=17408,
K=5120, 272 blocks, 5.7 blocks/SM, ~2.8 waves at 2 blocks/SM):

* 164 KB of weights per block, 80 chunks of 2 KB each
* ~2.8 us per chunk measured
* the outer product is 1024 FMA per thread per chunk -- ~1.2 us per chunk at
  128 FP32 lanes/SM shared by two blocks
* the loads themselves should be ~0.6 us

which leaves roughly 1 us per chunk unaccounted for.

### Splitting the GEMM: staging 80%, outer product 20% (round 79)

Quartering `gemm2d_outer_bf16`'s k loop and reading the t=1 forward:

| | t=1 |
|---|---|
| baseline | 271.46 ms |
| outer product k-loop at 1/4 | 231.38 ms |

so the full outer product costs (271.46 - 231.38) / 0.75 = **~53 ms, 20% of the
forward**. The remaining ~218 ms is the staging, which the profiler puts at
~58 GB/s.

Putting the two measurements together, the t=1 forward decomposes as:

| part | ms | share | rate |
|---|---|---|---|
| weight staging (nvfp4 + fp8 + bf16) | ~218 | 80% | ~58 GB/s |
| outer product | ~53 | 20% | -- |
| non-GEMM ops | ~13 | 5% | -- |

**And the GEMV path moves the same weights at ~148 GB/s.**

So the isolated probe was right about the *relative* ordering after all: 32 B/row
measured 90 GB/s against 206 GB/s contiguous, and the kernel shows 58 GB/s
against 148. The probe predicted the comparison correctly; what it could not
predict is that my particular fix would not move it, because the fix changed the
bytes per row without changing the shape that actually matters.

The shape that matters is how many *different rows* a single load instruction
touches. The staging's load presents **8 rows x 32 B** to the coalescer; the
probe's fast pattern presented **4 rows x 64 B**; the GEMV streams one row
contiguously. My round-77 attempt produced **8 rows x 64 B** -- more bytes, same
eight-way row scatter, which is why it was neutral.

### Rows per instruction is not the variable either (round 80)

The 4-row hypothesis was tested by adding a fourth pattern that reproduces
**exactly** what round 77 built into the kernel -- a single 16-byte load per
thread, 4 lanes per row, so one instruction's footprint is 8 rows x 64 B:

| pattern | rows per instruction | bytes/row | GB/s |
|---|---|---|---|
| 4 lane x 8 B | 8 | 32 | 89.8 |
| 8 lane x 8 B | 4 | 64 | 176.9 |
| 8 lane x 4 B | 4 | 32 | 101.8 |
| **4 lane x 16 B (round 77's fix)** | **8** | **64** | **181.0** |

**8 rows x 64 B is just as fast as 4 rows x 64 B.** Rows per instruction is not
the variable; bytes per row is (32 B -> ~90-102, 64 B -> ~177-181).

**Which makes round 77's null result conclusive rather than puzzling.** The
kernel change did produce the 8-row x 64 B pattern, and that pattern measures
181 GB/s in isolation -- yet it was neutral in the kernel. So the staging is
**not limited by its DRAM access pattern at all**. Changing a 90 GB/s pattern
into a 181 GB/s pattern bought nothing.

This closes the DRAM line for good. Three independent attempts have now been made
to exploit the pattern (rounds 60, 76, 77) and the profiler's 58 GB/s has
survived all of them.

**What is left for the staging is the work that is not the load**: the 16
`e2m1_to_float` dequantisations and 32 shared stores per thread per unit, and the
shared traffic the outer product then reads back. Round 71 removed the
dequantisation ALU from the *old* staging and saw nothing, but that was before the
pattern was known to be irrelevant; it should be repeated, and the shared stores
tested the same way, now that the load side is ruled out.

Note also what this says about the probe: it correctly ranked the patterns, and
it correctly showed the round-77 pattern was fast. It simply cannot tell you
whether a fast pattern is what the kernel is waiting on. **A probe measures what
a pattern can reach, never what the kernel is blocked by.**

### The dequant and the stores are ruled out too (round 81)

With the load side excluded, the remaining staging work was cut 4x -- the
inner `j` loop in `stage_wtile` from 8 to 2, so one quarter of the
`e2m1_to_float` calls and one quarter of the shared stores, with the loads kept
live through `lo`/`hi`:

| | t=1 |
|---|---|
| baseline | 271.46 ms |
| staging dequant + stores at 1/4 | 266.84 ms |

**1.7%.** So the dequantisation and the shared stores are not the cost either.

That is now three separate exclusions inside the staging:

| mechanism | test | result |
|---|---|---|
| load pattern | probe + kernel change (76, 77, 79) | not the cost |
| dequant ALU | 4x fewer calls (71, 81) | not the cost |
| shared stores | 4x fewer stores (81) | not the cost |

### What that actually means

The staging was attributed **218 ms of 271 (80%)** by disabling it (round 64/71:
`if (false) stage_wtile(...)` -> 200.52 ms). But every component of the staging
has now been individually tested and none of them is expensive. Those two facts
cannot both be true of a well-behaved probe.

**The disable-probe is the thing that is wrong.** Setting `stage_wtile` to
`if (false)` does not merely remove the staging work: it removes the global
loads entirely, changes the kernel's shared-memory footprint and its scheduling,
and leaves the outer product reading uninitialised shared memory. It measures
"the kernel without this stage", which is not the same as "the cost of this
stage", and the difference is large.

So the decomposition quoted for the last several rounds -- staging 80%, outer
product 20% -- should not be trusted, and neither should the round-61/64 numbers
built on it. The outer-product figure (53 ms) came from shortening its k loop,
which is a genuine component test and does survive; the staging figure came from
a disable-probe and does not.

**The rule from here: change a component's size and measure, never disable a
component and subtract.** And prefer the profiler's per-launch number, which is
absolute.

The profiler numbers stand on their own and remain the target:
`nvfp4_gemm_kernel` 825.6 us average, 635.9 us minimum, moving 44.6 MB per
launch -- 54 GB/s average, 70 GB/s at best, against 148 GB/s for the GEMV path on
the same weights.

### A valid component test, and the gap located (round 82)

Following the new rule -- change a component's *size*, read the profiler's
absolute per-launch number -- the outer product's k loop was halved:

| `nvfp4_gemm_kernel` | avg | min |
|---|---|---|
| baseline | 825.6 us | 635.9 us |
| outer product k at 1/2 | 740.6 us | 532.2 us |
| **outer product, doubled back** | **~170 us** | **~207 us** |

That is 21-25% of the launch, and it agrees with the t=1 estimate (53 ms of 271,
20%) that came from the same kind of test. **The two independent methods agree,
which is what a valid measurement looks like.**

Combining with round 81 (dequant + shared stores: a 4x reduction moved the whole
forward by 4.6 ms, i.e. ~2 us per launch -- negligible), the launch decomposes as:

| part | per launch | share |
|---|---|---|
| weight loads | **~630 us** | ~76% |
| outer product | ~170 us | ~21% |
| dequant + shared stores | ~2 us | ~0.2% |

**So the loads move 44.6 MB in 630 us = 71 GB/s average, 104 GB/s at the
minimum -- while the pattern probe says this exact access pattern can reach
181 GB/s.**

That is the gap, now located precisely and by a sound method: **the kernel's
weight loads run at 39% of what their own access pattern is capable of.** It is
not the pattern (181 GB/s is achievable), not the dequant, not the stores, and
not the outer product.

What differs between the probe and the kernel is **concurrency**. The probe runs
272 blocks of pure loads against an idle memory system. The kernel's blocks stage
one chunk, `__syncthreads()`, run the outer product, `__syncthreads()`, stage the
next -- so loads are issued in bursts separated by a barrier, and at any instant
far fewer loads are in flight than the probe sustains. The outer product is only
~27% as long as the loads, so it cannot hide them.

**That points at the pipeline shape, which is exactly what round 74 flagged and
then set aside.** The next experiment is to overlap more.

### The pipeline split, and it is slower (round 83)

`stage_wtile` was split into `load_wtile` (issue the global loads into registers)
and `commit_wtile` (dequantise and store to shared), with the outer product moved
between them:

```
load(chunk 0); commit(wt[1]); sync
for c:
    load(chunk c+1)          <- issue, do not consume
    outer(wt[cur])           <- runs while the loads are in flight
    commit(wt[nxt])          <- consume
    sync
```

Correct (`generate` 16/16) and **slower**:

| | t=1 |
|---|---|
| baseline | 271.46 ms |
| load/commit split | 289.39 ms |

Reverted. The likely cause is register pressure: `pk[2]` and `scl[2]` are now live
across the whole outer product rather than only inside the staging, and this
kernel is already register-sensitive (round 67 saw 122 registers from a much
smaller change and lost 12%).

### Where that leaves the concurrency hypothesis

The hypothesis was that the staging's 71 GB/s (against 181 GB/s for the same
pattern in isolation) is caused by loads being issued in bursts separated by
barriers. The split pipeline should have fixed exactly that and it did not; it
made things worse. So either

* the compiler was already hoisting the loads across the outer product, in which
  case the bursts were never the problem and 71 GB/s has another cause, or
* the bursts are real but the register cost of fixing them exceeds the gain.

Those are distinguishable: compile both versions with `-Xptxas -v` and compare
register counts and spills. If the split version spills, the second explanation
holds and the fix is to reduce the live range -- for instance by keeping only the
scale in registers and re-loading the packed weights in `commit_wtile`, or by
splitting only the scale fetch, which is a byte rather than 8.

That check costs one command and should be the first thing done next round.

### The answer was already in the codebase (round 84)

`nvfp4_gemv_tmpl` in `kernels/gemv.cu` carries a comment describing exactly this
problem, written when that kernel was fixed:

> *Issue every row's weight and scale load BEFORE any of the arithmetic. Fusing
> the loads into the compute loop leaves the compiler free to keep only one load
> in flight per warp, which starves DRAM; separating them gives ROWS independent
> loads.*

and its weight addressing is

```
e0 = i * kTile + lane * kVec;          // kVec = 8 elements = 4 bytes... see below
pk[r] = *(const uint2*)(w + row*rowbytes + (e0 >> 1));
```

With `lane * kVec` and an 8-byte `uint2` load, **consecutive lanes read consecutive
8-byte chunks of a single row**, so one warp instruction covers 256 contiguous
bytes of **one row**. The GEMV then issues all `ROWS` loads before any arithmetic,
and dequantises 8 elements at a time with `e2m1x8_to_float`.

**The GEMM staging does neither of those things.** It scatters one instruction
across 8 rows (32 B each), and it fuses load with dequant and store.

So the difference between the 148 GB/s path and the 54 GB/s path is documented in
the repository, and the fix has two halves:

1. **One row per warp instruction.** Change the staging's thread-to-(row, k)
   mapping so consecutive lanes read consecutive bytes of the *same* row, as the
   GEMV does, instead of spreading across 8 rows. This is the change rounds 76,
   77 and 79 never made -- all three kept the 8-row scatter and varied only the
   bytes per row.
2. **Issue all loads before arithmetic**, which round 83 attempted and which made
   things worse *on its own* -- consistent with the GEMV comment, which pairs it
   with (1): separating loads only helps if there are `ROWS` genuinely
   independent loads to separate, and with an 8-row scatter there are not.

That also explains the register-pressure regression in round 83: `pk[2]`/`scl[2]`
were kept live across the outer product to buy nothing, because the loads were
still not independent in the way the GEMV's `ROWS` loads are.

**This is now the single highest-value change available**, and unlike the last
several attempts it has a working reference implementation in the same repo to
copy from.

### A caution learned in round 64

The probes were scripted with a `cp` restore from a scratch copy that predated
round 63, which silently reverted that change. Always restore probes with
`git checkout`, and check `git status` after.

### Two real bugs found on the way (round 58)

Worth keeping because both produced misleading failures:

1. **The `ops.rs` launch sites.** The GEMMs use a one-line
   `LaunchConfig { ... block_dim: (256,1,1) ... }`, which is *not* the
   `block_dim: (block_for(n, 256), 1, 1)` form used by the rmsnorm kernels. A
   blanket replace of the latter hit five rmsnorm sites and left the three real
   GEMM sites at 256 -- so the kernels declared `__launch_bounds__(128)` and were
   launched with 256, giving `CUDA_ERROR_INVALID_VALUE` on every call.
2. **`GB10_TILE_T` in `ops.rs` must match `GB10_TT` in `gemm.cu`** (the comment
   says so). Leaving it at 64 while the kernel tile became 32 halved the grid's
   `y` extent and silently produced wrong tokens rather than an error.

### The earlier, reverted attempt (round 57)

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

## MEASUREMENT RULE, added round 135: the ablation has a ~7-8% run-to-run spread

Identical binaries, three runs each:

| | run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|
| `gemv_bw` probe (level 1) | 261.9 | 279.4 | 273.7 | **~7%** |
| `abl6` level 3 (scale path) | 214.0 | 209.9 | 226.3 | **~8%** |

**Round 131 ran each level once.** Its level-3 value (194.8) is an **outlier below**
the 209.9-226.3 range. The *level-1 vs level-3 gap* does survive -- level 1 averages
~266 against level 3's ~215, about **19%** -- so the mechanism claim that the scale
path costs the most still stands. **But a single run overstated it, and two later
decisions were made on numbers that cannot support them:**

* **round 132's "-3.3% faster"** and
* **round 134's "+2.5% worse"**

are **both inside a ~7% noise band**. Neither the widening nor the shared hoist is
established by the `forward-cost` harness as it was run -- both were single
measurements. Round 134's conclusion (the hoist is not worth keeping) is *directionally*
supported by the shared-memory latency argument, but the 2.5% figure itself is noise.

**Rule: any number that becomes a mechanism claim -- or a keep/revert decision -- must
be run at least 3 times, and the spread reported alongside it.** Rounds 59, 101, 104
and 106 each established that a counted or correlated measurement can mean nothing;
this round establishes that a *single* measurement can too, including one this session
generated and then built three rounds of reasoning on.

**What is still solid:** the probe's pattern ceiling (~266-279 GB/s, mean ~272) against
the real kernel's 190 GB/s. That gap is **1.43x**, far outside the noise, and it is the
one number driving this investigation that survives repetition.

### The re-measurement vindicates round 132 (round 136)

Three runs per build, same harness, the discipline round 135's rule requires:

| build | runs | mean | spread |
|---|---|---|---|
| plain per-lane byte load (pre-132) | 105.18 / 105.87 / 105.85 | **105.63 ms** | 0.7% |
| **wide `uint32` + `__shfl_sync` (round 132)** | 103.29 / 103.03 / 104.11 | **103.48 ms** | 1.0% |

**-2.0%, and the two ranges do not overlap** (105.18-105.87 against 103.03-104.11).
**The widening is a real gain, and it is now established rather than assumed.**
Restored -- it is the better build on evidence.

**The harness itself is far better than the ablation:** `forward-cost` t=1 repeats to
**~1%**, against the probe/ablation's **~7-8%**. That is why round 132's single reading
of 100.79 was misleading -- it happened to land 2.7% below the same build's own mean
of 103.48 -- and it is why round 135's rule matters more for the ablation than for
this harness: **`forward-cost` can resolve a 2% effect in 3 runs; the ablation cannot.**

Round 134's shared-memory rejection is consistent with this: a +2.5% regression is
outside the 1% harness noise, so that rejection stands on its own measurement, which
the round-135 rule had left open.

### The widening does not generalise -- and the fp8 path is the real next target (round 137)

Round 136 proposed applying the same widening to the fp8 and bf16 templates. Checked
before implementing, and **there is nothing to widen**:

* `fp8_gemv_tmpl` takes `wscale` as a `const float*` and has **no per-lane scale byte
  load inside the k loop** -- fp8's scale is per-channel and applied outside it.
* `bf16_gemv_tmpl` has no scale at all.

**So the round-132 gain was specific to NVFP4's group-16 layout**, and that path is
now done. The proposal was empty rather than wrong, and one grep settled it.

**What that leaves is the largest untouched block in the decode step.** From round
124's decomposition of the 104.2 ms B=1 step:

| kernel | launches | avg | total/step | share of step |
|---|---|---|---|---|
| `nvfp4_gemv_kernel` | 192 | 274.3 us | 52.7 ms | 51% |
| **`fp8_gemv_kernel`** | **208** | **174.2 us** | **36.2 ms** | **35%** |
| `gated_delta_rule_{chunk,step}` + `rmsnorm` | ~9,600 | small | ~15 ms | 14% |

**`fp8_gemv_kernel` is 35% of the step and has never been decomposed.** The NVFP4 path
looked equally unremarkable until the ablation ran on it, and that ablation found its
scale path -- so the same treatment is the obvious next move: extend the `LEVEL`
ablation in `bench/hw/gemv_bw.cu` to fp8's layout (one `uint8` weight byte per lane per
k-tile plus a per-channel float scale) and see which step costs the bandwidth.

### The fp8 ablation is unnecessary -- the gap is entirely NVFP4 (round 138)

Checked with arithmetic before spending a round on it. Bytes per step are derivable
from the model's shapes, and both times were already measured in round 124:

| path | bytes/step | measured time | GB/s | its pattern's ceiling | gap |
|---|---|---|---|---|---|
| **NVFP4** (64 x 3 MLP + `lm_head`, /2 for 4-bit) | **9.19 GB** | 52.7 ms | **174** | 272 | **1.56x** |
| **fp8** (the remaining 17.608 - 9.19) | **8.42 GB** | 36.2 ms | **232** | -- | **none** |

**`fp8_gemv_kernel` is running at 232 GB/s -- at or above the measured 228 GB/s peak.
There is no gap there, and round 137's proposed fp8 ablation would have been wasted.**

**This also corrects the number to chase.** The NVFP4 path is at **174 GB/s**, not the
190 I derived earlier from nsys' average across mixed matrix sizes. Against its own
pattern's **272 GB/s** that is a **1.56x gap in one kernel on one code path** -- the
most precisely scoped target this session has had.

**Method note:** this cost one `python3 -c` and no GPU time. **Two of the last three
rounds' proposals died to arithmetic or a single grep before touching the kernel** --
the fp8 widening (nothing to widen) and this one (nothing to gain). Cheapest check
first is what round 135's measurement rule is really enforcing.

## Shared-memory hoist: tested and rejected -- and it re-reads the ablation (round 134)

Implemented exactly as designed below (nvfp4 template only, `use_smem` guard, static
4096 B buffer, cooperative load + `__syncwarp()`):

| | round 132 | shared hoist |
|---|---|---|
| `generate` | 16/16 | **16/16** |
| **t=1 one-row forward** | **100.79 ms** | **103.35 ms (+2.5% WORSE)** |

Reverted.

**A negative that clarifies everything:** removing **9 of the 10** global scale loads
made the kernel *slower*. The only way that is possible is that **those loads were
already being served from L2** -- the scale row is 320 bytes per row and a warp reuses
it 10 times, so it is tiny and hot. The `__syncwarp()` and the shared traffic were
pure addition.

**So the scale stream is not a DRAM-bandwidth cost, and round 131's step-3 drop of
54 GB/s is not traffic -- it is the load's presence in the dependency chain, i.e. L2
latency.** That single re-reading explains all three measurements at once:

* **widening the load (round 132) helped only 3.3%** -- fewer load instructions, but
  the same latency to L2;
* **ablation steps 4 and 5 *raised* bandwidth** (210.5, 212.5) -- adding work per
  k-tile gave the scheduler something to overlap the L2 latency with;
* **shared memory did not help** -- shared has its own latency, plus a `__syncwarp()`.

**The fix is therefore to break the dependency, not to move the data:**
software-pipeline the scale load one k-tile ahead, so tile `i`'s scale fetch overlaps
tile `i-1`'s FMAs. The ablation's own shape predicts this is the mechanism -- two
separate steps each added work and each *raised* bandwidth, which is exactly the
signature of latency hiding.

## Original design note (kept: implemented and rejected in round 134)

Round 132 widened the scale load (8 lanes x `uint32` + `__shfl_sync`) and kept it:
correct 16/16, t=1 **104.2 -> 100.79 ms (-3.3%)**. That is a real gain but only a third
of the ablation's ~20% prediction, and the ablation's own shape explains why: **steps 4
and 5 *raised* bandwidth (210.5, 212.5), so adding work per k-tile hid the extra load's
latency.** The cost is the second stream's existence, not its width -- so it has to be
*removed*, not widened.

**The design, with the constraints that make it a one-pass change:**

* The scale row is `scalerow = K/16` bytes -- **320 B for K=5120**, and K is a multiple
  of 128, so `scalerow` is a multiple of 8 and every 4-byte group is aligned.
* A warp reads the *same* row for all `full_tiles = K/512 = 10` k-tiles, so the row is
  currently fetched **10 times** when it need only be fetched once.
* Static shared, no launch-config change: `__shared__ uint8_t ssc[4096];` indexed
  `ssc + (warp * ROWS + r) * scalerow`. Needs `nwarps * ROWS * scalerow <= 4096`:
  at the current launch (128 threads = 4 warps, ROWS=1, scalerow=320) that is
  **1280 B**, well inside. **Guard it:** `if (scalerow <= 512)` uses shared, else fall
  back to the round-132 path -- do not silently overflow.
* Cooperative load at the top of the row loop: lanes `0..(scalerow/4-1)` each load one
  `uint32` (that is 80 lanes' worth at K=5120, so loop it), then `__syncwarp()`.
  Keep the loads 4-byte aligned and the loop unrolled.
* The k loop then reads `ssc[... + i*kWarp + lane]` -- a shared load, no global traffic.

**Expected signals:** `generate` **must be 16/16**; t=1 must fall below **100.79 ms**.
If the `__syncwarp()` and shared traffic eat the gain, revert and record -- the
ablation is still the only thing that has located this, and a negative here would mean
the 26% is not addressable from the scale path at all.

---

**STATUS OF THE ABOVE NOTE: implemented and REJECTED in round 134** (+2.5% worse, and
that regression is outside the ~1% `forward-cost` noise). It is kept only as the record
of the design and its constraints. **Do not implement it again.** The 26% it was meant
to recover did not come from moving the data -- see the L2-latency re-reading and the
round-135 measurement rule earlier in this file.

**The current, verified target is narrower:** `nvfp4_gemv_kernel` at **174 GB/s** against
its own pattern's **272 GB/s** (1.56x). fp8 is already at 232 GB/s with no gap, so the
entire decode shortfall is that one kernel. Everything ruled out on the way there is
listed above, each with its measurement.
