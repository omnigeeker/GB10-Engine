# The full 262,144-token context

**The engine serves the checkpoint's native context.** The model's
`max_position_embeddings` is 262144; the server was hard-capped at 2048. This is
how that was lifted, what it cost, and the evidence that it works — a
needle-in-a-haystack test at 32 K, 128 K and 256 K, all of which pass.

## What was actually in the way

Four separate limits, found in this order. Only the first is obvious.

1. **The prefill attention kernel put one score per key in dynamic shared
   memory** — it requested `(start + n_tokens) * 4` bytes — so the reachable
   context was capped at the 48 KB a launch can request without opting into a
   larger carve-out, i.e. 12,288 keys, and the launch *failed* past that.

2. **The decode attention kernels did exactly the same thing**, for every
   sequence, on every step. This is the one that first showed up as
   `CUDA_ERROR_INVALID_VALUE` on the very first request at `--ctx 32768`, and
   it is why raising `MAX_SEQ` alone could never have worked.

3. **Scratch memory was sized to the context.** `Scratch::new(dev, cfg, max_seq)`
   allocates a per-token row for every buffer, ~586 KB per token, which at 256K
   is ~150 GB. Chunked prefill is what makes the window affordable at all.

4. **The dispatch cutover had an off-by-128-bytes bug.** `attn_prefill_kernel`
   claims 128 B of *static* shared memory for its block-reduction helper, and the
   per-block limit covers static and dynamic together, so a chunk landing exactly
   at 12,288 keys requested 49,152 B dynamic + 128 B static and failed to launch.
   Invisible on short prompts; it fires only once a chunk reaches the boundary.

## Fixes

* **`attn_prefill_tiled_kernel`** — new tiled causal attention with GQA and the
  online-softmax recurrence. Shared memory is a constant (41.5 KB) instead of
  `O(keys)`, so the key count no longer bounds anything but time. One block per
  (query head, 8 query rows), one thread per head dimension, output accumulators
  in registers.
* **Both decode kernels** rewritten to the same streaming recurrence, dropping
  their dynamic shared-memory requests to zero.
* **`Scratch` decoupled from the window**: sized to a 2048-token prefill chunk,
  with the prompt fed in successive chunks (`prefill_chunked`, and the same loop
  in `run_group`).
* **`--ctx` / `--concurrency`**, with a 40 GB KV budget that picks the slot count
  when `--concurrency` is omitted and refuses an explicit count that does not
  fit. `--ctx` above 262144 is refused rather than served.
* **Window check in the batcher**: `run_group` had *no* prompt+generation bound
  at all, so a long prompt with a large `max_tokens` would have appended past its
  KV slot into the next sequence's cache. That is a pre-existing bug, not one
  introduced here.
* **Every prefill now uses the tiled kernel.** Keeping the legacy kernel under
  the cutover for bit-identical results cost far more than the rounding it
  avoided: measured on the same 2048-token chunk, 82.3 s at 10,240 keys (legacy)
  against 39.2 s at 12,288 (tiled). The two agree to ~1e-7 relative.

## Verification

* `attn-tile` (new gate): tiled against the legacy kernel on random inputs
  across 13 shapes that straddle the tile edges (`nt` = 1/7/8/9/16/17/…, `start`
  = 37/100/511/1000/2047), agreeing to **~1e-7 RMS relative** — f32 summation
  order. Plus the production path against legacy at five sizes, and long spans
  (4096/16384/65536 keys) checked for non-finite or collapsed output.
* `generate` (existing gate): **16/16 exact** against the bf16 oracle after
  moving all prefill to the tiled kernel — the change is numerically safe.
* Round 250 gate: build, test, correctness, generate, batch-parity, bench — PASS.

### A measurement trap worth recording

The server's per-chunk times grow ~5 s per chunk, which no single op in the layer
forward accounts for, and the attention kernel does not reproduce it in
isolation (1.76 s for the same shape). That looked like a 14× discrepancy for a
while. It was not: the isolated measurement is **one kernel call, i.e. one
layer**, while the server's bucket covers all **16** attention layers. 16 × 1.76
≈ 28 s, which is exactly the chunk time. The in-situ per-layer numbers (0.157 s
at `start=0`, 0.467 s at `start=2048`) match the isolated kernel to within
noise. Forcing the key range to zero (`GB10_ATTN_START_ZERO`) flattened every
chunk to 15.2 s, confirming the growing key range is the whole story.

## Validating it: needle-in-a-haystack

`bench/longctx/needle.py` buries a code at a fixed depth in repeated filler and
asks for it, over `max_tokens=24`. Depths are fractions of the document, so a
pass cannot be carried by recency alone.

| prompt | leg | result |
|---|---|---|
| **34,779 tok (32 K class)** | depths 10% / 50% / 90% | **3/3 PASS** — 306.2 / 308.8 / 308.9 s |
| **34,819 tok, llama.cpp control** | depths 10% / 50% / 90% | **3/3 PASS** — 71.2 / 71.0 / 67.1 s |
| 32 K (32,733 tok, 16 chunks) | depths 10% / 50% / 90% | PASS 3/3 — 815.8 / 816.6 / 819.3 s (historical) |
| 128 K (130,693 tok, 64 chunks) | depth 50% | PASS — 10,784.7 s (historical) |
| 256 K (261,358 tok, 128 chunks) | depth 50% | PASS — 41,821.6 s (historical) |

The top two rows are one session, one harness, both engines, same prompts. That
is what certifies long-context retrieval today. The full transcript of the
original 32K/128K/256K run is committed as
`bench/longctx/results-32k-128k-256k.log`.

> **History: this section previously said the 32K leg did not reproduce "on
> gb10-engine *or* llama.cpp", and concluded that because llama.cpp failed the
> same prompts, the engine was not implicated. Both halves of that were wrong,
> and the way they were wrong is worth keeping.**
>
> **The engine was implicated.** The failure was a real regression in the
> committed default: the tensor-core prefill GEMM (default-on since round 282,
> commit `336d91d`) computed its operands in bf16, and 8 mantissa bits is not
> enough for this model. Past roughly 970 prompt tokens the reduced precision
> flipped one greedy argmax at the first generated position to EOS, so the engine
> answered nothing at all — deterministic `completion_tokens: 0`, erratic in
> *length* rather than monotone, identical under every `temperature` /
> `--no-prefix-cache` / `PREFILL_CHUNK` / `enable_thinking` setting. Every one of
> those observations was read as evidence of *not a regression*, and every one of
> them was equally consistent with "one flipped argmax at long context". It was
> found by `git bisect run`, and the cause turned out not to be numerical at all: the
> tensor-core path was **truncating work**, skipping every element past 16,776,960
> because eight element-wise kernels had no grid-stride loop under a 65535-block cap.
> Fixed, and the tensor-core GEMM is on by default again. `loop/run_round.sh` now runs
> both a long-prompt gate (`longctx-follow`) and a direct arithmetic gate
> (`tc-parity`) so it cannot come back silently. Full record in
> `bench/longctx/TC_GEMM_REGRESSION.md`.
>
> **llama.cpp never failed.** It answers these prompts in `reasoning_content`
> rather than `content`, and `needle.py` read only `content`, so every llama.cpp
> leg scored as a miss. It also needs a token budget larger than 24: it thinks
> before answering, and 24 tokens truncates it mid-thought (`'The user provided a
> very'`). With `reasoning_content` read and a 1024-token budget it passes 3/3,
> as the control row above shows. A harness that reads the wrong field will
> convict an innocent engine — and a control that fails is a reason to doubt the
> harness, not a reason to exonerate the subject.

The timings above were taken with the fp32 prefill GEMM, i.e. with the fast path
switched off, and they are superseded. The tensor-core GEMM is now both fast **and**
correct: its failure was never numerical precision but a truncated grid in eight
element-wise kernels, which silently skipped every element past 16,776,960 — and that
is also the *reason* those timings were slow, not a price that had to be paid. See
`bench/longctx/TC_GEMM_REGRESSION.md` for the root cause and `gb10-bench tc-parity`
for the gate that now pins it down.

The current same-session cold-TTFT picture (`bench/longctx/results-ttft-fixed.log`,
`--ctx 262144` on both engines) is

| context | gb10 | llama.cpp | ratio |
|---|---|---|---|
| 8K | 18.90 s | 13.45 s | 1.41x slower |
| 32K | 120.15 s | 56.94 s | 2.11x slower |
| 128K | 1224.16 s | 290.63 s | 4.21x slower |

and decomposes into a linear term within 7.5% of llama.cpp and a quadratic
(attention) term 11.18x slower. **The remaining work is prefill attention, and the
target is 11.2x** — the same at every context length, because the condition reduces to
`gb10_quad / k < llama_quad`. The `mma.sync` plan in `bench/longctx/comparison.md` is
the right lever; note it must be specified to ~12x rather than 9x, since 9x leaves 32K
and 128K still losing.


Two operational traps cost real time here, both invisible from the code.

**The harness kills by subprocess *scope*, so `setsid` does not detach.** The
first run used `setsid nohup` precisely to outlive the turn, and was killed
96 min in regardless:

```
dsh-subprocess-191061-e0db7c338b98.scope: Sending signal SIGTERM to
process 193058 (gb10-server) on client request.
```

That window covers the 32K leg (~40 min) and nothing longer — exactly the
pattern observed, with 32K completing and 128K never finishing. It now runs as a
`systemd-run --user` unit, which lives in the user manager's scope.

**A non-streaming HTTP request is bounded by the client's read timeout, and the
server says nothing until the prefill finishes.** `needle.py` had
`timeout=20000` s (5.6 h): enough for the ~3 h 128K leg, not for the ~12 h 256K
one, which would have been abandoned by the client after all eleven hours for no
visible reason. Raised to 48 h. The server has no HTTP timeout of its own, so
nothing else bounds the request.

## Measured cost

Prefill ≈ `5.11 ms·T + 5.93e-7·T²` seconds on this box — a least-squares fit to
the three needle runs measured end to end, which it reproduces to within 1.8%:

| prompt | prefill | fitted | |
|---|---|---|---|
| 5 K | ~50 s | 26 s | |
| 12 K | ~170 s | 146 s | |
| 32 K | 816.6 s | 802.2 s | −1.8% |
| 128 K | 10,784.7 s | 10,791.0 s | +0.1% |
| 256 K | 41,821.6 s | 41,820.2 s | −0.0% |

The quadratic term is the 16 full-attention layers reading a growing key range —
inherent to dense attention, not an artefact of chunking (chunking bounds
memory, not time). At 256K it is 97% of the total: 40,500 s of the 41,822 s. The
48 Gated-DeltaNet layers stay linear, which is the only reason a 256K prompt is
reachable at all.

## Known issues left in place

Two things were found while working on this and deliberately *not* changed, so
that the validated binary is the committed one. Both are follow-ups.

* **`Engine::generate` and `Engine::prefill_chunked` are dead code.**
  `gb10-server` has two prefill paths; the live one is `run_group`, and
  `generate` has no call sites at all. This is a trap rather than a tidiness
  problem: instrumentation added to `generate` did nothing, and a `max_tokens`
  clamp put there had no effect, because the request never reaches it. The
  duplication should be collapsed before it misleads anyone else.
* **`ModelState` still holds context-sized residual buffers.** `a`, `b` and
  `normed` are each `hidden * ctx * 4` bytes — 16 GB together at 256K — though
  a prefill chunk only ever touches `PREFILL_CHUNK` rows. The KV cache is the
  structure that legitimately scales with the window; these do not, and sizing
  them to the chunk would free ~16 GB at 256K.

## Gates

Verified by `attn-tile` (new), the existing `correctness`/`generate`/
`batch-parity` gates, and the round gate itself; see `loop/rounds/` for the
per-round record. Every gate passed, and `generate` is 16/16 token-exact against
the bf16 oracle after the kernel swap.
