# Round 251 — native 262,144-token context

**Milestone: the engine serves the checkpoint's full native context.** The model's
`max_position_embeddings` is 262144; the server was hard-capped at 2048.

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

## Measured cost

Prefill ≈ `8.8 ms·T + 6.0e-7·T²` seconds on this box. The quadratic term is the
16 full-attention layers reading a growing key range — inherent to dense
attention, not an artefact of chunking (chunking bounds memory, not time). The
48 Gated-DeltaNet layers stay linear.

| prompt | prefill |
|---|---|
| 5 K | ~50 s |
| 12 K | ~170 s |
| 32 K | ~13.5 min |
| 128 K | ~3 h |
| 256 K | ~12 h |

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness (layers) | pass |
| correctness (64-layer) | pass |
| generate | pass (16/16 exact) |
| batch-parity | pass |
| attn-tile | pass |
| benchmark | pass |

**status: PASS**
