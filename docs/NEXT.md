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

1. **Store fp8 weights as uint16 (bf16) instead of float.** e4m3 has 3 mantissa
   bits, bf16 has 7, so this is lossless. It halves the fp8 `wt` buffer and is
   what makes KC=64 fit at all.
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
