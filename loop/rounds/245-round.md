# Round 245 — 20260924T043512Z

**Milestone: accuracy evaluation (perplexity).** Not a performance round. Adds a
`perplexity` path to `gb10-verify`, measures it three ways on wikitext-2, and
records the result in `docs/ACCURACY.md`. Also closes a path that could turn an
asynchronous CUDA error into a silently wrong token.

## Headline

wikitext-2 test, `n_ctx=512`, **identical token ids for all three runs**
(297,054 tokens / 147,900 predictions each):

| implementation | weights | PPL | vs BF16 |
|---|---|---|---|
| llama.cpp | NVFP4 (GGUF of this checkpoint) | 7.2088 ± 0.0471 | +2.244 % |
| **gb10-engine** | NVFP4 (the checkpoint) | **7.0988** ± 0.0164 | **+0.684 %** |
| `transformers` | BF16 (base model) | 7.0506 ± 0.0165 | — |

**No accuracy drop.** The engine is 1.53 % below llama.cpp and, against the
unquantized BF16 model, carries 3.3x less quantization cost than llama.cpp does
on the same weights.

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness (layers) | pass |
| correctness (64-layer) | pass |
| benchmark | pass |

**status: PASS**

## Changes

| file | change |
|---|---|
| `crates/gb10-verify/src/main.rs` | `perplexity` subcommand, `--tokens/--text/--ctx/--chunks/--out`; tokenizer cross-check |
| `crates/gb10-model/src/model.rs` | `forward_normed` split out of `prefill_seq`; `check_err` before any token is trusted |
| `crates/gb10-cuda/src/ops.rs` | `copy_rows` launch wrapper |
| `crates/gb10-cuda/src/lib.rs` | `Device::check_err` |
| `kernels/elementwise.cu` | `copy_rows_kernel` |
| `docs/ACCURACY.md` | the evaluation |
| `bench/ppl/*` | scripts, artifacts, shared token stream |

## Why the result is trustworthy

* **Token stream is not a variable.** The engine's Rust tokenizer and
  `llama-tokenize` agree on **0 mismatches / 297,054 tokens**; the run aborts
  otherwise.
* **Weights are not a variable.** A GGUF inventory shows 193 NVFP4 tensors =
  3 x 64 MLP + `lm_head`, exactly the checkpoint's NVFP4 set. The FP8 attention
  projections became BF16, which is exact.
* **Determinism.** A 21-window re-run reproduces the full run's `[20] ppl=6.8307`
  exactly.
* **Independent confirmation.** The 64-layer gate is still 16/16 token-exact
  against the HF NVFP4-dequantized oracle.

## Closed: a silent-wrong-answer path

`CudaSlice::drop` (cudarc 0.19.9, `driver/safe/core.rs:816`) synchronises the
stream and stores the result via `record_err`, which keeps it in an atomic
instead of raising it. Nothing in the engine read that atomic. An asynchronous
CUDA failure would therefore leave the engine running on whatever the failed
kernel left in its output buffer -- on the argmax, a wrong token that still
decodes to fluent text. `Device::check_err()` now surfaces it, and is called
after `argmax` in `step`, `step_batch`, `step_timed` and `prefill_seq`, and at
the end of `forward_normed`.

## Open: one load-induced divergent token

One `generate` run, made while llama.cpp's perplexity job held the GPU, emitted
`[760, ...]` instead of the oracle's `[1421, ...]`, with no error printed. It did
not reproduce in **7 further runs**, 3 under the BF16 job's load and 4 under a
concurrent engine perplexity run. `check_err` changes its failure mode from
silent to loud; the root cause is still unknown and is the next thing to chase.

## Log

```
  seq  2 (prompt   13 tok): match
  seq  3 (prompt   17 tok): match
  seq  4 (prompt   21 tok): match
  seq  5 (prompt   25 tok): match
  seq  6 (prompt   29 tok): match
  seq  7 (prompt   33 tok): match
  seq  8 (prompt   37 tok): match
  seq  9 (prompt   41 tok): match
  seq 10 (prompt   45 tok): match
  seq 11 (prompt   49 tok): match
  seq 12 (prompt   53 tok): match
  seq 13 (prompt   57 tok): match
  seq 14 (prompt   61 tok): match
  seq 15 (prompt   65 tok): match
batch parity: 16/16 sequences exact over 16 tokens

batch-parity: OK
[round 245] batch-parity OK
[round 245] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 38.8s
interleave: false
  iter  0: 91.23 ms  192.4 GB/s
  iter  5: 100.76 ms  174.2 GB/s
  iter 10: 100.13 ms  175.3 GB/s
  iter 15: 96.42 ms  182.1 GB/s
  iter 20: 93.55 ms  187.7 GB/s
  iter 25: 92.65 ms  189.5 GB/s
  iter 29: 98.75 ms  177.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.64 ms
achieved bandwidth       : 181.7 GB/s
projected decode         : 10.35 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260924T043512Z.json
[round 245] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260924T043512Z.json
```
