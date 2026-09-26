# Round 247 — 20260926T051852Z

**Milestone: the determinism question answered with power, and gated.** No model
behaviour changed. `gb10-verify generate` gained `--repeat N`, which runs the
same prefill+decode N times from fresh state and fails if any run differs; the
round gate now uses `--repeat 8`. Full write-up in `docs/ACCURACY.md`.

## What was measured

The round-1 incident — one `generate` run emitting `[760, ...]` instead of
`[1421, ...]` under llama.cpp's load — was retried under the *exact* original
condition and did not recur:

| test | result |
|---|---|
| 8 prompts x 64 tokens, llama.cpp holding the GPU | **byte-identical to the clean run** — same divergence indices, same token ids; only TTFT moved (455 -> 963 ms) |
| `generate --n 16 --repeat 100`, same load | **99/99 extra repeats identical to run 0** (450 s of GPU work) |
| round gate `--repeat 8`, clean | 7/7 identical (31.9 s) |
| 580-window perplexity | bit-reproducible; a 21-window re-run reproduces `[20] ppl=6.8307` |

Roughly 700 forward passes, under load and idle, with no observed
nondeterminism. A non-deterministic path now fails the gate instead of being
discovered later.

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness (layers) | pass |
| correctness (64-layer) | pass |
| benchmark | pass |

**status: PASS**

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
[round 247] batch-parity OK
[round 247] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 41.3s
interleave: false
  iter  0: 89.98 ms  195.1 GB/s
  iter  5: 90.20 ms  194.6 GB/s
  iter 10: 89.82 ms  195.4 GB/s
  iter 15: 88.29 ms  198.8 GB/s
  iter 20: 86.92 ms  202.0 GB/s
  iter 25: 87.76 ms  200.0 GB/s
  iter 29: 89.47 ms  196.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 89.77 ms
achieved bandwidth       : 195.6 GB/s
projected decode         : 11.14 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 85.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260926T051852Z.json
[round 247] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260926T051852Z.json
```
