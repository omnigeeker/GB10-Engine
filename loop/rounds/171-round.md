# Round 171 — 20260921T160529Z

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
[round 171] batch-parity OK
[round 171] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 44.9s
interleave: false
  iter  0: 88.08 ms  199.3 GB/s
  iter  5: 88.07 ms  199.3 GB/s
  iter 10: 86.53 ms  202.9 GB/s
  iter 15: 85.76 ms  204.7 GB/s
  iter 20: 86.50 ms  202.9 GB/s
  iter 25: 87.15 ms  201.4 GB/s
  iter 29: 85.48 ms  205.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 86.58 ms
achieved bandwidth       : 202.8 GB/s
projected decode         : 11.55 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 88.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T160529Z.json
[round 171] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T160529Z.json
```
