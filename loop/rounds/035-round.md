# Round 35 — 20260920T083741Z

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
[round 35] batch-parity OK
[round 35] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.1s
interleave: false
  iter  0: 98.10 ms  178.9 GB/s
  iter  5: 97.89 ms  179.3 GB/s
  iter 10: 100.27 ms  175.1 GB/s
  iter 15: 94.94 ms  184.9 GB/s
  iter 20: 95.18 ms  184.5 GB/s
  iter 25: 97.73 ms  179.6 GB/s
  iter 29: 96.04 ms  182.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.95 ms
achieved bandwidth       : 181.1 GB/s
projected decode         : 10.31 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T083741Z.json
[round 35] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T083741Z.json
```
