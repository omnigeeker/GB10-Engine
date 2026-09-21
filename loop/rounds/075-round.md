# Round 75 — 20260921T060505Z

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
[round 75] batch-parity OK
[round 75] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.7s
interleave: false
  iter  0: 96.92 ms  181.1 GB/s
  iter  5: 96.38 ms  182.1 GB/s
  iter 10: 94.98 ms  184.8 GB/s
  iter 15: 97.66 ms  179.8 GB/s
  iter 20: 98.42 ms  178.4 GB/s
  iter 25: 94.95 ms  184.9 GB/s
  iter 29: 98.70 ms  177.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.49 ms
achieved bandwidth       : 180.1 GB/s
projected decode         : 10.26 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T060505Z.json
[round 75] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T060505Z.json
```
