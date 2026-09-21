# Round 213 — 20260921T202925Z

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
[round 213] batch-parity OK
[round 213] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 43.2s
interleave: false
  iter  0: 87.69 ms  200.2 GB/s
  iter  5: 85.62 ms  205.0 GB/s
  iter 10: 85.14 ms  206.2 GB/s
  iter 15: 87.63 ms  200.3 GB/s
  iter 20: 87.53 ms  200.6 GB/s
  iter 25: 87.48 ms  200.7 GB/s
  iter 29: 85.62 ms  205.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 86.78 ms
achieved bandwidth       : 202.3 GB/s
projected decode         : 11.52 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 88.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T202925Z.json
[round 213] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T202925Z.json
```
