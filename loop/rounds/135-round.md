# Round 135 — 20260921T122136Z

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
[round 135] batch-parity OK
[round 135] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.5s
interleave: false
  iter  0: 86.85 ms  202.1 GB/s
  iter  5: 87.62 ms  200.4 GB/s
  iter 10: 86.61 ms  202.7 GB/s
  iter 15: 86.55 ms  202.8 GB/s
  iter 20: 87.91 ms  199.7 GB/s
  iter 25: 86.26 ms  203.5 GB/s
  iter 29: 85.96 ms  204.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 87.32 ms
achieved bandwidth       : 201.0 GB/s
projected decode         : 11.45 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 88.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T122136Z.json
[round 135] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T122136Z.json
```
