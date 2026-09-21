# Round 174 — 20260921T161609Z

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
[round 174] batch-parity OK
[round 174] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 44.0s
interleave: false
  iter  0: 86.27 ms  203.5 GB/s
  iter  5: 86.22 ms  203.6 GB/s
  iter 10: 85.75 ms  204.7 GB/s
  iter 15: 86.05 ms  204.0 GB/s
  iter 20: 85.45 ms  205.5 GB/s
  iter 25: 84.58 ms  207.6 GB/s
  iter 29: 85.26 ms  205.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 85.94 ms
achieved bandwidth       : 204.3 GB/s
projected decode         : 11.64 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 89.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T161609Z.json
[round 174] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T161609Z.json
```
