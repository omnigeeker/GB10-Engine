# Round 177 — 20260921T162637Z

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
[round 177] batch-parity OK
[round 177] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 44.5s
interleave: false
  iter  0: 87.59 ms  200.4 GB/s
  iter  5: 87.33 ms  201.0 GB/s
  iter 10: 86.36 ms  203.3 GB/s
  iter 15: 89.22 ms  196.8 GB/s
  iter 20: 90.86 ms  193.2 GB/s
  iter 25: 84.75 ms  207.2 GB/s
  iter 29: 85.52 ms  205.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 86.87 ms
achieved bandwidth       : 202.1 GB/s
projected decode         : 11.51 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 88.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T162637Z.json
[round 177] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T162637Z.json
```
