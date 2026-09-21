# Round 97 — 20260921T084016Z

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
[round 97] batch-parity OK
[round 97] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 43.5s
interleave: false
  iter  0: 99.00 ms  177.3 GB/s
  iter  5: 99.27 ms  176.8 GB/s
  iter 10: 95.18 ms  184.4 GB/s
  iter 15: 97.78 ms  179.5 GB/s
  iter 20: 97.36 ms  180.3 GB/s
  iter 25: 99.08 ms  177.2 GB/s
  iter 29: 99.85 ms  175.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.33 ms
achieved bandwidth       : 180.4 GB/s
projected decode         : 10.27 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T084016Z.json
[round 97] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T084016Z.json
```
