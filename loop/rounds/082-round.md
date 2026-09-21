# Round 82 — 20260921T064433Z

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
[round 82] batch-parity OK
[round 82] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.2s
interleave: false
  iter  0: 95.60 ms  183.6 GB/s
  iter  5: 98.08 ms  179.0 GB/s
  iter 10: 98.27 ms  178.6 GB/s
  iter 15: 93.28 ms  188.2 GB/s
  iter 20: 102.53 ms  171.2 GB/s
  iter 25: 99.56 ms  176.3 GB/s
  iter 29: 98.31 ms  178.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.57 ms
achieved bandwidth       : 179.9 GB/s
projected decode         : 10.25 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T064433Z.json
[round 82] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T064433Z.json
```
