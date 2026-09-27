# Round 251 — 20260927T054020Z

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
[round 251] batch-parity OK
[round 251] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 38.4s
interleave: false
  iter  0: 92.97 ms  188.8 GB/s
  iter  5: 92.94 ms  188.9 GB/s
  iter 10: 95.19 ms  184.4 GB/s
  iter 15: 95.17 ms  184.5 GB/s
  iter 20: 93.06 ms  188.7 GB/s
  iter 25: 97.94 ms  179.3 GB/s
  iter 29: 95.77 ms  183.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 94.19 ms
achieved bandwidth       : 186.4 GB/s
projected decode         : 10.62 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 81.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T054020Z.json
[round 251] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T054020Z.json
```
