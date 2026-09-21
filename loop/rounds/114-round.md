# Round 114 — 20260921T103144Z

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
[round 114] batch-parity OK
[round 114] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.9s
interleave: false
  iter  0: 88.55 ms  198.3 GB/s
  iter  5: 89.02 ms  197.2 GB/s
  iter 10: 88.02 ms  199.4 GB/s
  iter 15: 89.73 ms  195.7 GB/s
  iter 20: 88.15 ms  199.2 GB/s
  iter 25: 88.24 ms  198.9 GB/s
  iter 29: 88.77 ms  197.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 88.79 ms
achieved bandwidth       : 197.7 GB/s
projected decode         : 11.26 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 86.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T103144Z.json
[round 114] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T103144Z.json
```
