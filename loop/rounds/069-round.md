# Round 69 — 20260921T052830Z

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
[round 69] batch-parity OK
[round 69] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.0s
interleave: false
  iter  0: 99.09 ms  177.2 GB/s
  iter  5: 96.50 ms  181.9 GB/s
  iter 10: 96.94 ms  181.1 GB/s
  iter 15: 97.17 ms  180.7 GB/s
  iter 20: 96.19 ms  182.5 GB/s
  iter 25: 97.30 ms  180.4 GB/s
  iter 29: 99.66 ms  176.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.22 ms
achieved bandwidth       : 180.6 GB/s
projected decode         : 10.29 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T052830Z.json
[round 69] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260921T052830Z.json
```
