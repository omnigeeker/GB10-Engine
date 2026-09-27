# Round 252 — 20260927T074316Z

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness (layers) | pass |
| correctness (64-layer) | pass |
| decode-bench (warp vs serial) | pass |
| benchmark | pass |

**status: PASS**

## Log

```
  seq  9 (prompt   41 tok): match
  seq 10 (prompt   45 tok): match
  seq 11 (prompt   49 tok): match
  seq 12 (prompt   53 tok): match
  seq 13 (prompt   57 tok): match
  seq 14 (prompt   61 tok): match
  seq 15 (prompt   65 tok): match
batch parity: 16/16 sequences exact over 16 tokens

batch-parity: OK
[round 252] batch-parity OK
[round 252] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.550       1.574    6.07x     2.07e-6   (169 GB/s serial, 1023 GB/s warp)

decode-bench: OK
[round 252] decode-bench OK
[round 252] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 38.1s
interleave: false
  iter  0: 99.81 ms  175.9 GB/s
  iter  5: 98.11 ms  178.9 GB/s
  iter 10: 99.61 ms  176.2 GB/s
  iter 15: 96.04 ms  182.8 GB/s
  iter 20: 97.00 ms  181.0 GB/s
  iter 25: 96.75 ms  181.5 GB/s
  iter 29: 94.96 ms  184.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.21 ms
achieved bandwidth       : 180.6 GB/s
projected decode         : 10.29 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T074316Z.json
[round 252] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T074316Z.json
```
