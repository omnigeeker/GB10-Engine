# Round 263 — 20260927T134615Z

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness (layers) | pass |
| correctness (64-layer) | pass |
| decode-bench (warp vs serial) | pass |
| prefix cache A/B | pass |
| benchmark | pass |

**status: PASS**

## Log

```
batch-parity: OK
[round 263] batch-parity OK
[round 263] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 263] prefix cache OK
[round 263] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      14.872       1.505    9.88x     2.00e-6   (108 GB/s serial, 1070 GB/s warp)

decode-bench: OK
[round 263] decode-bench OK
[round 263] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 95.26 ms  184.3 GB/s
  iter  5: 94.76 ms  185.3 GB/s
  iter 10: 99.75 ms  176.0 GB/s
  iter 15: 95.95 ms  183.0 GB/s
  iter 20: 95.02 ms  184.7 GB/s
  iter 25: 97.22 ms  180.6 GB/s
  iter 29: 97.13 ms  180.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.39 ms
achieved bandwidth       : 182.1 GB/s
projected decode         : 10.37 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T134615Z.json
[round 263] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T134615Z.json
```
