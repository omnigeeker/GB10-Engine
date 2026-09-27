# Round 270 — 20260927T154833Z

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
[round 270] batch-parity OK
[round 270] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 270] prefix cache OK
[round 270] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.764       1.649    9.56x     2.00e-6   (102 GB/s serial, 977 GB/s warp)

decode-bench: OK
[round 270] decode-bench OK
[round 270] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 96.96 ms  181.0 GB/s
  iter  5: 95.57 ms  183.7 GB/s
  iter 10: 97.91 ms  179.3 GB/s
  iter 15: 95.67 ms  183.5 GB/s
  iter 20: 97.13 ms  180.7 GB/s
  iter 25: 96.06 ms  182.8 GB/s
  iter 29: 98.49 ms  178.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.00 ms
achieved bandwidth       : 181.0 GB/s
projected decode         : 10.31 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T154833Z.json
[round 270] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T154833Z.json
```
