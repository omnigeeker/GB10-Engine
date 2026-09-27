# Round 301 — 20260927T200912Z

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
[round 301] batch-parity OK
[round 301] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 301] prefix cache OK
[round 301] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.160       1.659    9.14x     2.00e-6   (106 GB/s serial, 971 GB/s warp)

decode-bench: OK
[round 301] decode-bench OK
[round 301] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.2s
interleave: false
  iter  0: 95.01 ms  184.8 GB/s
  iter  5: 93.49 ms  187.8 GB/s
  iter 10: 95.89 ms  183.1 GB/s
  iter 15: 97.09 ms  180.8 GB/s
  iter 20: 96.09 ms  182.7 GB/s
  iter 25: 94.88 ms  185.0 GB/s
  iter 29: 97.94 ms  179.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.21 ms
achieved bandwidth       : 182.5 GB/s
projected decode         : 10.39 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T200912Z.json
[round 301] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T200912Z.json
```
