# Round 355 — 20260928T072208Z

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
[round 355] batch-parity OK
[round 355] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 355] prefix cache OK
[round 355] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.463       0.969   13.89x     1.96e-6   (120 GB/s serial, 1662 GB/s warp)

decode-bench: OK
[round 355] decode-bench OK
[round 355] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.9s
interleave: false
  iter  0: 112.77 ms  155.7 GB/s
  iter  5: 116.39 ms  150.8 GB/s
  iter 10: 116.15 ms  151.1 GB/s
  iter 15: 120.40 ms  145.8 GB/s
  iter 20: 116.74 ms  150.4 GB/s
  iter 25: 116.79 ms  150.3 GB/s
  iter 29: 115.33 ms  152.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.29 ms
achieved bandwidth       : 151.0 GB/s
projected decode         : 8.60 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T072208Z.json
[round 355] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T072208Z.json
```
