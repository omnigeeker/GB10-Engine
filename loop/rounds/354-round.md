# Round 354 — 20260928T071329Z

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
[round 354] batch-parity OK
[round 354] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 354] prefix cache OK
[round 354] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.380       0.995   13.45x     1.96e-6   (120 GB/s serial, 1619 GB/s warp)

decode-bench: OK
[round 354] decode-bench OK
[round 354] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.0s
interleave: false
  iter  0: 113.96 ms  154.0 GB/s
  iter  5: 115.57 ms  151.9 GB/s
  iter 10: 115.41 ms  152.1 GB/s
  iter 15: 116.80 ms  150.3 GB/s
  iter 20: 115.89 ms  151.5 GB/s
  iter 25: 120.63 ms  145.5 GB/s
  iter 29: 120.50 ms  145.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.63 ms
achieved bandwidth       : 149.2 GB/s
projected decode         : 8.50 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T071329Z.json
[round 354] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T071329Z.json
```
