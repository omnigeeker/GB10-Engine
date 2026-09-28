# Round 371 — 20260928T111244Z

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
[round 371] batch-parity OK
[round 371] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 371] prefix cache OK
[round 371] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.473       1.089   12.37x     1.96e-6   (120 GB/s serial, 1479 GB/s warp)

decode-bench: OK
[round 371] decode-bench OK
[round 371] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.2s
interleave: false
  iter  0: 116.74 ms  150.4 GB/s
  iter  5: 116.11 ms  151.2 GB/s
  iter 10: 113.14 ms  155.2 GB/s
  iter 15: 115.29 ms  152.3 GB/s
  iter 20: 120.35 ms  145.9 GB/s
  iter 25: 114.15 ms  153.8 GB/s
  iter 29: 117.17 ms  149.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.03 ms
achieved bandwidth       : 150.0 GB/s
projected decode         : 8.55 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T111244Z.json
[round 371] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T111244Z.json
```
