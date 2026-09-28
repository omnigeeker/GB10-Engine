# Round 373 — 20260928T113136Z

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
[round 373] batch-parity OK
[round 373] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 373] prefix cache OK
[round 373] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.820       1.118   12.36x     1.96e-6   (117 GB/s serial, 1440 GB/s warp)

decode-bench: OK
[round 373] decode-bench OK
[round 373] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.3s
interleave: false
  iter  0: 120.46 ms  145.7 GB/s
  iter  5: 117.06 ms  150.0 GB/s
  iter 10: 114.55 ms  153.3 GB/s
  iter 15: 117.34 ms  149.6 GB/s
  iter 20: 116.44 ms  150.8 GB/s
  iter 25: 121.91 ms  144.0 GB/s
  iter 29: 115.31 ms  152.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.94 ms
achieved bandwidth       : 150.1 GB/s
projected decode         : 8.55 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T113136Z.json
[round 373] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T113136Z.json
```
