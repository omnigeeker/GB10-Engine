# Round 388 — 20260928T135209Z

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
[round 388] batch-parity OK
[round 388] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 388] prefix cache OK
[round 388] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.723       1.789    7.11x     1.96e-6   (127 GB/s serial, 900 GB/s warp)

decode-bench: OK
[round 388] decode-bench OK
[round 388] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.3s
interleave: false
  iter  0: 121.57 ms  144.4 GB/s
  iter  5: 114.21 ms  153.7 GB/s
  iter 10: 115.45 ms  152.1 GB/s
  iter 15: 114.13 ms  153.8 GB/s
  iter 20: 111.25 ms  157.8 GB/s
  iter 25: 116.95 ms  150.1 GB/s
  iter 29: 115.26 ms  152.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 115.37 ms
achieved bandwidth       : 152.2 GB/s
projected decode         : 8.67 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T135209Z.json
[round 388] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T135209Z.json
```
