# Round 357 — 20260928T090639Z

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
[round 357] batch-parity OK
[round 357] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 357] prefix cache OK
[round 357] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.663       0.906   15.08x     1.96e-6   (118 GB/s serial, 1778 GB/s warp)

decode-bench: OK
[round 357] decode-bench OK
[round 357] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.5s
interleave: false
  iter  0: 115.47 ms  152.0 GB/s
  iter  5: 116.33 ms  150.9 GB/s
  iter 10: 121.05 ms  145.0 GB/s
  iter 15: 116.62 ms  150.5 GB/s
  iter 20: 114.83 ms  152.9 GB/s
  iter 25: 116.15 ms  151.1 GB/s
  iter 29: 117.92 ms  148.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.05 ms
achieved bandwidth       : 150.0 GB/s
projected decode         : 8.54 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T090639Z.json
[round 357] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T090639Z.json
```
