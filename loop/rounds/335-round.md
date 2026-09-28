# Round 335 — 20260928T015315Z

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
[round 335] batch-parity OK
[round 335] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 335] prefix cache OK
[round 335] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.746       1.421    8.97x     2.00e-6   (126 GB/s serial, 1134 GB/s warp)

decode-bench: OK
[round 335] decode-bench OK
[round 335] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.4s
interleave: false
  iter  0: 120.06 ms  146.2 GB/s
  iter  5: 117.73 ms  149.1 GB/s
  iter 10: 115.56 ms  151.9 GB/s
  iter 15: 115.09 ms  152.5 GB/s
  iter 20: 118.65 ms  148.0 GB/s
  iter 25: 115.39 ms  152.1 GB/s
  iter 29: 116.17 ms  151.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 115.87 ms
achieved bandwidth       : 151.5 GB/s
projected decode         : 8.63 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T015315Z.json
[round 335] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T015315Z.json
```
