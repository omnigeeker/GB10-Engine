# Round 333 — 20260928T013425Z

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
[round 333] batch-parity OK
[round 333] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 333] prefix cache OK
[round 333] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.814       1.658    7.73x     2.00e-6   (126 GB/s serial, 971 GB/s warp)

decode-bench: OK
[round 333] decode-bench OK
[round 333] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.0s
interleave: false
  iter  0: 117.94 ms  148.9 GB/s
  iter  5: 116.51 ms  150.7 GB/s
  iter 10: 114.74 ms  153.0 GB/s
  iter 15: 115.13 ms  152.5 GB/s
  iter 20: 115.62 ms  151.8 GB/s
  iter 25: 119.15 ms  147.3 GB/s
  iter 29: 119.21 ms  147.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.32 ms
achieved bandwidth       : 150.9 GB/s
projected decode         : 8.60 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T013425Z.json
[round 333] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T013425Z.json
```
