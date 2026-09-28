# Round 337 — 20260928T021014Z

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
[round 337] batch-parity OK
[round 337] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 337] prefix cache OK
[round 337] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.991       1.700    7.64x     2.00e-6   (124 GB/s serial, 948 GB/s warp)

decode-bench: OK
[round 337] decode-bench OK
[round 337] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.5s
interleave: false
  iter  0: 114.02 ms  154.0 GB/s
  iter  5: 113.03 ms  155.3 GB/s
  iter 10: 118.51 ms  148.1 GB/s
  iter 15: 116.26 ms  151.0 GB/s
  iter 20: 113.81 ms  154.2 GB/s
  iter 25: 116.83 ms  150.3 GB/s
  iter 29: 116.91 ms  150.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 115.71 ms
achieved bandwidth       : 151.7 GB/s
projected decode         : 8.64 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T021014Z.json
[round 337] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T021014Z.json
```
