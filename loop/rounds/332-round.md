# Round 332 — 20260928T011934Z

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
[round 332] batch-parity OK
[round 332] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 332] prefix cache OK
[round 332] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.552       1.638    7.66x     2.00e-6   (128 GB/s serial, 983 GB/s warp)

decode-bench: OK
[round 332] decode-bench OK
[round 332] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.8s
interleave: false
  iter  0: 115.06 ms  152.6 GB/s
  iter  5: 119.28 ms  147.2 GB/s
  iter 10: 116.08 ms  151.2 GB/s
  iter 15: 117.01 ms  150.0 GB/s
  iter 20: 112.73 ms  155.7 GB/s
  iter 25: 116.98 ms  150.1 GB/s
  iter 29: 114.87 ms  152.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.61 ms
achieved bandwidth       : 150.5 GB/s
projected decode         : 8.58 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T011934Z.json
[round 332] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T011934Z.json
```
