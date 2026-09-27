# Round 283 — 20260927T174910Z

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
[round 283] batch-parity OK
[round 283] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 283] prefix cache OK
[round 283] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.609       1.291   12.09x     2.00e-6   (103 GB/s serial, 1247 GB/s warp)

decode-bench: OK
[round 283] decode-bench OK
[round 283] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 95.70 ms  183.4 GB/s
  iter  5: 95.80 ms  183.3 GB/s
  iter 10: 95.21 ms  184.4 GB/s
  iter 15: 96.30 ms  182.3 GB/s
  iter 20: 96.71 ms  181.5 GB/s
  iter 25: 94.59 ms  185.6 GB/s
  iter 29: 95.43 ms  184.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 95.83 ms
achieved bandwidth       : 183.2 GB/s
projected decode         : 10.43 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T174910Z.json
[round 283] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T174910Z.json
```
