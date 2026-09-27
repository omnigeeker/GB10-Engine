# Round 305 — 20260927T204800Z

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
[round 305] batch-parity OK
[round 305] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 305] prefix cache OK
[round 305] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.494       1.414   10.96x     2.00e-6   (104 GB/s serial, 1139 GB/s warp)

decode-bench: OK
[round 305] decode-bench OK
[round 305] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.3s
interleave: false
  iter  0: 99.05 ms  177.2 GB/s
  iter  5: 99.56 ms  176.3 GB/s
  iter 10: 98.21 ms  178.8 GB/s
  iter 15: 99.59 ms  176.3 GB/s
  iter 20: 101.09 ms  173.7 GB/s
  iter 25: 98.09 ms  179.0 GB/s
  iter 29: 97.83 ms  179.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.03 ms
achieved bandwidth       : 179.1 GB/s
projected decode         : 10.20 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T204800Z.json
[round 305] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T204800Z.json
```
