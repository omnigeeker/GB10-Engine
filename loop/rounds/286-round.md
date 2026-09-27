# Round 286 — 20260927T181209Z

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
[round 286] batch-parity OK
[round 286] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 286] prefix cache OK
[round 286] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.575       1.378    6.95x     2.00e-6   (168 GB/s serial, 1169 GB/s warp)

decode-bench: OK
[round 286] decode-bench OK
[round 286] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.9s
interleave: false
  iter  0: 98.97 ms  177.4 GB/s
  iter  5: 97.88 ms  179.4 GB/s
  iter 10: 95.80 ms  183.2 GB/s
  iter 15: 95.33 ms  184.2 GB/s
  iter 20: 97.39 ms  180.3 GB/s
  iter 25: 99.64 ms  176.2 GB/s
  iter 29: 95.86 ms  183.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.83 ms
achieved bandwidth       : 179.4 GB/s
projected decode         : 10.22 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T181209Z.json
[round 286] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T181209Z.json
```
