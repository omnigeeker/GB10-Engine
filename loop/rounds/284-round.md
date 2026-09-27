# Round 284 — 20260927T175626Z

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
[round 284] batch-parity OK
[round 284] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 284] prefix cache OK
[round 284] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.569       1.406    6.81x     2.00e-6   (168 GB/s serial, 1146 GB/s warp)

decode-bench: OK
[round 284] decode-bench OK
[round 284] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.1s
interleave: false
  iter  0: 99.99 ms  175.6 GB/s
  iter  5: 99.10 ms  177.2 GB/s
  iter 10: 97.89 ms  179.3 GB/s
  iter 15: 99.13 ms  177.1 GB/s
  iter 20: 98.93 ms  177.4 GB/s
  iter 25: 99.21 ms  176.9 GB/s
  iter 29: 97.66 ms  179.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 99.33 ms
achieved bandwidth       : 176.7 GB/s
projected decode         : 10.07 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 77.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T175626Z.json
[round 284] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T175626Z.json
```
