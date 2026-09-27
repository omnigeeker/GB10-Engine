# Round 265 — 20260927T143702Z

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
[round 265] batch-parity OK
[round 265] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 265] prefix cache OK
[round 265] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.391       1.363   11.30x     2.00e-6   (105 GB/s serial, 1182 GB/s warp)

decode-bench: OK
[round 265] decode-bench OK
[round 265] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.5s
interleave: false
  iter  0: 95.80 ms  183.3 GB/s
  iter  5: 96.93 ms  181.1 GB/s
  iter 10: 95.43 ms  184.0 GB/s
  iter 15: 96.66 ms  181.6 GB/s
  iter 20: 97.91 ms  179.3 GB/s
  iter 25: 99.26 ms  176.9 GB/s
  iter 29: 96.24 ms  182.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.06 ms
achieved bandwidth       : 180.9 GB/s
projected decode         : 10.30 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T143702Z.json
[round 265] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T143702Z.json
```
