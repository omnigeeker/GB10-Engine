# Round 403 — 20260928T173010Z

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
[round 403] batch-parity OK
[round 403] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 403] prefix cache OK
[round 403] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.168       1.249    9.74x     1.96e-6   (132 GB/s serial, 1289 GB/s warp)

decode-bench: OK
[round 403] decode-bench OK
[round 403] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.6s
interleave: false
  iter  0: 98.47 ms  178.3 GB/s
  iter  5: 97.95 ms  179.2 GB/s
  iter 10: 95.90 ms  183.1 GB/s
  iter 15: 95.86 ms  183.1 GB/s
  iter 20: 95.93 ms  183.0 GB/s
  iter 25: 98.14 ms  178.9 GB/s
  iter 29: 96.85 ms  181.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.11 ms
achieved bandwidth       : 180.8 GB/s
projected decode         : 10.30 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T173010Z.json
[round 403] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T173010Z.json
```
