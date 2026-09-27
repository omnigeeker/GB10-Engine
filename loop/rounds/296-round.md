# Round 296 — 20260927T192725Z

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
[round 296] batch-parity OK
[round 296] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 296] prefix cache OK
[round 296] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.587       1.407    6.81x     2.00e-6   (168 GB/s serial, 1144 GB/s warp)

decode-bench: OK
[round 296] decode-bench OK
[round 296] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.1s
interleave: false
  iter  0: 97.78 ms  179.5 GB/s
  iter  5: 96.95 ms  181.1 GB/s
  iter 10: 96.42 ms  182.1 GB/s
  iter 15: 99.90 ms  175.7 GB/s
  iter 20: 95.71 ms  183.4 GB/s
  iter 25: 96.59 ms  181.8 GB/s
  iter 29: 98.73 ms  177.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.11 ms
achieved bandwidth       : 180.8 GB/s
projected decode         : 10.30 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T192725Z.json
[round 296] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T192725Z.json
```
