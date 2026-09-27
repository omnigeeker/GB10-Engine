# Round 324 — 20260927T234439Z

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
[round 324] batch-parity OK
[round 324] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 324] prefix cache OK
[round 324] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.576       1.349    7.10x     2.00e-6   (168 GB/s serial, 1194 GB/s warp)

decode-bench: OK
[round 324] decode-bench OK
[round 324] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.4s
interleave: false
  iter  0: 99.83 ms  175.9 GB/s
  iter  5: 97.16 ms  180.7 GB/s
  iter 10: 95.17 ms  184.5 GB/s
  iter 15: 97.77 ms  179.6 GB/s
  iter 20: 96.82 ms  181.3 GB/s
  iter 25: 97.40 ms  180.2 GB/s
  iter 29: 95.95 ms  183.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.06 ms
achieved bandwidth       : 180.9 GB/s
projected decode         : 10.30 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T234439Z.json
[round 324] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T234439Z.json
```
