# Round 313 — 20260927T220337Z

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
[round 313] batch-parity OK
[round 313] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 313] prefix cache OK
[round 313] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.656       1.409    6.85x     2.00e-6   (167 GB/s serial, 1143 GB/s warp)

decode-bench: OK
[round 313] decode-bench OK
[round 313] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.3s
interleave: false
  iter  0: 97.74 ms  179.6 GB/s
  iter  5: 96.58 ms  181.8 GB/s
  iter 10: 98.37 ms  178.5 GB/s
  iter 15: 96.38 ms  182.2 GB/s
  iter 20: 97.22 ms  180.6 GB/s
  iter 25: 97.92 ms  179.3 GB/s
  iter 29: 95.97 ms  182.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.03 ms
achieved bandwidth       : 180.9 GB/s
projected decode         : 10.31 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T220337Z.json
[round 313] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T220337Z.json
```
