# Round 285 — 20260927T180311Z

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
[round 285] batch-parity OK
[round 285] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 285] prefix cache OK
[round 285] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.577       1.359    7.05x     2.00e-6   (168 GB/s serial, 1185 GB/s warp)

decode-bench: OK
[round 285] decode-bench OK
[round 285] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.9s
interleave: false
  iter  0: 96.22 ms  182.5 GB/s
  iter  5: 98.05 ms  179.0 GB/s
  iter 10: 97.76 ms  179.6 GB/s
  iter 15: 98.34 ms  178.5 GB/s
  iter 20: 97.09 ms  180.8 GB/s
  iter 25: 97.49 ms  180.1 GB/s
  iter 29: 99.22 ms  176.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.69 ms
achieved bandwidth       : 179.7 GB/s
projected decode         : 10.24 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T180311Z.json
[round 285] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T180311Z.json
```
