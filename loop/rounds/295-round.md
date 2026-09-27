# Round 295 — 20260927T192127Z

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
[round 295] batch-parity OK
[round 295] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 295] prefix cache OK
[round 295] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.311       1.396    6.67x     2.00e-6   (173 GB/s serial, 1154 GB/s warp)

decode-bench: OK
[round 295] decode-bench OK
[round 295] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.6s
interleave: false
  iter  0: 98.90 ms  177.5 GB/s
  iter  5: 99.72 ms  176.0 GB/s
  iter 10: 100.53 ms  174.6 GB/s
  iter 15: 96.98 ms  181.0 GB/s
  iter 20: 97.63 ms  179.8 GB/s
  iter 25: 98.95 ms  177.4 GB/s
  iter 29: 96.23 ms  182.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.57 ms
achieved bandwidth       : 178.1 GB/s
projected decode         : 10.14 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T192127Z.json
[round 295] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T192127Z.json
```
