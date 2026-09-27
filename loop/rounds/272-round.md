# Round 272 — 20260927T160922Z

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
[round 272] batch-parity OK
[round 272] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 272] prefix cache OK
[round 272] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.493       1.390   11.14x     2.00e-6   (104 GB/s serial, 1158 GB/s warp)

decode-bench: OK
[round 272] decode-bench OK
[round 272] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.1s
interleave: false
  iter  0: 97.79 ms  179.5 GB/s
  iter  5: 98.44 ms  178.3 GB/s
  iter 10: 94.80 ms  185.2 GB/s
  iter 15: 96.09 ms  182.7 GB/s
  iter 20: 98.38 ms  178.4 GB/s
  iter 25: 99.04 ms  177.3 GB/s
  iter 29: 97.83 ms  179.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.40 ms
achieved bandwidth       : 180.2 GB/s
projected decode         : 10.27 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T160922Z.json
[round 272] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T160922Z.json
```
