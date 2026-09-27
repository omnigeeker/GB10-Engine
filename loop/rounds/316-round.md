# Round 316 — 20260927T222907Z

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
[round 316] batch-parity OK
[round 316] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 316] prefix cache OK
[round 316] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.707       1.440    6.74x     2.00e-6   (166 GB/s serial, 1118 GB/s warp)

decode-bench: OK
[round 316] decode-bench OK
[round 316] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.4s
interleave: false
  iter  0: 99.52 ms  176.4 GB/s
  iter  5: 99.29 ms  176.8 GB/s
  iter 10: 97.98 ms  179.2 GB/s
  iter 15: 97.44 ms  180.2 GB/s
  iter 20: 100.05 ms  175.5 GB/s
  iter 25: 98.37 ms  178.5 GB/s
  iter 29: 99.32 ms  176.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.39 ms
achieved bandwidth       : 178.4 GB/s
projected decode         : 10.16 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T222907Z.json
[round 316] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T222907Z.json
```
