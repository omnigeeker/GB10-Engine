# Round 292 — 20260927T190055Z

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
[round 292] batch-parity OK
[round 292] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 292] prefix cache OK
[round 292] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.503       1.385    6.86x     2.00e-6   (169 GB/s serial, 1163 GB/s warp)

decode-bench: OK
[round 292] decode-bench OK
[round 292] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.8s
interleave: false
  iter  0: 98.22 ms  178.7 GB/s
  iter  5: 96.15 ms  182.6 GB/s
  iter 10: 96.72 ms  181.5 GB/s
  iter 15: 97.35 ms  180.3 GB/s
  iter 20: 98.62 ms  178.0 GB/s
  iter 25: 99.35 ms  176.7 GB/s
  iter 29: 96.27 ms  182.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.03 ms
achieved bandwidth       : 179.1 GB/s
projected decode         : 10.20 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T190055Z.json
[round 292] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T190055Z.json
```
