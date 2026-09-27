# Round 290 — 20260927T184840Z

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
[round 290] batch-parity OK
[round 290] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 290] prefix cache OK
[round 290] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.536       1.491    6.40x     2.00e-6   (169 GB/s serial, 1080 GB/s warp)

decode-bench: OK
[round 290] decode-bench OK
[round 290] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 35.5s
interleave: false
  iter  0: 98.62 ms  178.0 GB/s
  iter  5: 99.73 ms  176.0 GB/s
  iter 10: 99.76 ms  176.0 GB/s
  iter 15: 99.06 ms  177.2 GB/s
  iter 20: 98.42 ms  178.4 GB/s
  iter 25: 98.50 ms  178.2 GB/s
  iter 29: 100.15 ms  175.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 99.06 ms
achieved bandwidth       : 177.2 GB/s
projected decode         : 10.10 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 77.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T184840Z.json
[round 290] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T184840Z.json
```
