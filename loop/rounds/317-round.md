# Round 317 — 20260927T224015Z

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
[round 317] batch-parity OK
[round 317] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 317] prefix cache OK
[round 317] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.003       1.614    9.29x     2.00e-6   (107 GB/s serial, 998 GB/s warp)

decode-bench: OK
[round 317] decode-bench OK
[round 317] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.5s
interleave: false
  iter  0: 97.52 ms  180.0 GB/s
  iter  5: 98.49 ms  178.2 GB/s
  iter 10: 99.14 ms  177.1 GB/s
  iter 15: 98.84 ms  177.6 GB/s
  iter 20: 99.87 ms  175.8 GB/s
  iter 25: 95.80 ms  183.2 GB/s
  iter 29: 99.71 ms  176.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.55 ms
achieved bandwidth       : 178.1 GB/s
projected decode         : 10.15 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T224015Z.json
[round 317] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T224015Z.json
```
