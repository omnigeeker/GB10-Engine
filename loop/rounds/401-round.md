# Round 401 — 20260928T171643Z

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
[round 401] batch-parity OK
[round 401] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 401] prefix cache OK
[round 401] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.866       1.198   10.74x     1.96e-6   (125 GB/s serial, 1344 GB/s warp)

decode-bench: OK
[round 401] decode-bench OK
[round 401] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.7s
interleave: false
  iter  0: 97.68 ms  179.7 GB/s
  iter  5: 98.71 ms  177.9 GB/s
  iter 10: 98.72 ms  177.8 GB/s
  iter 15: 99.15 ms  177.1 GB/s
  iter 20: 98.33 ms  178.5 GB/s
  iter 25: 97.18 ms  180.6 GB/s
  iter 29: 97.12 ms  180.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.03 ms
achieved bandwidth       : 179.1 GB/s
projected decode         : 10.20 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T171643Z.json
[round 401] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T171643Z.json
```
