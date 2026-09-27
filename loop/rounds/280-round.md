# Round 280 — 20260927T171250Z

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
[round 280] batch-parity OK
[round 280] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 280] prefix cache OK
[round 280] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.405       1.407    6.69x     2.00e-6   (171 GB/s serial, 1145 GB/s warp)

decode-bench: OK
[round 280] decode-bench OK
[round 280] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.4s
interleave: false
  iter  0: 99.32 ms  176.8 GB/s
  iter  5: 97.80 ms  179.5 GB/s
  iter 10: 97.13 ms  180.7 GB/s
  iter 15: 99.03 ms  177.3 GB/s
  iter 20: 98.85 ms  177.6 GB/s
  iter 25: 96.35 ms  182.2 GB/s
  iter 29: 97.82 ms  179.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.21 ms
achieved bandwidth       : 178.8 GB/s
projected decode         : 10.18 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T171250Z.json
[round 280] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T171250Z.json
```
