# Round 311 — 20260927T214529Z

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
[round 311] batch-parity OK
[round 311] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 311] prefix cache OK
[round 311] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.483       1.373    6.91x     2.00e-6   (170 GB/s serial, 1173 GB/s warp)

decode-bench: OK
[round 311] decode-bench OK
[round 311] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.1s
interleave: false
  iter  0: 98.52 ms  178.2 GB/s
  iter  5: 99.23 ms  176.9 GB/s
  iter 10: 98.24 ms  178.7 GB/s
  iter 15: 97.89 ms  179.3 GB/s
  iter 20: 97.48 ms  180.1 GB/s
  iter 25: 97.24 ms  180.5 GB/s
  iter 29: 98.96 ms  177.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.86 ms
achieved bandwidth       : 179.4 GB/s
projected decode         : 10.22 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T214529Z.json
[round 311] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T214529Z.json
```
