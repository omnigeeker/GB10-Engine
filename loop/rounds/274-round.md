# Round 274 — 20260927T162441Z

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
[round 274] batch-parity OK
[round 274] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 274] prefix cache OK
[round 274] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.669       1.465    6.60x     2.00e-6   (167 GB/s serial, 1100 GB/s warp)

decode-bench: OK
[round 274] decode-bench OK
[round 274] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 98.86 ms  177.6 GB/s
  iter  5: 98.24 ms  178.7 GB/s
  iter 10: 98.92 ms  177.5 GB/s
  iter 15: 98.25 ms  178.7 GB/s
  iter 20: 97.73 ms  179.6 GB/s
  iter 25: 98.15 ms  178.9 GB/s
  iter 29: 98.19 ms  178.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.70 ms
achieved bandwidth       : 179.7 GB/s
projected decode         : 10.24 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T162441Z.json
[round 274] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T162441Z.json
```
