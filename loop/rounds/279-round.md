# Round 279 — 20260927T170555Z

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
[round 279] batch-parity OK
[round 279] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 279] prefix cache OK
[round 279] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.527       1.395   11.13x     2.00e-6   (104 GB/s serial, 1154 GB/s warp)

decode-bench: OK
[round 279] decode-bench OK
[round 279] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 98.28 ms  178.6 GB/s
  iter  5: 96.08 ms  182.7 GB/s
  iter 10: 97.10 ms  180.8 GB/s
  iter 15: 97.66 ms  179.8 GB/s
  iter 20: 97.43 ms  180.2 GB/s
  iter 25: 98.74 ms  177.8 GB/s
  iter 29: 97.90 ms  179.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.86 ms
achieved bandwidth       : 181.2 GB/s
projected decode         : 10.32 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T170555Z.json
[round 279] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T170555Z.json
```
