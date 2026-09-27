# Round 310 — 20260927T213849Z

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
[round 310] batch-parity OK
[round 310] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 310] prefix cache OK
[round 310] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.382       1.377    6.81x     2.00e-6   (172 GB/s serial, 1170 GB/s warp)

decode-bench: OK
[round 310] decode-bench OK
[round 310] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.2s
interleave: false
  iter  0: 98.67 ms  177.9 GB/s
  iter  5: 98.74 ms  177.8 GB/s
  iter 10: 97.53 ms  180.0 GB/s
  iter 15: 95.43 ms  184.0 GB/s
  iter 20: 93.97 ms  186.8 GB/s
  iter 25: 95.58 ms  183.7 GB/s
  iter 29: 96.56 ms  181.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.82 ms
achieved bandwidth       : 181.3 GB/s
projected decode         : 10.33 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T213849Z.json
[round 310] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T213849Z.json
```
