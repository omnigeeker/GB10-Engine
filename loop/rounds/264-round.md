# Round 264 — 20260927T140938Z

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
[round 264] batch-parity OK
[round 264] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 264] prefix cache OK
[round 264] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.645       1.410    6.84x     2.00e-6   (167 GB/s serial, 1143 GB/s warp)

decode-bench: OK
[round 264] decode-bench OK
[round 264] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.2s
interleave: false
  iter  0: 95.87 ms  183.1 GB/s
  iter  5: 95.50 ms  183.8 GB/s
  iter 10: 94.63 ms  185.5 GB/s
  iter 15: 95.08 ms  184.6 GB/s
  iter 20: 94.35 ms  186.1 GB/s
  iter 25: 96.21 ms  182.5 GB/s
  iter 29: 93.64 ms  187.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 94.94 ms
achieved bandwidth       : 184.9 GB/s
projected decode         : 10.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 81.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T140938Z.json
[round 264] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T140938Z.json
```
