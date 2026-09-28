# Round 394 — 20260928T145347Z

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
[round 394] batch-parity OK
[round 394] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 394] prefix cache OK
[round 394] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.355       1.247    9.91x     1.96e-6   (130 GB/s serial, 1292 GB/s warp)

decode-bench: OK
[round 394] decode-bench OK
[round 394] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.3s
interleave: false
  iter  0: 97.83 ms  179.5 GB/s
  iter  5: 97.95 ms  179.2 GB/s
  iter 10: 95.70 ms  183.4 GB/s
  iter 15: 98.93 ms  177.4 GB/s
  iter 20: 98.86 ms  177.6 GB/s
  iter 25: 98.49 ms  178.2 GB/s
  iter 29: 99.11 ms  177.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.44 ms
achieved bandwidth       : 178.3 GB/s
projected decode         : 10.16 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T145347Z.json
[round 394] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T145347Z.json
```
