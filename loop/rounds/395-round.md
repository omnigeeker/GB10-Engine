# Round 395 — 20260928T145954Z

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
[round 395] batch-parity OK
[round 395] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 395] prefix cache OK
[round 395] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.938       1.201   10.77x     1.96e-6   (124 GB/s serial, 1341 GB/s warp)

decode-bench: OK
[round 395] decode-bench OK
[round 395] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.5s
interleave: false
  iter  0: 97.79 ms  179.5 GB/s
  iter  5: 97.15 ms  180.7 GB/s
  iter 10: 97.44 ms  180.2 GB/s
  iter 15: 97.18 ms  180.7 GB/s
  iter 20: 97.10 ms  180.8 GB/s
  iter 25: 100.11 ms  175.4 GB/s
  iter 29: 97.46 ms  180.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.53 ms
achieved bandwidth       : 180.0 GB/s
projected decode         : 10.25 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T145954Z.json
[round 395] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T145954Z.json
```
