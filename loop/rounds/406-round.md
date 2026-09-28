# Round 406 — 20260928T191655Z

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
[round 406] batch-parity OK
[round 406] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 406] prefix cache OK
[round 406] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.948       1.247   10.38x     1.96e-6   (124 GB/s serial, 1292 GB/s warp)

decode-bench: OK
[round 406] decode-bench OK
[round 406] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.7s
interleave: false
  iter  0: 99.83 ms  175.9 GB/s
  iter  5: 96.99 ms  181.0 GB/s
  iter 10: 99.40 ms  176.6 GB/s
  iter 15: 97.28 ms  180.5 GB/s
  iter 20: 98.55 ms  178.1 GB/s
  iter 25: 96.96 ms  181.1 GB/s
  iter 29: 96.01 ms  182.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.61 ms
achieved bandwidth       : 179.8 GB/s
projected decode         : 10.24 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T191655Z.json
[round 406] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T191655Z.json
```
