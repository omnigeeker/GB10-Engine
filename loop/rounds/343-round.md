# Round 343 — 20260928T031716Z

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
[round 343] batch-parity OK
[round 343] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 343] prefix cache OK
[round 343] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.393       1.506    8.23x     2.00e-6   (130 GB/s serial, 1069 GB/s warp)

decode-bench: OK
[round 343] decode-bench OK
[round 343] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.1s
interleave: false
  iter  0: 112.93 ms  155.5 GB/s
  iter  5: 116.84 ms  150.3 GB/s
  iter 10: 112.54 ms  156.0 GB/s
  iter 15: 112.00 ms  156.7 GB/s
  iter 20: 113.83 ms  154.2 GB/s
  iter 25: 112.40 ms  156.2 GB/s
  iter 29: 116.24 ms  151.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 113.75 ms
achieved bandwidth       : 154.3 GB/s
projected decode         : 8.79 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 67.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T031716Z.json
[round 343] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T031716Z.json
```
