# Round 391 — 20260928T142124Z

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
[round 391] batch-parity OK
[round 391] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 391] prefix cache OK
[round 391] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.336       1.237    9.97x     1.96e-6   (131 GB/s serial, 1302 GB/s warp)

decode-bench: OK
[round 391] decode-bench OK
[round 391] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.5s
interleave: false
  iter  0: 96.02 ms  182.8 GB/s
  iter  5: 96.45 ms  182.0 GB/s
  iter 10: 96.89 ms  181.2 GB/s
  iter 15: 97.95 ms  179.2 GB/s
  iter 20: 95.10 ms  184.6 GB/s
  iter 25: 94.74 ms  185.3 GB/s
  iter 29: 95.94 ms  183.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.00 ms
achieved bandwidth       : 182.9 GB/s
projected decode         : 10.42 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T142124Z.json
[round 391] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T142124Z.json
```
