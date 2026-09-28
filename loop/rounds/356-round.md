# Round 356 — 20260928T085655Z

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
[round 356] batch-parity OK
[round 356] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 356] prefix cache OK
[round 356] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.335       0.962   13.86x     1.96e-6   (121 GB/s serial, 1674 GB/s warp)

decode-bench: OK
[round 356] decode-bench OK
[round 356] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.0s
interleave: false
  iter  0: 116.66 ms  150.5 GB/s
  iter  5: 122.05 ms  143.8 GB/s
  iter 10: 116.74 ms  150.4 GB/s
  iter 15: 116.49 ms  150.7 GB/s
  iter 20: 120.92 ms  145.2 GB/s
  iter 25: 117.51 ms  149.4 GB/s
  iter 29: 115.91 ms  151.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.35 ms
achieved bandwidth       : 149.6 GB/s
projected decode         : 8.52 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T085655Z.json
[round 356] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T085655Z.json
```
