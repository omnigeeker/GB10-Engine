# Round 408 — 20260929T024829Z

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
[round 408] batch-parity OK
[round 408] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 408] prefix cache OK
[round 408] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.710       1.898    6.70x     1.96e-6   (127 GB/s serial, 848 GB/s warp)

decode-bench: OK
[round 408] decode-bench OK
[round 408] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.1s
interleave: false
  iter  0: 117.85 ms  149.0 GB/s
  iter  5: 116.90 ms  150.2 GB/s
  iter 10: 120.73 ms  145.4 GB/s
  iter 15: 113.39 ms  154.8 GB/s
  iter 20: 119.01 ms  147.5 GB/s
  iter 25: 116.89 ms  150.2 GB/s
  iter 29: 114.15 ms  153.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 118.24 ms
achieved bandwidth       : 148.5 GB/s
projected decode         : 8.46 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260929T024829Z.json
[round 408] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260929T024829Z.json
```
