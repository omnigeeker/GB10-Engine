# Round 349 — 20260928T053826Z

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
[round 349] batch-parity OK
[round 349] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 349] prefix cache OK
[round 349] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.006       1.658    7.85x     2.00e-6   (124 GB/s serial, 972 GB/s warp)

decode-bench: OK
[round 349] decode-bench OK
[round 349] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.3s
interleave: false
  iter  0: 120.29 ms  145.9 GB/s
  iter  5: 115.74 ms  151.7 GB/s
  iter 10: 116.03 ms  151.3 GB/s
  iter 15: 117.46 ms  149.5 GB/s
  iter 20: 117.94 ms  148.8 GB/s
  iter 25: 117.99 ms  148.8 GB/s
  iter 29: 116.17 ms  151.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.06 ms
achieved bandwidth       : 150.0 GB/s
projected decode         : 8.54 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T053826Z.json
[round 349] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T053826Z.json
```
