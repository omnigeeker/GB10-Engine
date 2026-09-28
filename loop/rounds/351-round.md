# Round 351 — 20260928T060714Z

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
[round 351] batch-parity OK
[round 351] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 351] prefix cache OK
[round 351] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.918       1.608    8.04x     2.00e-6   (125 GB/s serial, 1002 GB/s warp)

decode-bench: OK
[round 351] decode-bench OK
[round 351] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.7s
interleave: false
  iter  0: 119.78 ms  146.6 GB/s
  iter  5: 117.22 ms  149.8 GB/s
  iter 10: 116.47 ms  150.7 GB/s
  iter 15: 116.02 ms  151.3 GB/s
  iter 20: 113.02 ms  155.3 GB/s
  iter 25: 115.83 ms  151.6 GB/s
  iter 29: 117.04 ms  150.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.68 ms
achieved bandwidth       : 149.2 GB/s
projected decode         : 8.50 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T060714Z.json
[round 351] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T060714Z.json
```
