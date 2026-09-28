# Round 370 — 20260928T110258Z

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
[round 370] batch-parity OK
[round 370] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 370] prefix cache OK
[round 370] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      14.000       0.999   14.01x     1.96e-6   (115 GB/s serial, 1612 GB/s warp)

decode-bench: OK
[round 370] decode-bench OK
[round 370] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.3s
interleave: false
  iter  0: 118.78 ms  147.8 GB/s
  iter  5: 116.82 ms  150.3 GB/s
  iter 10: 118.85 ms  147.7 GB/s
  iter 15: 119.04 ms  147.5 GB/s
  iter 20: 119.22 ms  147.3 GB/s
  iter 25: 117.15 ms  149.9 GB/s
  iter 29: 116.52 ms  150.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.18 ms
achieved bandwidth       : 149.8 GB/s
projected decode         : 8.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T110258Z.json
[round 370] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T110258Z.json
```
