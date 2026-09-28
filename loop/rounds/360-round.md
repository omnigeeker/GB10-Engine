# Round 360 — 20260928T093301Z

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
[round 360] batch-parity OK
[round 360] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 360] prefix cache OK
[round 360] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      14.047       0.881   15.94x     1.96e-6   (115 GB/s serial, 1828 GB/s warp)

decode-bench: OK
[round 360] decode-bench OK
[round 360] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.9s
interleave: false
  iter  0: 117.44 ms  149.5 GB/s
  iter  5: 120.03 ms  146.3 GB/s
  iter 10: 116.91 ms  150.2 GB/s
  iter 15: 120.91 ms  145.2 GB/s
  iter 20: 118.07 ms  148.7 GB/s
  iter 25: 113.09 ms  155.2 GB/s
  iter 29: 116.61 ms  150.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.47 ms
achieved bandwidth       : 149.4 GB/s
projected decode         : 8.51 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T093301Z.json
[round 360] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T093301Z.json
```
