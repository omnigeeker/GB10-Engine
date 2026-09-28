# Round 390 — 20260928T141038Z

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
[round 390] batch-parity OK
[round 390] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 390] prefix cache OK
[round 390] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.722       1.417    9.68x     1.96e-6   (117 GB/s serial, 1136 GB/s warp)

decode-bench: OK
[round 390] decode-bench OK
[round 390] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.9s
interleave: false
  iter  0: 120.09 ms  146.2 GB/s
  iter  5: 117.59 ms  149.3 GB/s
  iter 10: 117.25 ms  149.7 GB/s
  iter 15: 118.54 ms  148.1 GB/s
  iter 20: 114.88 ms  152.8 GB/s
  iter 25: 116.25 ms  151.0 GB/s
  iter 29: 119.44 ms  147.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.26 ms
achieved bandwidth       : 149.7 GB/s
projected decode         : 8.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T141038Z.json
[round 390] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T141038Z.json
```
