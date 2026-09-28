# Round 334 — 20260928T014431Z

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
[round 334] batch-parity OK
[round 334] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 334] prefix cache OK
[round 334] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.475       1.588    7.85x     2.00e-6   (129 GB/s serial, 1014 GB/s warp)

decode-bench: OK
[round 334] decode-bench OK
[round 334] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.4s
interleave: false
  iter  0: 115.32 ms  152.2 GB/s
  iter  5: 116.87 ms  150.2 GB/s
  iter 10: 115.09 ms  152.5 GB/s
  iter 15: 116.43 ms  150.8 GB/s
  iter 20: 114.90 ms  152.8 GB/s
  iter 25: 118.00 ms  148.8 GB/s
  iter 29: 115.90 ms  151.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.32 ms
achieved bandwidth       : 150.9 GB/s
projected decode         : 8.60 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T014431Z.json
[round 334] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T014431Z.json
```
