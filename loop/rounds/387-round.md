# Round 387 — 20260928T134050Z

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
[round 387] batch-parity OK
[round 387] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 387] prefix cache OK
[round 387] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.403       1.556    8.61x     1.96e-6   (120 GB/s serial, 1035 GB/s warp)

decode-bench: OK
[round 387] decode-bench OK
[round 387] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.8s
interleave: false
  iter  0: 116.70 ms  150.4 GB/s
  iter  5: 113.02 ms  155.3 GB/s
  iter 10: 116.38 ms  150.8 GB/s
  iter 15: 114.06 ms  153.9 GB/s
  iter 20: 115.36 ms  152.2 GB/s
  iter 25: 114.21 ms  153.7 GB/s
  iter 29: 116.03 ms  151.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.26 ms
achieved bandwidth       : 151.0 GB/s
projected decode         : 8.60 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T134050Z.json
[round 387] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T134050Z.json
```
