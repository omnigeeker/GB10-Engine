# Round 341 — 20260928T024637Z

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
[round 341] batch-parity OK
[round 341] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 341] prefix cache OK
[round 341] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.096       1.714    7.64x     2.00e-6   (123 GB/s serial, 940 GB/s warp)

decode-bench: OK
[round 341] decode-bench OK
[round 341] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.5s
interleave: false
  iter  0: 113.80 ms  154.3 GB/s
  iter  5: 118.07 ms  148.7 GB/s
  iter 10: 117.50 ms  149.4 GB/s
  iter 15: 114.77 ms  153.0 GB/s
  iter 20: 116.44 ms  150.8 GB/s
  iter 25: 114.25 ms  153.7 GB/s
  iter 29: 116.29 ms  151.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.50 ms
achieved bandwidth       : 150.7 GB/s
projected decode         : 8.58 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T024637Z.json
[round 341] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T024637Z.json
```
