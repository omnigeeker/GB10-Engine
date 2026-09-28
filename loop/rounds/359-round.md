# Round 359 — 20260928T092402Z

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
[round 359] batch-parity OK
[round 359] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 359] prefix cache OK
[round 359] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.451       0.862   15.60x     1.96e-6   (120 GB/s serial, 1868 GB/s warp)

decode-bench: OK
[round 359] decode-bench OK
[round 359] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.8s
interleave: false
  iter  0: 119.37 ms  147.1 GB/s
  iter  5: 117.75 ms  149.1 GB/s
  iter 10: 115.57 ms  151.9 GB/s
  iter 15: 115.53 ms  151.9 GB/s
  iter 20: 118.19 ms  148.5 GB/s
  iter 25: 114.67 ms  153.1 GB/s
  iter 29: 115.62 ms  151.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.25 ms
achieved bandwidth       : 149.7 GB/s
projected decode         : 8.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T092402Z.json
[round 359] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T092402Z.json
```
