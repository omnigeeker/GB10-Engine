# Round 362 — 20260928T094934Z

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
[round 362] batch-parity OK
[round 362] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 362] prefix cache OK
[round 362] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.637       1.028   13.26x     1.96e-6   (118 GB/s serial, 1566 GB/s warp)

decode-bench: OK
[round 362] decode-bench OK
[round 362] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.9s
interleave: false
  iter  0: 119.54 ms  146.9 GB/s
  iter  5: 118.86 ms  147.7 GB/s
  iter 10: 118.86 ms  147.7 GB/s
  iter 15: 115.90 ms  151.5 GB/s
  iter 20: 114.48 ms  153.3 GB/s
  iter 25: 119.74 ms  146.6 GB/s
  iter 29: 114.39 ms  153.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.28 ms
achieved bandwidth       : 149.7 GB/s
projected decode         : 8.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T094934Z.json
[round 362] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T094934Z.json
```
