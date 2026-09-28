# Round 344 — 20260928T043606Z

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
[round 344] batch-parity OK
[round 344] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 344] prefix cache OK
[round 344] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.638       1.528    8.27x     2.00e-6   (127 GB/s serial, 1054 GB/s warp)

decode-bench: OK
[round 344] decode-bench OK
[round 344] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.3s
interleave: false
  iter  0: 117.32 ms  149.6 GB/s
  iter  5: 113.95 ms  154.1 GB/s
  iter 10: 114.24 ms  153.7 GB/s
  iter 15: 115.58 ms  151.9 GB/s
  iter 20: 111.79 ms  157.0 GB/s
  iter 25: 114.23 ms  153.7 GB/s
  iter 29: 117.10 ms  149.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 114.67 ms
achieved bandwidth       : 153.1 GB/s
projected decode         : 8.72 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 67.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T043606Z.json
[round 344] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T043606Z.json
```
