# Round 375 — 20260928T114856Z

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
[round 375] batch-parity OK
[round 375] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 375] prefix cache OK
[round 375] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.556       1.031   13.15x     1.96e-6   (119 GB/s serial, 1562 GB/s warp)

decode-bench: OK
[round 375] decode-bench OK
[round 375] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.5s
interleave: false
  iter  0: 117.52 ms  149.4 GB/s
  iter  5: 118.83 ms  147.7 GB/s
  iter 10: 117.54 ms  149.4 GB/s
  iter 15: 114.51 ms  153.3 GB/s
  iter 20: 118.03 ms  148.7 GB/s
  iter 25: 117.01 ms  150.0 GB/s
  iter 29: 117.83 ms  149.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.74 ms
achieved bandwidth       : 149.1 GB/s
projected decode         : 8.49 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T114856Z.json
[round 375] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T114856Z.json
```
