# Round 277 — 20260927T165036Z

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
[round 277] batch-parity OK
[round 277] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 277] prefix cache OK
[round 277] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.302       1.374   11.13x     2.00e-6   (105 GB/s serial, 1172 GB/s warp)

decode-bench: OK
[round 277] decode-bench OK
[round 277] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 98.05 ms  179.0 GB/s
  iter  5: 96.00 ms  182.9 GB/s
  iter 10: 96.69 ms  181.6 GB/s
  iter 15: 97.70 ms  179.7 GB/s
  iter 20: 97.53 ms  180.0 GB/s
  iter 25: 95.04 ms  184.7 GB/s
  iter 29: 96.93 ms  181.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 95.94 ms
achieved bandwidth       : 183.0 GB/s
projected decode         : 10.42 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T165036Z.json
[round 277] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T165036Z.json
```
