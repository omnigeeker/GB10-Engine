# Round 275 — 20260927T163428Z

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
[round 275] batch-parity OK
[round 275] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 275] prefix cache OK
[round 275] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.523       1.717    9.04x     2.00e-6   (104 GB/s serial, 938 GB/s warp)

decode-bench: OK
[round 275] decode-bench OK
[round 275] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 96.98 ms  181.0 GB/s
  iter  5: 96.52 ms  181.9 GB/s
  iter 10: 96.65 ms  181.6 GB/s
  iter 15: 100.93 ms  173.9 GB/s
  iter 20: 97.83 ms  179.4 GB/s
  iter 25: 97.07 ms  180.8 GB/s
  iter 29: 97.39 ms  180.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.52 ms
achieved bandwidth       : 180.0 GB/s
projected decode         : 10.25 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T163428Z.json
[round 275] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T163428Z.json
```
