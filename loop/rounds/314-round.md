# Round 314 — 20260927T221258Z

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
[round 314] batch-parity OK
[round 314] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 314] prefix cache OK
[round 314] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.660       1.379   11.35x     2.00e-6   (103 GB/s serial, 1168 GB/s warp)

decode-bench: OK
[round 314] decode-bench OK
[round 314] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 96.87 ms  181.2 GB/s
  iter  5: 96.46 ms  182.0 GB/s
  iter 10: 96.33 ms  182.2 GB/s
  iter 15: 98.94 ms  177.4 GB/s
  iter 20: 94.67 ms  185.4 GB/s
  iter 25: 100.60 ms  174.5 GB/s
  iter 29: 98.82 ms  177.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.63 ms
achieved bandwidth       : 179.8 GB/s
projected decode         : 10.24 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T221258Z.json
[round 314] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T221258Z.json
```
