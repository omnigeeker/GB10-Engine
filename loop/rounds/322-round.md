# Round 322 — 20260927T232759Z

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
[round 322] batch-parity OK
[round 322] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 322] prefix cache OK
[round 322] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.667       1.415    6.83x     2.00e-6   (167 GB/s serial, 1138 GB/s warp)

decode-bench: OK
[round 322] decode-bench OK
[round 322] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.6s
interleave: false
  iter  0: 96.52 ms  181.9 GB/s
  iter  5: 96.77 ms  181.4 GB/s
  iter 10: 96.36 ms  182.2 GB/s
  iter 15: 97.52 ms  180.0 GB/s
  iter 20: 97.83 ms  179.5 GB/s
  iter 25: 96.89 ms  181.2 GB/s
  iter 29: 98.81 ms  177.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.52 ms
achieved bandwidth       : 180.0 GB/s
projected decode         : 10.25 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T232759Z.json
[round 322] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T232759Z.json
```
