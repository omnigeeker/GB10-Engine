# Round 339 — 20260928T022706Z

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
[round 339] batch-parity OK
[round 339] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 339] prefix cache OK
[round 339] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.834       1.493    8.60x     2.00e-6   (125 GB/s serial, 1079 GB/s warp)

decode-bench: OK
[round 339] decode-bench OK
[round 339] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 57.8s
interleave: false
  iter  0: 114.26 ms  153.6 GB/s
  iter  5: 117.45 ms  149.5 GB/s
  iter 10: 115.72 ms  151.7 GB/s
  iter 15: 116.31 ms  150.9 GB/s
  iter 20: 112.15 ms  156.5 GB/s
  iter 25: 111.04 ms  158.1 GB/s
  iter 29: 115.47 ms  152.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 115.06 ms
achieved bandwidth       : 152.6 GB/s
projected decode         : 8.69 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T022706Z.json
[round 339] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T022706Z.json
```
