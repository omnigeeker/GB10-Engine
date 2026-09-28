# Round 330 — 20260928T003915Z

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
[round 330] batch-parity OK
[round 330] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 330] prefix cache OK
[round 330] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      14.224       1.354   10.51x     2.00e-6   (113 GB/s serial, 1190 GB/s warp)

decode-bench: OK
[round 330] decode-bench OK
[round 330] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.7s
interleave: false
  iter  0: 96.06 ms  182.7 GB/s
  iter  5: 97.06 ms  180.9 GB/s
  iter 10: 96.69 ms  181.6 GB/s
  iter 15: 95.99 ms  182.9 GB/s
  iter 20: 97.07 ms  180.9 GB/s
  iter 25: 99.00 ms  177.3 GB/s
  iter 29: 97.11 ms  180.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.98 ms
achieved bandwidth       : 181.0 GB/s
projected decode         : 10.31 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T003915Z.json
[round 330] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T003915Z.json
```
