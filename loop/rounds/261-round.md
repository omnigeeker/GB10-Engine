# Round 261 — 20260927T130334Z

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
[round 261] batch-parity OK
[round 261] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 261] prefix cache OK
[round 261] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.331       1.481    9.00x     2.00e-6   (121 GB/s serial, 1088 GB/s warp)

decode-bench: OK
[round 261] decode-bench OK
[round 261] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.6s
interleave: false
  iter  0: 96.25 ms  182.4 GB/s
  iter  5: 96.94 ms  181.1 GB/s
  iter 10: 94.48 ms  185.8 GB/s
  iter 15: 96.68 ms  181.6 GB/s
  iter 20: 95.58 ms  183.7 GB/s
  iter 25: 95.84 ms  183.2 GB/s
  iter 29: 98.33 ms  178.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.81 ms
achieved bandwidth       : 181.3 GB/s
projected decode         : 10.33 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T130334Z.json
[round 261] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T130334Z.json
```
