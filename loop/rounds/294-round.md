# Round 294 — 20260927T191252Z

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
[round 294] batch-parity OK
[round 294] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 294] prefix cache OK
[round 294] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      14.962       1.398   10.71x     2.00e-6   (108 GB/s serial, 1152 GB/s warp)

decode-bench: OK
[round 294] decode-bench OK
[round 294] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 34.9s
interleave: false
  iter  0: 96.28 ms  182.3 GB/s
  iter  5: 96.61 ms  181.7 GB/s
  iter 10: 95.85 ms  183.2 GB/s
  iter 15: 95.04 ms  184.7 GB/s
  iter 20: 96.23 ms  182.4 GB/s
  iter 25: 97.01 ms  181.0 GB/s
  iter 29: 97.47 ms  180.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.05 ms
achieved bandwidth       : 182.8 GB/s
projected decode         : 10.41 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T191252Z.json
[round 294] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T191252Z.json
```
