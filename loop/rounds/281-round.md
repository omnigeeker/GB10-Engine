# Round 281 — 20260927T172808Z

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
[round 281] batch-parity OK
[round 281] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 281] prefix cache OK
[round 281] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.526       1.400    6.81x     2.00e-6   (169 GB/s serial, 1151 GB/s warp)

decode-bench: OK
[round 281] decode-bench OK
[round 281] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.9s
interleave: false
  iter  0: 95.07 ms  184.7 GB/s
  iter  5: 95.70 ms  183.4 GB/s
  iter 10: 96.60 ms  181.7 GB/s
  iter 15: 97.31 ms  180.4 GB/s
  iter 20: 95.95 ms  183.0 GB/s
  iter 25: 95.87 ms  183.1 GB/s
  iter 29: 96.02 ms  182.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.37 ms
achieved bandwidth       : 182.2 GB/s
projected decode         : 10.38 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T172808Z.json
[round 281] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T172808Z.json
```
