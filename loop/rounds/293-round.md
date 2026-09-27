# Round 293 — 20260927T190654Z

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
[round 293] batch-parity OK
[round 293] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 293] prefix cache OK
[round 293] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.400       1.389    6.77x     2.00e-6   (171 GB/s serial, 1159 GB/s warp)

decode-bench: OK
[round 293] decode-bench OK
[round 293] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.8s
interleave: false
  iter  0: 95.84 ms  183.2 GB/s
  iter  5: 95.12 ms  184.6 GB/s
  iter 10: 96.79 ms  181.4 GB/s
  iter 15: 97.31 ms  180.4 GB/s
  iter 20: 95.24 ms  184.3 GB/s
  iter 25: 97.20 ms  180.6 GB/s
  iter 29: 97.74 ms  179.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.72 ms
achieved bandwidth       : 181.5 GB/s
projected decode         : 10.34 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T190654Z.json
[round 293] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T190654Z.json
```
