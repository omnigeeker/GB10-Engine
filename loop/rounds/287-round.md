# Round 287 — 20260927T182342Z

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
[round 287] batch-parity OK
[round 287] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 287] prefix cache OK
[round 287] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      14.877       1.376   10.81x     2.00e-6   (108 GB/s serial, 1171 GB/s warp)

decode-bench: OK
[round 287] decode-bench OK
[round 287] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.1s
interleave: false
  iter  0: 96.51 ms  181.9 GB/s
  iter  5: 96.81 ms  181.3 GB/s
  iter 10: 97.20 ms  180.6 GB/s
  iter 15: 98.78 ms  177.7 GB/s
  iter 20: 96.90 ms  181.2 GB/s
  iter 25: 97.86 ms  179.4 GB/s
  iter 29: 97.68 ms  179.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.52 ms
achieved bandwidth       : 181.9 GB/s
projected decode         : 10.36 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T182342Z.json
[round 287] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T182342Z.json
```
