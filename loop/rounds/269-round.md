# Round 269 — 20260927T153625Z

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
[round 269] batch-parity OK
[round 269] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 269] prefix cache OK
[round 269] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.489       1.326   11.68x     2.00e-6   (104 GB/s serial, 1215 GB/s warp)

decode-bench: OK
[round 269] decode-bench OK
[round 269] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.2s
interleave: false
  iter  0: 96.01 ms  182.9 GB/s
  iter  5: 97.30 ms  180.4 GB/s
  iter 10: 96.79 ms  181.4 GB/s
  iter 15: 95.41 ms  184.0 GB/s
  iter 20: 95.89 ms  183.1 GB/s
  iter 25: 95.04 ms  184.7 GB/s
  iter 29: 96.66 ms  181.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.20 ms
achieved bandwidth       : 182.5 GB/s
projected decode         : 10.40 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T153625Z.json
[round 269] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T153625Z.json
```
