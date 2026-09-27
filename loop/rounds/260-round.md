# Round 260 — 20260927T125546Z

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
[round 260] batch-parity OK
[round 260] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 260] prefix cache OK
[round 260] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.519       1.865    8.32x     2.00e-6   (104 GB/s serial, 863 GB/s warp)

decode-bench: OK
[round 260] decode-bench OK
[round 260] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.4s
interleave: false
  iter  0: 98.54 ms  178.2 GB/s
  iter  5: 97.95 ms  179.2 GB/s
  iter 10: 95.97 ms  182.9 GB/s
  iter 15: 97.01 ms  181.0 GB/s
  iter 20: 98.39 ms  178.4 GB/s
  iter 25: 97.11 ms  180.8 GB/s
  iter 29: 97.75 ms  179.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.07 ms
achieved bandwidth       : 180.8 GB/s
projected decode         : 10.30 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T125546Z.json
[round 260] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T125546Z.json
```
