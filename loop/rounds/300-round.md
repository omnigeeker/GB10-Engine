# Round 300 — 20260927T195818Z

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
[round 300] batch-parity OK
[round 300] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 300] prefix cache OK
[round 300] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.154       1.754    8.64x     2.00e-6   (106 GB/s serial, 918 GB/s warp)

decode-bench: OK
[round 300] decode-bench OK
[round 300] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.9s
interleave: false
  iter  0: 97.60 ms  179.9 GB/s
  iter  5: 97.02 ms  180.9 GB/s
  iter 10: 97.38 ms  180.3 GB/s
  iter 15: 98.56 ms  178.1 GB/s
  iter 20: 97.93 ms  179.3 GB/s
  iter 25: 98.84 ms  177.6 GB/s
  iter 29: 97.70 ms  179.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.62 ms
achieved bandwidth       : 179.8 GB/s
projected decode         : 10.24 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T195818Z.json
[round 300] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T195818Z.json
```
