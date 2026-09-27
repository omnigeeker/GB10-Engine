# Round 259 — 20260927T123617Z

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
[round 259] batch-parity OK
[round 259] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 259] prefix cache OK
[round 259] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.451       1.705    9.06x     2.00e-6   (104 GB/s serial, 945 GB/s warp)

decode-bench: OK
[round 259] decode-bench OK
[round 259] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.7s
interleave: false
  iter  0: 97.38 ms  180.3 GB/s
  iter  5: 92.93 ms  188.9 GB/s
  iter 10: 96.78 ms  181.4 GB/s
  iter 15: 95.81 ms  183.2 GB/s
  iter 20: 96.14 ms  182.6 GB/s
  iter 25: 95.47 ms  183.9 GB/s
  iter 29: 97.71 ms  179.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 95.87 ms
achieved bandwidth       : 183.1 GB/s
projected decode         : 10.43 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T123617Z.json
[round 259] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T123617Z.json
```
