# Round 276 — 20260927T164300Z

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
[round 276] batch-parity OK
[round 276] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 276] prefix cache OK
[round 276] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.163       1.350   11.24x     2.00e-6   (106 GB/s serial, 1193 GB/s warp)

decode-bench: OK
[round 276] decode-bench OK
[round 276] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.3s
interleave: false
  iter  0: 99.16 ms  177.0 GB/s
  iter  5: 100.86 ms  174.1 GB/s
  iter 10: 98.01 ms  179.1 GB/s
  iter 15: 97.22 ms  180.6 GB/s
  iter 20: 95.98 ms  182.9 GB/s
  iter 25: 98.94 ms  177.4 GB/s
  iter 29: 98.83 ms  177.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.30 ms
achieved bandwidth       : 178.6 GB/s
projected decode         : 10.17 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T164300Z.json
[round 276] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T164300Z.json
```
