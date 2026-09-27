# Round 308 — 20260927T211638Z

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
[round 308] batch-parity OK
[round 308] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 308] prefix cache OK
[round 308] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.439       1.772    8.71x     2.00e-6   (104 GB/s serial, 909 GB/s warp)

decode-bench: OK
[round 308] decode-bench OK
[round 308] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.1s
interleave: false
  iter  0: 99.86 ms  175.8 GB/s
  iter  5: 99.84 ms  175.8 GB/s
  iter 10: 100.30 ms  175.0 GB/s
  iter 15: 97.29 ms  180.4 GB/s
  iter 20: 99.87 ms  175.8 GB/s
  iter 25: 98.54 ms  178.2 GB/s
  iter 29: 98.66 ms  177.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.62 ms
achieved bandwidth       : 178.0 GB/s
projected decode         : 10.14 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T211638Z.json
[round 308] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T211638Z.json
```
