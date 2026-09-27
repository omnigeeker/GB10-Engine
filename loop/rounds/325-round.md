# Round 325 — 20260927T235051Z

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
[round 325] batch-parity OK
[round 325] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 325] prefix cache OK
[round 325] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.433       1.365   11.31x     2.00e-6   (104 GB/s serial, 1180 GB/s warp)

decode-bench: OK
[round 325] decode-bench OK
[round 325] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.3s
interleave: false
  iter  0: 97.17 ms  180.7 GB/s
  iter  5: 97.33 ms  180.4 GB/s
  iter 10: 98.46 ms  178.3 GB/s
  iter 15: 99.80 ms  175.9 GB/s
  iter 20: 98.67 ms  177.9 GB/s
  iter 25: 97.19 ms  180.6 GB/s
  iter 29: 98.31 ms  178.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.30 ms
achieved bandwidth       : 178.6 GB/s
projected decode         : 10.17 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T235051Z.json
[round 325] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T235051Z.json
```
