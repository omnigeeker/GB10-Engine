# Round 267 — 20260927T152100Z

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
[round 267] batch-parity OK
[round 267] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 267] prefix cache OK
[round 267] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.303       1.315   11.64x     2.00e-6   (105 GB/s serial, 1225 GB/s warp)

decode-bench: OK
[round 267] decode-bench OK
[round 267] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.8s
interleave: false
  iter  0: 97.82 ms  179.5 GB/s
  iter  5: 100.17 ms  175.2 GB/s
  iter 10: 98.24 ms  178.7 GB/s
  iter 15: 98.58 ms  178.1 GB/s
  iter 20: 97.93 ms  179.3 GB/s
  iter 25: 99.41 ms  176.6 GB/s
  iter 29: 95.54 ms  183.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.78 ms
achieved bandwidth       : 179.5 GB/s
projected decode         : 10.23 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T152100Z.json
[round 267] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T152100Z.json
```
