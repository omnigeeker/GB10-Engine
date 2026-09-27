# Round 320 — 20260927T231004Z

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
[round 320] batch-parity OK
[round 320] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 320] prefix cache OK
[round 320] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.533       1.367   11.36x     2.00e-6   (104 GB/s serial, 1178 GB/s warp)

decode-bench: OK
[round 320] decode-bench OK
[round 320] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 99.38 ms  176.6 GB/s
  iter  5: 98.27 ms  178.7 GB/s
  iter 10: 98.07 ms  179.0 GB/s
  iter 15: 99.69 ms  176.1 GB/s
  iter 20: 98.75 ms  177.8 GB/s
  iter 25: 97.91 ms  179.3 GB/s
  iter 29: 97.54 ms  180.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.31 ms
achieved bandwidth       : 178.6 GB/s
projected decode         : 10.17 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T231004Z.json
[round 320] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T231004Z.json
```
