# Round 257 — 20260927T114255Z

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
[round 257] batch-parity OK
[round 257] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 257] prefix cache OK
[round 257] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.683       1.611    9.73x     2.07e-6   (103 GB/s serial, 1000 GB/s warp)

decode-bench: OK
[round 257] decode-bench OK
[round 257] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 98.10 ms  179.0 GB/s
  iter  5: 98.56 ms  178.1 GB/s
  iter 10: 110.88 ms  158.3 GB/s
  iter 15: 99.75 ms  176.0 GB/s
  iter 20: 96.75 ms  181.5 GB/s
  iter 25: 96.21 ms  182.5 GB/s
  iter 29: 98.24 ms  178.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 100.42 ms
achieved bandwidth       : 174.8 GB/s
projected decode         : 9.96 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 76.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T114255Z.json
[round 257] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T114255Z.json
```
