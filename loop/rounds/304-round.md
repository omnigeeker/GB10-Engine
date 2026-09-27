# Round 304 — 20260927T203951Z

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
[round 304] batch-parity OK
[round 304] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 304] prefix cache OK
[round 304] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.556       1.360   11.43x     2.00e-6   (104 GB/s serial, 1184 GB/s warp)

decode-bench: OK
[round 304] decode-bench OK
[round 304] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.9s
interleave: false
  iter  0: 97.54 ms  180.0 GB/s
  iter  5: 97.21 ms  180.6 GB/s
  iter 10: 98.67 ms  177.9 GB/s
  iter 15: 96.25 ms  182.4 GB/s
  iter 20: 95.87 ms  183.1 GB/s
  iter 25: 96.85 ms  181.3 GB/s
  iter 29: 98.42 ms  178.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.02 ms
achieved bandwidth       : 180.9 GB/s
projected decode         : 10.31 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T203951Z.json
[round 304] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T203951Z.json
```
