# Round 321 — 20260927T231856Z

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
[round 321] batch-parity OK
[round 321] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 321] prefix cache OK
[round 321] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.098       1.423   10.61x     2.00e-6   (107 GB/s serial, 1132 GB/s warp)

decode-bench: OK
[round 321] decode-bench OK
[round 321] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.5s
interleave: false
  iter  0: 98.66 ms  177.9 GB/s
  iter  5: 99.15 ms  177.1 GB/s
  iter 10: 98.34 ms  178.5 GB/s
  iter 15: 99.62 ms  176.2 GB/s
  iter 20: 97.93 ms  179.3 GB/s
  iter 25: 97.52 ms  180.0 GB/s
  iter 29: 97.81 ms  179.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.96 ms
achieved bandwidth       : 179.2 GB/s
projected decode         : 10.21 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T231856Z.json
[round 321] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T231856Z.json
```
