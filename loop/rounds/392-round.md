# Round 392 — 20260928T142950Z

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
[round 392] batch-parity OK
[round 392] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 392] prefix cache OK
[round 392] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.449       1.242    7.61x     1.96e-6   (170 GB/s serial, 1297 GB/s warp)

decode-bench: OK
[round 392] decode-bench OK
[round 392] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.9s
interleave: false
  iter  0: 98.97 ms  177.4 GB/s
  iter  5: 97.32 ms  180.4 GB/s
  iter 10: 95.71 ms  183.4 GB/s
  iter 15: 97.55 ms  180.0 GB/s
  iter 20: 97.09 ms  180.8 GB/s
  iter 25: 95.74 ms  183.4 GB/s
  iter 29: 98.20 ms  178.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.90 ms
achieved bandwidth       : 179.3 GB/s
projected decode         : 10.21 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T142950Z.json
[round 392] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T142950Z.json
```
