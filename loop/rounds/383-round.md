# Round 383 — 20260928T130457Z

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
[round 383] batch-parity OK
[round 383] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 383] prefix cache OK
[round 383] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      14.323       1.519    9.43x     1.96e-6   (112 GB/s serial, 1060 GB/s warp)

decode-bench: OK
[round 383] decode-bench OK
[round 383] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.6s
interleave: false
  iter  0: 118.28 ms  148.4 GB/s
  iter  5: 116.92 ms  150.1 GB/s
  iter 10: 115.83 ms  151.6 GB/s
  iter 15: 118.65 ms  148.0 GB/s
  iter 20: 119.16 ms  147.3 GB/s
  iter 25: 118.84 ms  147.7 GB/s
  iter 29: 115.57 ms  151.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.19 ms
achieved bandwidth       : 149.8 GB/s
projected decode         : 8.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T130457Z.json
[round 383] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T130457Z.json
```
