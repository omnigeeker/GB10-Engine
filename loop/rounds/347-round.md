# Round 347 — 20260928T050855Z

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
[round 347] batch-parity OK
[round 347] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 347] prefix cache OK
[round 347] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.364       0.997   13.41x     1.96e-6   (121 GB/s serial, 1616 GB/s warp)

decode-bench: OK
[round 347] decode-bench OK
[round 347] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.9s
interleave: false
  iter  0: 121.11 ms  145.0 GB/s
  iter  5: 119.26 ms  147.2 GB/s
  iter 10: 123.15 ms  142.6 GB/s
  iter 15: 117.62 ms  149.3 GB/s
  iter 20: 117.22 ms  149.8 GB/s
  iter 25: 115.14 ms  152.5 GB/s
  iter 29: 117.67 ms  149.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.07 ms
achieved bandwidth       : 150.0 GB/s
projected decode         : 8.54 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T050855Z.json
[round 347] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T050855Z.json
```
