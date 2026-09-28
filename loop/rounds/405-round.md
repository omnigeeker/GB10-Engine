# Round 405 — 20260928T181228Z

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
[round 405] batch-parity OK
[round 405] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 405] prefix cache OK
[round 405] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.529       1.289   10.50x     1.96e-6   (119 GB/s serial, 1250 GB/s warp)

decode-bench: OK
[round 405] decode-bench OK
[round 405] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.8s
interleave: false
  iter  0: 94.42 ms  185.9 GB/s
  iter  5: 95.77 ms  183.3 GB/s
  iter 10: 97.16 ms  180.7 GB/s
  iter 15: 94.14 ms  186.5 GB/s
  iter 20: 95.41 ms  184.0 GB/s
  iter 25: 96.30 ms  182.3 GB/s
  iter 29: 96.16 ms  182.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 95.64 ms
achieved bandwidth       : 183.6 GB/s
projected decode         : 10.46 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T181228Z.json
[round 405] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T181228Z.json
```
