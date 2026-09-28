# Round 379 — 20260928T123054Z

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
[round 379] batch-parity OK
[round 379] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 379] prefix cache OK
[round 379] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.880       1.778    7.24x     1.96e-6   (125 GB/s serial, 906 GB/s warp)

decode-bench: OK
[round 379] decode-bench OK
[round 379] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.9s
interleave: false
  iter  0: 118.79 ms  147.8 GB/s
  iter  5: 118.48 ms  148.2 GB/s
  iter 10: 123.68 ms  141.9 GB/s
  iter 15: 117.34 ms  149.6 GB/s
  iter 20: 115.76 ms  151.7 GB/s
  iter 25: 119.18 ms  147.3 GB/s
  iter 29: 118.01 ms  148.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 118.30 ms
achieved bandwidth       : 148.4 GB/s
projected decode         : 8.45 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T123054Z.json
[round 379] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T123054Z.json
```
