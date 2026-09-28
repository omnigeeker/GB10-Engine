# Round 366 — 20260928T102357Z

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
[round 366] batch-parity OK
[round 366] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 366] prefix cache OK
[round 366] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.945       1.029   13.55x     1.96e-6   (115 GB/s serial, 1565 GB/s warp)

decode-bench: OK
[round 366] decode-bench OK
[round 366] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.1s
interleave: false
  iter  0: 117.97 ms  148.8 GB/s
  iter  5: 119.15 ms  147.3 GB/s
  iter 10: 118.75 ms  147.8 GB/s
  iter 15: 120.77 ms  145.4 GB/s
  iter 20: 117.37 ms  149.6 GB/s
  iter 25: 115.51 ms  152.0 GB/s
  iter 29: 112.90 ms  155.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.36 ms
achieved bandwidth       : 149.6 GB/s
projected decode         : 8.52 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T102357Z.json
[round 366] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T102357Z.json
```
