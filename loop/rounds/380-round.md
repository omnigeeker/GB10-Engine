# Round 380 — 20260928T123918Z

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
[round 380] batch-parity OK
[round 380] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 380] prefix cache OK
[round 380] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.682       1.562    8.76x     1.96e-6   (118 GB/s serial, 1031 GB/s warp)

decode-bench: OK
[round 380] decode-bench OK
[round 380] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.9s
interleave: false
  iter  0: 117.57 ms  149.3 GB/s
  iter  5: 121.79 ms  144.1 GB/s
  iter 10: 118.18 ms  148.6 GB/s
  iter 15: 116.02 ms  151.3 GB/s
  iter 20: 118.02 ms  148.7 GB/s
  iter 25: 117.15 ms  149.8 GB/s
  iter 29: 116.16 ms  151.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 118.06 ms
achieved bandwidth       : 148.7 GB/s
projected decode         : 8.47 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T123918Z.json
[round 380] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T123918Z.json
```
