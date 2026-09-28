# Round 364 — 20260928T100720Z

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
[round 364] batch-parity OK
[round 364] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 364] prefix cache OK
[round 364] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.842       0.913   14.07x     1.96e-6   (125 GB/s serial, 1765 GB/s warp)

decode-bench: OK
[round 364] decode-bench OK
[round 364] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.1s
interleave: false
  iter  0: 116.89 ms  150.2 GB/s
  iter  5: 113.86 ms  154.2 GB/s
  iter 10: 113.96 ms  154.0 GB/s
  iter 15: 114.30 ms  153.6 GB/s
  iter 20: 118.70 ms  147.9 GB/s
  iter 25: 117.32 ms  149.6 GB/s
  iter 29: 118.47 ms  148.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.60 ms
achieved bandwidth       : 150.6 GB/s
projected decode         : 8.58 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T100720Z.json
[round 364] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T100720Z.json
```
