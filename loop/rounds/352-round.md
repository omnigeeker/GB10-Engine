# Round 352 — 20260928T062206Z

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
[round 352] batch-parity OK
[round 352] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 352] prefix cache OK
[round 352] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.813       1.677    7.64x     2.00e-6   (126 GB/s serial, 960 GB/s warp)

decode-bench: OK
[round 352] decode-bench OK
[round 352] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.2s
interleave: false
  iter  0: 118.91 ms  147.6 GB/s
  iter  5: 117.02 ms  150.0 GB/s
  iter 10: 120.95 ms  145.1 GB/s
  iter 15: 117.19 ms  149.8 GB/s
  iter 20: 117.87 ms  148.9 GB/s
  iter 25: 119.06 ms  147.5 GB/s
  iter 29: 118.87 ms  147.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 118.07 ms
achieved bandwidth       : 148.7 GB/s
projected decode         : 8.47 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T062206Z.json
[round 352] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T062206Z.json
```
