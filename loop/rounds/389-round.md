# Round 389 — 20260928T140148Z

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
[round 389] batch-parity OK
[round 389] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 389] prefix cache OK
[round 389] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.485       1.769    7.62x     1.96e-6   (119 GB/s serial, 910 GB/s warp)

decode-bench: OK
[round 389] decode-bench OK
[round 389] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.6s
interleave: false
  iter  0: 118.67 ms  147.9 GB/s
  iter  5: 114.65 ms  153.1 GB/s
  iter 10: 115.85 ms  151.5 GB/s
  iter 15: 115.12 ms  152.5 GB/s
  iter 20: 115.03 ms  152.6 GB/s
  iter 25: 116.51 ms  150.7 GB/s
  iter 29: 114.00 ms  154.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.57 ms
achieved bandwidth       : 150.6 GB/s
projected decode         : 8.58 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T140148Z.json
[round 389] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T140148Z.json
```
