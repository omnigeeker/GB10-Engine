# Round 377 — 20260928T120835Z

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
[round 377] batch-parity OK
[round 377] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 377] prefix cache OK
[round 377] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.897       1.751    7.36x     1.96e-6   (125 GB/s serial, 920 GB/s warp)

decode-bench: OK
[round 377] decode-bench OK
[round 377] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.4s
interleave: false
  iter  0: 118.50 ms  148.1 GB/s
  iter  5: 117.89 ms  148.9 GB/s
  iter 10: 118.71 ms  147.9 GB/s
  iter 15: 115.75 ms  151.7 GB/s
  iter 20: 117.43 ms  149.5 GB/s
  iter 25: 115.68 ms  151.8 GB/s
  iter 29: 116.50 ms  150.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.11 ms
achieved bandwidth       : 149.9 GB/s
projected decode         : 8.54 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T120835Z.json
[round 377] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T120835Z.json
```
