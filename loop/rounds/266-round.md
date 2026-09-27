# Round 266 — 20260927T145944Z

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
[round 266] batch-parity OK
[round 266] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 266] prefix cache OK
[round 266] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.203       1.357   11.20x     2.00e-6   (106 GB/s serial, 1187 GB/s warp)

decode-bench: OK
[round 266] decode-bench OK
[round 266] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.7s
interleave: false
  iter  0: 95.80 ms  183.3 GB/s
  iter  5: 94.95 ms  184.9 GB/s
  iter 10: 94.52 ms  185.7 GB/s
  iter 15: 94.47 ms  185.8 GB/s
  iter 20: 94.70 ms  185.4 GB/s
  iter 25: 95.05 ms  184.7 GB/s
  iter 29: 94.92 ms  185.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 94.94 ms
achieved bandwidth       : 184.9 GB/s
projected decode         : 10.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 81.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T145944Z.json
[round 266] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T145944Z.json
```
