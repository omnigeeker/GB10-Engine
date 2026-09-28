# Round 327 — 20260928T000938Z

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
[round 327] batch-parity OK
[round 327] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 327] prefix cache OK
[round 327] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.422       1.756    8.78x     2.00e-6   (104 GB/s serial, 917 GB/s warp)

decode-bench: OK
[round 327] decode-bench OK
[round 327] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.3s
interleave: false
  iter  0: 96.91 ms  181.1 GB/s
  iter  5: 95.76 ms  183.3 GB/s
  iter 10: 97.32 ms  180.4 GB/s
  iter 15: 95.96 ms  182.9 GB/s
  iter 20: 96.40 ms  182.1 GB/s
  iter 25: 97.67 ms  179.7 GB/s
  iter 29: 98.32 ms  178.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.80 ms
achieved bandwidth       : 181.4 GB/s
projected decode         : 10.33 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T000938Z.json
[round 327] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T000938Z.json
```
