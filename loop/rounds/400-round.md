# Round 400 — 20260928T170435Z

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
[round 400] batch-parity OK
[round 400] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 400] prefix cache OK
[round 400] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.260       1.200   10.22x     1.96e-6   (131 GB/s serial, 1342 GB/s warp)

decode-bench: OK
[round 400] decode-bench OK
[round 400] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.5s
interleave: false
  iter  0: 98.11 ms  178.9 GB/s
  iter  5: 95.38 ms  184.1 GB/s
  iter 10: 98.25 ms  178.7 GB/s
  iter 15: 95.98 ms  182.9 GB/s
  iter 20: 96.40 ms  182.1 GB/s
  iter 25: 97.06 ms  180.9 GB/s
  iter 29: 96.66 ms  181.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.83 ms
achieved bandwidth       : 181.3 GB/s
projected decode         : 10.33 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T170435Z.json
[round 400] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T170435Z.json
```
