# Round 255 — 20260927T095149Z

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
[round 255] batch-parity OK
[round 255] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 255] prefix cache OK
[round 255] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.434       1.577    5.98x     2.07e-6   (171 GB/s serial, 1021 GB/s warp)

decode-bench: OK
[round 255] decode-bench OK
[round 255] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 38.1s
interleave: false
  iter  0: 99.99 ms  175.6 GB/s
  iter  5: 96.74 ms  181.5 GB/s
  iter 10: 97.19 ms  180.6 GB/s
  iter 15: 96.67 ms  181.6 GB/s
  iter 20: 96.03 ms  182.8 GB/s
  iter 25: 96.72 ms  181.5 GB/s
  iter 29: 96.33 ms  182.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.37 ms
achieved bandwidth       : 180.3 GB/s
projected decode         : 10.27 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T095149Z.json
[round 255] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T095149Z.json
```
