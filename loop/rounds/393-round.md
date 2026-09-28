# Round 393 — 20260928T143832Z

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
[round 393] batch-parity OK
[round 393] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 393] prefix cache OK
[round 393] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.297       1.214    7.66x     1.96e-6   (173 GB/s serial, 1327 GB/s warp)

decode-bench: OK
[round 393] decode-bench OK
[round 393] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.2s
interleave: false
  iter  0: 99.11 ms  177.1 GB/s
  iter  5: 100.14 ms  175.3 GB/s
  iter 10: 99.48 ms  176.5 GB/s
  iter 15: 99.02 ms  177.3 GB/s
  iter 20: 97.14 ms  180.7 GB/s
  iter 25: 96.94 ms  181.1 GB/s
  iter 29: 99.52 ms  176.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.65 ms
achieved bandwidth       : 178.0 GB/s
projected decode         : 10.14 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T143832Z.json
[round 393] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T143832Z.json
```
