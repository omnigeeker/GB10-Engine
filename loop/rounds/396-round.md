# Round 396 — 20260928T150917Z

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
[round 396] batch-parity OK
[round 396] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 396] prefix cache OK
[round 396] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.588       1.264    9.96x     1.96e-6   (128 GB/s serial, 1274 GB/s warp)

decode-bench: OK
[round 396] decode-bench OK
[round 396] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.8s
interleave: false
  iter  0: 99.37 ms  176.7 GB/s
  iter  5: 98.68 ms  177.9 GB/s
  iter 10: 99.26 ms  176.9 GB/s
  iter 15: 98.52 ms  178.2 GB/s
  iter 20: 97.77 ms  179.6 GB/s
  iter 25: 98.39 ms  178.4 GB/s
  iter 29: 98.54 ms  178.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.31 ms
achieved bandwidth       : 178.6 GB/s
projected decode         : 10.17 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T150917Z.json
[round 396] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T150917Z.json
```
