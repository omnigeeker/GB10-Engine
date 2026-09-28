# Round 397 — 20260928T151640Z

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
[round 397] batch-parity OK
[round 397] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 397] prefix cache OK
[round 397] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.269       1.221   10.87x     1.96e-6   (121 GB/s serial, 1319 GB/s warp)

decode-bench: OK
[round 397] decode-bench OK
[round 397] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.7s
interleave: false
  iter  0: 95.32 ms  184.2 GB/s
  iter  5: 95.76 ms  183.3 GB/s
  iter 10: 95.71 ms  183.4 GB/s
  iter 15: 98.78 ms  177.7 GB/s
  iter 20: 96.91 ms  181.1 GB/s
  iter 25: 97.30 ms  180.4 GB/s
  iter 29: 96.80 ms  181.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.93 ms
achieved bandwidth       : 181.1 GB/s
projected decode         : 10.32 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T151640Z.json
[round 397] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T151640Z.json
```
