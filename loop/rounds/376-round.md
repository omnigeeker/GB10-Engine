# Round 376 — 20260928T115727Z

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
[round 376] batch-parity OK
[round 376] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 376] prefix cache OK
[round 376] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.434       0.922   13.49x     1.96e-6   (130 GB/s serial, 1748 GB/s warp)

decode-bench: OK
[round 376] decode-bench OK
[round 376] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 56.4s
interleave: false
  iter  0: 119.46 ms  147.0 GB/s
  iter  5: 117.47 ms  149.4 GB/s
  iter 10: 117.77 ms  149.1 GB/s
  iter 15: 122.86 ms  142.9 GB/s
  iter 20: 119.41 ms  147.0 GB/s
  iter 25: 118.44 ms  148.2 GB/s
  iter 29: 121.10 ms  145.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 118.25 ms
achieved bandwidth       : 148.5 GB/s
projected decode         : 8.46 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T115727Z.json
[round 376] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T115727Z.json
```
