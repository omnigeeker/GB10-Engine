# Round 358 — 20260928T091510Z

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
[round 358] batch-parity OK
[round 358] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 358] prefix cache OK
[round 358] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.676       0.925   14.78x     1.96e-6   (118 GB/s serial, 1741 GB/s warp)

decode-bench: OK
[round 358] decode-bench OK
[round 358] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.9s
interleave: false
  iter  0: 115.36 ms  152.2 GB/s
  iter  5: 119.29 ms  147.2 GB/s
  iter 10: 121.04 ms  145.0 GB/s
  iter 15: 118.99 ms  147.5 GB/s
  iter 20: 119.71 ms  146.7 GB/s
  iter 25: 117.45 ms  149.5 GB/s
  iter 29: 113.62 ms  154.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.64 ms
achieved bandwidth       : 149.2 GB/s
projected decode         : 8.50 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T091510Z.json
[round 358] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T091510Z.json
```
