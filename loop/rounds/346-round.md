# Round 346 — 20260928T050101Z

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
[round 346] batch-parity OK
[round 346] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 346] prefix cache OK
[round 346] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.031       1.024   12.72x     1.96e-6   (124 GB/s serial, 1572 GB/s warp)

decode-bench: OK
[round 346] decode-bench OK
[round 346] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.4s
interleave: false
  iter  0: 119.09 ms  147.4 GB/s
  iter  5: 118.04 ms  148.7 GB/s
  iter 10: 117.71 ms  149.1 GB/s
  iter 15: 117.76 ms  149.1 GB/s
  iter 20: 115.18 ms  152.4 GB/s
  iter 25: 118.65 ms  148.0 GB/s
  iter 29: 121.34 ms  144.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.29 ms
achieved bandwidth       : 149.7 GB/s
projected decode         : 8.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T050101Z.json
[round 346] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T050101Z.json
```
