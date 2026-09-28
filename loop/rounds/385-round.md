# Round 385 — 20260928T132231Z

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
[round 385] batch-parity OK
[round 385] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 385] prefix cache OK
[round 385] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.665       1.670    8.19x     1.96e-6   (118 GB/s serial, 965 GB/s warp)

decode-bench: OK
[round 385] decode-bench OK
[round 385] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.4s
interleave: false
  iter  0: 117.01 ms  150.0 GB/s
  iter  5: 119.20 ms  147.3 GB/s
  iter 10: 117.50 ms  149.4 GB/s
  iter 15: 115.06 ms  152.6 GB/s
  iter 20: 118.95 ms  147.6 GB/s
  iter 25: 120.45 ms  145.8 GB/s
  iter 29: 116.33 ms  150.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.34 ms
achieved bandwidth       : 149.6 GB/s
projected decode         : 8.52 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T132231Z.json
[round 385] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T132231Z.json
```
