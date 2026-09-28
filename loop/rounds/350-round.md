# Round 350 — 20260928T054711Z

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
[round 350] batch-parity OK
[round 350] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 350] prefix cache OK
[round 350] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.739       1.396    9.13x     2.00e-6   (126 GB/s serial, 1154 GB/s warp)

decode-bench: OK
[round 350] decode-bench OK
[round 350] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.3s
interleave: false
  iter  0: 116.55 ms  150.6 GB/s
  iter  5: 118.66 ms  148.0 GB/s
  iter 10: 113.75 ms  154.3 GB/s
  iter 15: 115.01 ms  152.6 GB/s
  iter 20: 120.27 ms  146.0 GB/s
  iter 25: 115.45 ms  152.1 GB/s
  iter 29: 115.39 ms  152.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.69 ms
achieved bandwidth       : 150.4 GB/s
projected decode         : 8.57 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T054711Z.json
[round 350] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T054711Z.json
```
