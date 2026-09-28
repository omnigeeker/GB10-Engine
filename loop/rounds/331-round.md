# Round 331 — 20260928T005138Z

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
[round 331] batch-parity OK
[round 331] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 331] prefix cache OK
[round 331] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.706       1.407   11.16x     2.00e-6   (103 GB/s serial, 1145 GB/s warp)

decode-bench: OK
[round 331] decode-bench OK
[round 331] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.8s
interleave: false
  iter  0: 108.43 ms  161.9 GB/s
  iter  5: 113.88 ms  154.2 GB/s
  iter 10: 112.60 ms  155.9 GB/s
  iter 15: 114.96 ms  152.7 GB/s
  iter 20: 115.07 ms  152.6 GB/s
  iter 25: 105.90 ms  165.8 GB/s
  iter 29: 113.71 ms  154.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 110.56 ms
achieved bandwidth       : 158.8 GB/s
projected decode         : 9.04 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 69.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T005138Z.json
[round 331] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T005138Z.json
```
