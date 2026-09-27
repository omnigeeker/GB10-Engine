# Round 299 — 20260927T195210Z

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
[round 299] batch-parity OK
[round 299] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 299] prefix cache OK
[round 299] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.669       1.327   11.80x     2.00e-6   (103 GB/s serial, 1213 GB/s warp)

decode-bench: OK
[round 299] decode-bench OK
[round 299] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.9s
interleave: false
  iter  0: 99.48 ms  176.5 GB/s
  iter  5: 98.35 ms  178.5 GB/s
  iter 10: 97.61 ms  179.8 GB/s
  iter 15: 97.11 ms  180.8 GB/s
  iter 20: 97.32 ms  180.4 GB/s
  iter 25: 96.20 ms  182.5 GB/s
  iter 29: 96.07 ms  182.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.20 ms
achieved bandwidth       : 180.6 GB/s
projected decode         : 10.29 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T195210Z.json
[round 299] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T195210Z.json
```
