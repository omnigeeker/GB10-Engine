# Round 381 — 20260928T124746Z

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
[round 381] batch-parity OK
[round 381] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 381] prefix cache OK
[round 381] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.761       1.358   10.14x     1.96e-6   (117 GB/s serial, 1186 GB/s warp)

decode-bench: OK
[round 381] decode-bench OK
[round 381] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 57.7s
interleave: false
  iter  0: 119.42 ms  147.0 GB/s
  iter  5: 120.16 ms  146.1 GB/s
  iter 10: 116.97 ms  150.1 GB/s
  iter 15: 114.24 ms  153.7 GB/s
  iter 20: 115.13 ms  152.5 GB/s
  iter 25: 114.89 ms  152.8 GB/s
  iter 29: 115.51 ms  152.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.29 ms
achieved bandwidth       : 149.7 GB/s
projected decode         : 8.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T124746Z.json
[round 381] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T124746Z.json
```
