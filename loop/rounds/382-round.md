# Round 382 — 20260928T125606Z

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
[round 382] batch-parity OK
[round 382] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 382] prefix cache OK
[round 382] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.102       1.772    7.39x     1.96e-6   (123 GB/s serial, 909 GB/s warp)

decode-bench: OK
[round 382] decode-bench OK
[round 382] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.7s
interleave: false
  iter  0: 119.66 ms  146.7 GB/s
  iter  5: 113.36 ms  154.9 GB/s
  iter 10: 117.73 ms  149.1 GB/s
  iter 15: 114.61 ms  153.2 GB/s
  iter 20: 117.35 ms  149.6 GB/s
  iter 25: 116.18 ms  151.1 GB/s
  iter 29: 119.05 ms  147.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.78 ms
achieved bandwidth       : 150.3 GB/s
projected decode         : 8.56 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T125606Z.json
[round 382] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T125606Z.json
```
