# Round 369 — 20260928T105201Z

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
[round 369] batch-parity OK
[round 369] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 369] prefix cache OK
[round 369] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.068       0.989   13.22x     1.96e-6   (123 GB/s serial, 1629 GB/s warp)

decode-bench: OK
[round 369] decode-bench OK
[round 369] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.5s
interleave: false
  iter  0: 120.08 ms  146.2 GB/s
  iter  5: 118.23 ms  148.5 GB/s
  iter 10: 119.08 ms  147.4 GB/s
  iter 15: 117.49 ms  149.4 GB/s
  iter 20: 113.41 ms  154.8 GB/s
  iter 25: 118.09 ms  148.7 GB/s
  iter 29: 119.16 ms  147.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.19 ms
achieved bandwidth       : 149.8 GB/s
projected decode         : 8.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T105201Z.json
[round 369] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T105201Z.json
```
