# Round 329 — 20260928T002924Z

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
[round 329] batch-parity OK
[round 329] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 329] prefix cache OK
[round 329] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.565       1.761    8.84x     2.00e-6   (103 GB/s serial, 915 GB/s warp)

decode-bench: OK
[round 329] decode-bench OK
[round 329] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.5s
interleave: false
  iter  0: 98.41 ms  178.4 GB/s
  iter  5: 99.33 ms  176.7 GB/s
  iter 10: 97.18 ms  180.6 GB/s
  iter 15: 97.65 ms  179.8 GB/s
  iter 20: 94.13 ms  186.5 GB/s
  iter 25: 95.14 ms  184.5 GB/s
  iter 29: 95.44 ms  183.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.64 ms
achieved bandwidth       : 181.7 GB/s
projected decode         : 10.35 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T002924Z.json
[round 329] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T002924Z.json
```
