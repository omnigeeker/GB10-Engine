# Round 361 — 20260928T094058Z

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
[round 361] batch-parity OK
[round 361] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 361] prefix cache OK
[round 361] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.326       0.929   14.34x     1.96e-6   (121 GB/s serial, 1734 GB/s warp)

decode-bench: OK
[round 361] decode-bench OK
[round 361] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.1s
interleave: false
  iter  0: 117.11 ms  149.9 GB/s
  iter  5: 119.04 ms  147.5 GB/s
  iter 10: 113.06 ms  155.3 GB/s
  iter 15: 113.69 ms  154.4 GB/s
  iter 20: 119.02 ms  147.5 GB/s
  iter 25: 117.50 ms  149.4 GB/s
  iter 29: 120.53 ms  145.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.89 ms
achieved bandwidth       : 150.2 GB/s
projected decode         : 8.55 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T094058Z.json
[round 361] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T094058Z.json
```
