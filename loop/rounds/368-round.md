# Round 368 — 20260928T104325Z

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
[round 368] batch-parity OK
[round 368] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 368] prefix cache OK
[round 368] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      14.014       0.893   15.70x     1.96e-6   (115 GB/s serial, 1804 GB/s warp)

decode-bench: OK
[round 368] decode-bench OK
[round 368] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.5s
interleave: false
  iter  0: 117.12 ms  149.9 GB/s
  iter  5: 117.29 ms  149.7 GB/s
  iter 10: 116.68 ms  150.5 GB/s
  iter 15: 118.44 ms  148.2 GB/s
  iter 20: 117.41 ms  149.5 GB/s
  iter 25: 115.80 ms  151.6 GB/s
  iter 29: 117.42 ms  149.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.04 ms
achieved bandwidth       : 150.0 GB/s
projected decode         : 8.54 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T104325Z.json
[round 368] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T104325Z.json
```
