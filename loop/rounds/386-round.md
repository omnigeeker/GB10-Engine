# Round 386 — 20260928T133218Z

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
[round 386] batch-parity OK
[round 386] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 386] prefix cache OK
[round 386] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.677       1.363   10.03x     1.96e-6   (118 GB/s serial, 1182 GB/s warp)

decode-bench: OK
[round 386] decode-bench OK
[round 386] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.3s
interleave: false
  iter  0: 116.45 ms  150.7 GB/s
  iter  5: 111.83 ms  157.0 GB/s
  iter 10: 116.60 ms  150.6 GB/s
  iter 15: 121.44 ms  144.6 GB/s
  iter 20: 118.37 ms  148.3 GB/s
  iter 25: 118.01 ms  148.8 GB/s
  iter 29: 116.48 ms  150.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.66 ms
achieved bandwidth       : 150.5 GB/s
projected decode         : 8.57 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T133218Z.json
[round 386] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T133218Z.json
```
