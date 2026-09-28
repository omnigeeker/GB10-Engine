# Round 338 — 20260928T021847Z

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
[round 338] batch-parity OK
[round 338] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 338] prefix cache OK
[round 338] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.032       1.661    7.84x     2.00e-6   (124 GB/s serial, 970 GB/s warp)

decode-bench: OK
[round 338] decode-bench OK
[round 338] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.4s
interleave: false
  iter  0: 119.31 ms  147.1 GB/s
  iter  5: 114.28 ms  153.6 GB/s
  iter 10: 117.11 ms  149.9 GB/s
  iter 15: 116.52 ms  150.7 GB/s
  iter 20: 116.37 ms  150.9 GB/s
  iter 25: 116.88 ms  150.2 GB/s
  iter 29: 115.11 ms  152.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.21 ms
achieved bandwidth       : 151.1 GB/s
projected decode         : 8.60 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T021847Z.json
[round 338] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T021847Z.json
```
