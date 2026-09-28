# Round 340 — 20260928T023652Z

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
[round 340] batch-parity OK
[round 340] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 340] prefix cache OK
[round 340] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      12.870       1.449    8.88x     2.00e-6   (125 GB/s serial, 1112 GB/s warp)

decode-bench: OK
[round 340] decode-bench OK
[round 340] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.6s
interleave: false
  iter  0: 117.72 ms  149.1 GB/s
  iter  5: 119.22 ms  147.3 GB/s
  iter 10: 116.11 ms  151.2 GB/s
  iter 15: 117.32 ms  149.6 GB/s
  iter 20: 114.10 ms  153.9 GB/s
  iter 25: 115.28 ms  152.3 GB/s
  iter 29: 113.50 ms  154.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.31 ms
achieved bandwidth       : 150.9 GB/s
projected decode         : 8.60 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T023652Z.json
[round 340] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T023652Z.json
```
