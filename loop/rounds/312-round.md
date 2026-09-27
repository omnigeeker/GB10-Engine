# Round 312 — 20260927T215206Z

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
[round 312] batch-parity OK
[round 312] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 312] prefix cache OK
[round 312] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.504       1.519   10.21x     2.00e-6   (104 GB/s serial, 1060 GB/s warp)

decode-bench: OK
[round 312] decode-bench OK
[round 312] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.4s
interleave: false
  iter  0: 99.81 ms  175.9 GB/s
  iter  5: 98.34 ms  178.5 GB/s
  iter 10: 96.78 ms  181.4 GB/s
  iter 15: 97.10 ms  180.8 GB/s
  iter 20: 98.42 ms  178.4 GB/s
  iter 25: 99.34 ms  176.7 GB/s
  iter 29: 98.92 ms  177.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.07 ms
achieved bandwidth       : 179.0 GB/s
projected decode         : 10.20 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T215206Z.json
[round 312] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T215206Z.json
```
