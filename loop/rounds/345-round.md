# Round 345 — 20260928T044427Z

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
[round 345] batch-parity OK
[round 345] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 345] prefix cache OK
[round 345] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.147       1.643    8.00x     2.00e-6   (123 GB/s serial, 980 GB/s warp)

decode-bench: OK
[round 345] decode-bench OK
[round 345] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.6s
interleave: false
  iter  0: 119.57 ms  146.8 GB/s
  iter  5: 116.80 ms  150.3 GB/s
  iter 10: 117.56 ms  149.3 GB/s
  iter 15: 120.57 ms  145.6 GB/s
  iter 20: 120.25 ms  146.0 GB/s
  iter 25: 117.92 ms  148.9 GB/s
  iter 29: 118.26 ms  148.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.91 ms
achieved bandwidth       : 148.9 GB/s
projected decode         : 8.48 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T044427Z.json
[round 345] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T044427Z.json
```
