# Round 384 — 20260928T131314Z

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
[round 384] batch-parity OK
[round 384] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 384] prefix cache OK
[round 384] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.113       1.670    7.85x     1.96e-6   (123 GB/s serial, 964 GB/s warp)

decode-bench: OK
[round 384] decode-bench OK
[round 384] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.4s
interleave: false
  iter  0: 120.98 ms  145.1 GB/s
  iter  5: 116.70 ms  150.4 GB/s
  iter 10: 116.60 ms  150.6 GB/s
  iter 15: 117.92 ms  148.9 GB/s
  iter 20: 116.81 ms  150.3 GB/s
  iter 25: 121.60 ms  144.4 GB/s
  iter 29: 119.28 ms  147.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 118.05 ms
achieved bandwidth       : 148.7 GB/s
projected decode         : 8.47 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T131314Z.json
[round 384] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T131314Z.json
```
