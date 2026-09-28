# Round 365 — 20260928T101544Z

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
[round 365] batch-parity OK
[round 365] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 365] prefix cache OK
[round 365] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.655       0.904   15.11x     1.96e-6   (118 GB/s serial, 1782 GB/s warp)

decode-bench: OK
[round 365] decode-bench OK
[round 365] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.5s
interleave: false
  iter  0: 118.18 ms  148.5 GB/s
  iter  5: 115.88 ms  151.5 GB/s
  iter 10: 115.95 ms  151.4 GB/s
  iter 15: 114.28 ms  153.6 GB/s
  iter 20: 118.26 ms  148.5 GB/s
  iter 25: 117.69 ms  149.2 GB/s
  iter 29: 115.58 ms  151.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.76 ms
achieved bandwidth       : 150.4 GB/s
projected decode         : 8.56 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T101544Z.json
[round 365] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T101544Z.json
```
