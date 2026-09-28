# Round 404 — 20260928T180625Z

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
[round 404] batch-parity OK
[round 404] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 404] prefix cache OK
[round 404] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.487       1.271    7.47x     1.96e-6   (170 GB/s serial, 1268 GB/s warp)

decode-bench: OK
[round 404] decode-bench OK
[round 404] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.6s
interleave: false
  iter  0: 97.39 ms  180.3 GB/s
  iter  5: 97.90 ms  179.3 GB/s
  iter 10: 97.61 ms  179.9 GB/s
  iter 15: 97.99 ms  179.2 GB/s
  iter 20: 97.17 ms  180.7 GB/s
  iter 25: 101.01 ms  173.8 GB/s
  iter 29: 102.58 ms  171.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.60 ms
achieved bandwidth       : 178.1 GB/s
projected decode         : 10.14 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T180625Z.json
[round 404] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T180625Z.json
```
