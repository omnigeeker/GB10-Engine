# Round 407 — 20260928T192246Z

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
[round 407] batch-parity OK
[round 407] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 407] prefix cache OK
[round 407] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.421       1.245   10.78x     1.96e-6   (120 GB/s serial, 1294 GB/s warp)

decode-bench: OK
[round 407] decode-bench OK
[round 407] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.8s
interleave: false
  iter  0: 97.46 ms  180.1 GB/s
  iter  5: 96.87 ms  181.2 GB/s
  iter 10: 97.46 ms  180.1 GB/s
  iter 15: 97.32 ms  180.4 GB/s
  iter 20: 95.77 ms  183.3 GB/s
  iter 25: 96.03 ms  182.8 GB/s
  iter 29: 98.61 ms  178.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.53 ms
achieved bandwidth       : 181.9 GB/s
projected decode         : 10.36 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T192246Z.json
[round 407] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T192246Z.json
```
