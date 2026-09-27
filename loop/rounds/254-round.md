# Round 254 — 20260927T092455Z

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
[round 254] batch-parity OK
[round 254] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 254] prefix cache OK
[round 254] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.818       1.638    9.66x     2.07e-6   (102 GB/s serial, 983 GB/s warp)

decode-bench: OK
[round 254] decode-bench OK
[round 254] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.7s
interleave: false
  iter  0: 97.01 ms  181.0 GB/s
  iter  5: 97.68 ms  179.7 GB/s
  iter 10: 100.76 ms  174.2 GB/s
  iter 15: 95.96 ms  182.9 GB/s
  iter 20: 98.56 ms  178.1 GB/s
  iter 25: 96.58 ms  181.8 GB/s
  iter 29: 98.78 ms  177.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.67 ms
achieved bandwidth       : 179.7 GB/s
projected decode         : 10.24 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T092455Z.json
[round 254] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T092455Z.json
```
