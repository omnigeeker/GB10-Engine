# Round 306 — 20260927T210024Z

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
[round 306] batch-parity OK
[round 306] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 306] prefix cache OK
[round 306] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.408       1.290   11.94x     2.00e-6   (105 GB/s serial, 1248 GB/s warp)

decode-bench: OK
[round 306] decode-bench OK
[round 306] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.3s
interleave: false
  iter  0: 98.00 ms  179.1 GB/s
  iter  5: 97.66 ms  179.8 GB/s
  iter 10: 98.03 ms  179.1 GB/s
  iter 15: 98.12 ms  178.9 GB/s
  iter 20: 97.27 ms  180.5 GB/s
  iter 25: 97.55 ms  180.0 GB/s
  iter 29: 97.08 ms  180.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.76 ms
achieved bandwidth       : 179.6 GB/s
projected decode         : 10.23 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T210024Z.json
[round 306] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T210024Z.json
```
