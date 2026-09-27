# Round 315 — 20260927T222238Z

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
[round 315] batch-parity OK
[round 315] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 315] prefix cache OK
[round 315] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.301       1.720    8.90x     2.00e-6   (105 GB/s serial, 936 GB/s warp)

decode-bench: OK
[round 315] decode-bench OK
[round 315] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.2s
interleave: false
  iter  0: 99.18 ms  177.0 GB/s
  iter  5: 96.36 ms  182.2 GB/s
  iter 10: 97.47 ms  180.1 GB/s
  iter 15: 96.60 ms  181.7 GB/s
  iter 20: 97.86 ms  179.4 GB/s
  iter 25: 97.99 ms  179.2 GB/s
  iter 29: 98.63 ms  178.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.32 ms
achieved bandwidth       : 178.6 GB/s
projected decode         : 10.17 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T222238Z.json
[round 315] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T222238Z.json
```
