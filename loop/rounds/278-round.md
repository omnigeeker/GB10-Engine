# Round 278 — 20260927T165846Z

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
[round 278] batch-parity OK
[round 278] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 278] prefix cache OK
[round 278] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.270       1.416   10.79x     2.00e-6   (105 GB/s serial, 1138 GB/s warp)

decode-bench: OK
[round 278] decode-bench OK
[round 278] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.8s
interleave: false
  iter  0: 97.79 ms  179.5 GB/s
  iter  5: 97.81 ms  179.5 GB/s
  iter 10: 98.47 ms  178.3 GB/s
  iter 15: 96.07 ms  182.7 GB/s
  iter 20: 96.84 ms  181.3 GB/s
  iter 25: 98.21 ms  178.8 GB/s
  iter 29: 98.41 ms  178.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.46 ms
achieved bandwidth       : 180.1 GB/s
projected decode         : 10.26 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T165846Z.json
[round 278] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T165846Z.json
```
