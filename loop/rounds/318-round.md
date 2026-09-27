# Round 318 — 20260927T224954Z

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
[round 318] batch-parity OK
[round 318] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 318] prefix cache OK
[round 318] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.674       1.795    8.73x     2.00e-6   (103 GB/s serial, 897 GB/s warp)

decode-bench: OK
[round 318] decode-bench OK
[round 318] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 97.41 ms  180.2 GB/s
  iter  5: 97.88 ms  179.4 GB/s
  iter 10: 93.76 ms  187.2 GB/s
  iter 15: 96.86 ms  181.2 GB/s
  iter 20: 94.74 ms  185.3 GB/s
  iter 25: 96.39 ms  182.1 GB/s
  iter 29: 93.80 ms  187.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 95.27 ms
achieved bandwidth       : 184.3 GB/s
projected decode         : 10.50 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.8% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T224954Z.json
[round 318] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T224954Z.json
```
