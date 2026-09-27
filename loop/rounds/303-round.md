# Round 303 — 20260927T203050Z

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
[round 303] batch-parity OK
[round 303] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 303] prefix cache OK
[round 303] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.213       1.379   11.03x     2.00e-6   (106 GB/s serial, 1168 GB/s warp)

decode-bench: OK
[round 303] decode-bench OK
[round 303] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.2s
interleave: false
  iter  0: 96.20 ms  182.5 GB/s
  iter  5: 94.63 ms  185.5 GB/s
  iter 10: 94.29 ms  186.2 GB/s
  iter 15: 94.39 ms  186.0 GB/s
  iter 20: 97.52 ms  180.0 GB/s
  iter 25: 94.23 ms  186.3 GB/s
  iter 29: 94.28 ms  186.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 95.71 ms
achieved bandwidth       : 183.4 GB/s
projected decode         : 10.45 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T203050Z.json
[round 303] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T203050Z.json
```
