# Round 253 — 20260927T084354Z

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
[round 253] batch-parity OK
[round 253] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 253] prefix cache OK
[round 253] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.274       1.599    9.55x     2.07e-6   (105 GB/s serial, 1007 GB/s warp)

decode-bench: OK
[round 253] decode-bench OK
[round 253] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.9s
interleave: false
  iter  0: 96.41 ms  182.1 GB/s
  iter  5: 95.67 ms  183.5 GB/s
  iter 10: 96.15 ms  182.6 GB/s
  iter 15: 95.31 ms  184.2 GB/s
  iter 20: 97.24 ms  180.5 GB/s
  iter 25: 98.07 ms  179.0 GB/s
  iter 29: 97.67 ms  179.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.77 ms
achieved bandwidth       : 181.4 GB/s
projected decode         : 10.33 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T084354Z.json
[round 253] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T084354Z.json
```
