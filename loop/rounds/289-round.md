# Round 289 — 20260927T184210Z

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
[round 289] batch-parity OK
[round 289] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 289] prefix cache OK
[round 289] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.454       1.365   11.32x     2.00e-6   (104 GB/s serial, 1180 GB/s warp)

decode-bench: OK
[round 289] decode-bench OK
[round 289] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 96.80 ms  181.4 GB/s
  iter  5: 93.70 ms  187.4 GB/s
  iter 10: 96.35 ms  182.2 GB/s
  iter 15: 95.70 ms  183.4 GB/s
  iter 20: 92.61 ms  189.6 GB/s
  iter 25: 99.86 ms  175.8 GB/s
  iter 29: 98.02 ms  179.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 95.66 ms
achieved bandwidth       : 183.5 GB/s
projected decode         : 10.45 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T184210Z.json
[round 289] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T184210Z.json
```
