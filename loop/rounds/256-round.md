# Round 256 — 20260927T102017Z

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
[round 256] batch-parity OK
[round 256] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 256] prefix cache OK
[round 256] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.502       1.563    9.92x     2.07e-6   (104 GB/s serial, 1031 GB/s warp)

decode-bench: OK
[round 256] decode-bench OK
[round 256] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 38.4s
interleave: false
  iter  0: 97.81 ms  179.5 GB/s
  iter  5: 97.46 ms  180.1 GB/s
  iter 10: 95.45 ms  183.9 GB/s
  iter 15: 92.61 ms  189.6 GB/s
  iter 20: 94.04 ms  186.7 GB/s
  iter 25: 93.90 ms  187.0 GB/s
  iter 29: 92.77 ms  189.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 94.64 ms
achieved bandwidth       : 185.5 GB/s
projected decode         : 10.57 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 81.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T102017Z.json
[round 256] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T102017Z.json
```
