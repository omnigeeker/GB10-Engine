# Round 262 — 20260927T132421Z

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
[round 262] batch-parity OK
[round 262] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 262] prefix cache OK
[round 262] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.534       1.437   10.81x     2.00e-6   (104 GB/s serial, 1121 GB/s warp)

decode-bench: OK
[round 262] decode-bench OK
[round 262] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.7s
interleave: false
  iter  0: 96.61 ms  181.7 GB/s
  iter  5: 95.82 ms  183.2 GB/s
  iter 10: 97.35 ms  180.3 GB/s
  iter 15: 97.55 ms  180.0 GB/s
  iter 20: 96.83 ms  181.3 GB/s
  iter 25: 98.73 ms  177.8 GB/s
  iter 29: 97.85 ms  179.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.23 ms
achieved bandwidth       : 180.6 GB/s
projected decode         : 10.29 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T132421Z.json
[round 262] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T132421Z.json
```
