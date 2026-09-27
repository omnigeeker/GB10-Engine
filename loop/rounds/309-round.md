# Round 309 — 20260927T212616Z

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
[round 309] batch-parity OK
[round 309] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 309] prefix cache OK
[round 309] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.294       1.373   11.14x     2.00e-6   (105 GB/s serial, 1173 GB/s warp)

decode-bench: OK
[round 309] decode-bench OK
[round 309] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.0s
interleave: false
  iter  0: 101.33 ms  173.2 GB/s
  iter  5: 97.04 ms  180.9 GB/s
  iter 10: 98.86 ms  177.6 GB/s
  iter 15: 97.96 ms  179.2 GB/s
  iter 20: 97.78 ms  179.5 GB/s
  iter 25: 98.27 ms  178.6 GB/s
  iter 29: 97.44 ms  180.2 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.36 ms
achieved bandwidth       : 178.5 GB/s
projected decode         : 10.17 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T212616Z.json
[round 309] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T212616Z.json
```
