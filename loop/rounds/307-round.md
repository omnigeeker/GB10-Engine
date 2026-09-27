# Round 307 — 20260927T211008Z

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
[round 307] batch-parity OK
[round 307] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 307] prefix cache OK
[round 307] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.353       1.373    6.81x     2.00e-6   (172 GB/s serial, 1173 GB/s warp)

decode-bench: OK
[round 307] decode-bench OK
[round 307] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.2s
interleave: false
  iter  0: 97.77 ms  179.6 GB/s
  iter  5: 96.68 ms  181.6 GB/s
  iter 10: 98.04 ms  179.1 GB/s
  iter 15: 98.81 ms  177.7 GB/s
  iter 20: 97.46 ms  180.1 GB/s
  iter 25: 96.28 ms  182.3 GB/s
  iter 29: 97.77 ms  179.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.20 ms
achieved bandwidth       : 180.6 GB/s
projected decode         : 10.29 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T211008Z.json
[round 307] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T211008Z.json
```
