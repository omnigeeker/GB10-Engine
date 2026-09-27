# Round 291 — 20260927T185443Z

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
[round 291] batch-parity OK
[round 291] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 291] prefix cache OK
[round 291] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.345       1.342    6.96x     2.00e-6   (172 GB/s serial, 1200 GB/s warp)

decode-bench: OK
[round 291] decode-bench OK
[round 291] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.9s
interleave: false
  iter  0: 93.99 ms  186.8 GB/s
  iter  5: 100.80 ms  174.2 GB/s
  iter 10: 94.40 ms  186.0 GB/s
  iter 15: 94.88 ms  185.0 GB/s
  iter 20: 93.12 ms  188.5 GB/s
  iter 25: 96.01 ms  182.9 GB/s
  iter 29: 96.08 ms  182.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 95.68 ms
achieved bandwidth       : 183.5 GB/s
projected decode         : 10.45 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T185443Z.json
[round 291] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T185443Z.json
```
