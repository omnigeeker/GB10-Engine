# Round 298 — 20260927T194556Z

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
[round 298] batch-parity OK
[round 298] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 298] prefix cache OK
[round 298] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192       9.669       1.342    7.21x     2.00e-6   (167 GB/s serial, 1200 GB/s warp)

decode-bench: OK
[round 298] decode-bench OK
[round 298] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.8s
interleave: false
  iter  0: 95.14 ms  184.5 GB/s
  iter  5: 96.53 ms  181.9 GB/s
  iter 10: 94.79 ms  185.2 GB/s
  iter 15: 94.78 ms  185.2 GB/s
  iter 20: 96.12 ms  182.6 GB/s
  iter 25: 94.36 ms  186.0 GB/s
  iter 29: 97.15 ms  180.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 94.99 ms
achieved bandwidth       : 184.8 GB/s
projected decode         : 10.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 81.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T194556Z.json
[round 298] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T194556Z.json
```
