# Round 282 — 20260927T173643Z

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
[round 282] batch-parity OK
[round 282] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 282] prefix cache OK
[round 282] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      14.875       1.690    8.80x     2.00e-6   (108 GB/s serial, 953 GB/s warp)

decode-bench: OK
[round 282] decode-bench OK
[round 282] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.2s
interleave: false
  iter  0: 95.23 ms  184.3 GB/s
  iter  5: 94.87 ms  185.0 GB/s
  iter 10: 93.77 ms  187.2 GB/s
  iter 15: 94.62 ms  185.5 GB/s
  iter 20: 94.25 ms  186.3 GB/s
  iter 25: 94.35 ms  186.1 GB/s
  iter 29: 94.44 ms  185.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 94.50 ms
achieved bandwidth       : 185.8 GB/s
projected decode         : 10.58 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 81.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T173643Z.json
[round 282] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T173643Z.json
```
