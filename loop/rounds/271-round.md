# Round 271 — 20260927T160251Z

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
[round 271] batch-parity OK
[round 271] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 271] prefix cache OK
[round 271] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.534       1.375   11.29x     2.00e-6   (104 GB/s serial, 1171 GB/s warp)

decode-bench: OK
[round 271] decode-bench OK
[round 271] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.6s
interleave: false
  iter  0: 92.84 ms  189.1 GB/s
  iter  5: 96.68 ms  181.6 GB/s
  iter 10: 96.22 ms  182.4 GB/s
  iter 15: 97.40 ms  180.2 GB/s
  iter 20: 97.86 ms  179.4 GB/s
  iter 25: 98.77 ms  177.7 GB/s
  iter 29: 96.73 ms  181.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.38 ms
achieved bandwidth       : 182.1 GB/s
projected decode         : 10.38 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T160251Z.json
[round 271] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T160251Z.json
```
