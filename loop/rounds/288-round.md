# Round 288 — 20260927T183605Z

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
[round 288] batch-parity OK
[round 288] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 288] prefix cache OK
[round 288] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.512       1.631    9.51x     2.00e-6   (104 GB/s serial, 988 GB/s warp)

decode-bench: OK
[round 288] decode-bench OK
[round 288] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.9s
interleave: false
  iter  0: 95.87 ms  183.1 GB/s
  iter  5: 95.02 ms  184.8 GB/s
  iter 10: 93.69 ms  187.4 GB/s
  iter 15: 95.64 ms  183.6 GB/s
  iter 20: 96.33 ms  182.2 GB/s
  iter 25: 96.59 ms  181.7 GB/s
  iter 29: 96.29 ms  182.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.07 ms
achieved bandwidth       : 182.7 GB/s
projected decode         : 10.41 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.2% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T183605Z.json
[round 288] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T183605Z.json
```
