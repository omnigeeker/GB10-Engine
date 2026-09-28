# Round 398 — 20260928T153909Z

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
[round 398] batch-parity OK
[round 398] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 398] prefix cache OK
[round 398] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.638       1.229   11.10x     1.96e-6   (118 GB/s serial, 1311 GB/s warp)

decode-bench: OK
[round 398] decode-bench OK
[round 398] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.4s
interleave: false
  iter  0: 95.60 ms  183.6 GB/s
  iter  5: 98.14 ms  178.9 GB/s
  iter 10: 99.98 ms  175.6 GB/s
  iter 15: 96.76 ms  181.4 GB/s
  iter 20: 98.27 ms  178.6 GB/s
  iter 25: 97.09 ms  180.8 GB/s
  iter 29: 96.67 ms  181.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.55 ms
achieved bandwidth       : 180.0 GB/s
projected decode         : 10.25 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T153909Z.json
[round 398] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T153909Z.json
```
