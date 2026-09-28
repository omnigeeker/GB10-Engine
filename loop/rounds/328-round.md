# Round 328 — 20260928T001914Z

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
[round 328] batch-parity OK
[round 328] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 328] prefix cache OK
[round 328] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.056       1.830    8.23x     2.00e-6   (107 GB/s serial, 880 GB/s warp)

decode-bench: OK
[round 328] decode-bench OK
[round 328] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 36.7s
interleave: false
  iter  0: 99.26 ms  176.9 GB/s
  iter  5: 99.01 ms  177.3 GB/s
  iter 10: 97.82 ms  179.5 GB/s
  iter 15: 98.05 ms  179.0 GB/s
  iter 20: 98.11 ms  178.9 GB/s
  iter 25: 101.02 ms  173.8 GB/s
  iter 29: 99.02 ms  177.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.59 ms
achieved bandwidth       : 178.1 GB/s
projected decode         : 10.14 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T001914Z.json
[round 328] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T001914Z.json
```
