# Round 326 — 20260928T000104Z

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
[round 326] batch-parity OK
[round 326] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 326] prefix cache OK
[round 326] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.083       1.601    9.42x     2.00e-6   (107 GB/s serial, 1006 GB/s warp)

decode-bench: OK
[round 326] decode-bench OK
[round 326] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.8s
interleave: false
  iter  0: 97.87 ms  179.4 GB/s
  iter  5: 97.88 ms  179.3 GB/s
  iter 10: 100.40 ms  174.9 GB/s
  iter 15: 97.66 ms  179.8 GB/s
  iter 20: 98.76 ms  177.8 GB/s
  iter 25: 103.65 ms  169.4 GB/s
  iter 29: 101.03 ms  173.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 99.26 ms
achieved bandwidth       : 176.9 GB/s
projected decode         : 10.07 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 77.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T000104Z.json
[round 326] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T000104Z.json
```
