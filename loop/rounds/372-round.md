# Round 372 — 20260928T112127Z

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
[round 372] batch-parity OK
[round 372] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 372] prefix cache OK
[round 372] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.526       1.085   12.47x     1.96e-6   (119 GB/s serial, 1484 GB/s warp)

decode-bench: OK
[round 372] decode-bench OK
[round 372] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.1s
interleave: false
  iter  0: 118.58 ms  148.0 GB/s
  iter  5: 118.59 ms  148.0 GB/s
  iter 10: 117.33 ms  149.6 GB/s
  iter 15: 118.05 ms  148.7 GB/s
  iter 20: 117.29 ms  149.7 GB/s
  iter 25: 113.99 ms  154.0 GB/s
  iter 29: 117.78 ms  149.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.11 ms
achieved bandwidth       : 149.9 GB/s
projected decode         : 8.54 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T112127Z.json
[round 372] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T112127Z.json
```
