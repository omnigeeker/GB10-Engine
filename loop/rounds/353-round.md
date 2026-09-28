# Round 353 — 20260928T063651Z

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
[round 353] batch-parity OK
[round 353] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 353] prefix cache OK
[round 353] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.919       0.861   16.16x     1.96e-6   (116 GB/s serial, 1870 GB/s warp)

decode-bench: OK
[round 353] decode-bench OK
[round 353] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 59.1s
interleave: false
  iter  0: 117.00 ms  150.0 GB/s
  iter  5: 119.40 ms  147.0 GB/s
  iter 10: 118.13 ms  148.6 GB/s
  iter 15: 114.42 ms  153.4 GB/s
  iter 20: 118.17 ms  148.6 GB/s
  iter 25: 119.32 ms  147.1 GB/s
  iter 29: 113.94 ms  154.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.45 ms
achieved bandwidth       : 149.5 GB/s
projected decode         : 8.51 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T063651Z.json
[round 353] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T063651Z.json
```
