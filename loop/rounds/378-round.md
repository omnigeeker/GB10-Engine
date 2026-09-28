# Round 378 — 20260928T122215Z

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
[round 378] batch-parity OK
[round 378] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 378] prefix cache OK
[round 378] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.370       1.710    7.82x     1.96e-6   (120 GB/s serial, 942 GB/s warp)

decode-bench: OK
[round 378] decode-bench OK
[round 378] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.8s
interleave: false
  iter  0: 118.19 ms  148.5 GB/s
  iter  5: 115.34 ms  152.2 GB/s
  iter 10: 116.96 ms  150.1 GB/s
  iter 15: 116.32 ms  150.9 GB/s
  iter 20: 114.91 ms  152.8 GB/s
  iter 25: 117.44 ms  149.5 GB/s
  iter 29: 115.14 ms  152.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.54 ms
achieved bandwidth       : 150.6 GB/s
projected decode         : 8.58 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T122215Z.json
[round 378] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T122215Z.json
```
