# Round 410 — 20260929T105839Z

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness (layers) | pass |
| correctness (64-layer) | pass |
| longctx-follow (long prompt, no silent EOS) | pass |
| tc-parity (prefill GEMM vs fp32, real weights) | pass |
| decode-bench (warp vs serial) | pass |
| prefix cache A/B | pass |
| benchmark | pass |

**status: PASS**

## Log

```
batch-parity: OK
[round 410] batch-parity OK
[round 410] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'Session alpha acknowledged.\n\nI have received the repeated de'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 410] prefix cache OK
[round 410] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.633       1.672    8.16x     1.96e-6   (118 GB/s serial, 964 GB/s warp)

decode-bench: OK
[round 410] decode-bench OK
[round 410] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.4s
interleave: false
  iter  0: 117.00 ms  150.0 GB/s
  iter  5: 118.02 ms  148.7 GB/s
  iter 10: 117.67 ms  149.2 GB/s
  iter 15: 114.99 ms  152.7 GB/s
  iter 20: 118.38 ms  148.3 GB/s
  iter 25: 117.73 ms  149.1 GB/s
  iter 29: 117.94 ms  148.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.88 ms
achieved bandwidth       : 150.2 GB/s
projected decode         : 8.56 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260929T105839Z.json
[round 410] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260929T105839Z.json
```
