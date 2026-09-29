# Round 411 — 20260929T122011Z

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
[round 411] batch-parity OK
[round 411] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'Session alpha acknowledged.\n\nI have received the repeated de'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 411] prefix cache OK
[round 411] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.225       1.735    7.62x     1.96e-6   (122 GB/s serial, 928 GB/s warp)

decode-bench: OK
[round 411] decode-bench OK
[round 411] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.5s
interleave: false
  iter  0: 120.36 ms  145.9 GB/s
  iter  5: 116.63 ms  150.5 GB/s
  iter 10: 117.00 ms  150.0 GB/s
  iter 15: 115.99 ms  151.4 GB/s
  iter 20: 117.83 ms  149.0 GB/s
  iter 25: 115.81 ms  151.6 GB/s
  iter 29: 116.78 ms  150.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.27 ms
achieved bandwidth       : 149.7 GB/s
projected decode         : 8.53 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260929T122011Z.json
[round 411] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260929T122011Z.json
```
