# Round 409 — 20260929T103316Z

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
[round 409] batch-parity OK
[round 409] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 409] prefix cache OK
[round 409] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.150       1.807    7.28x     1.96e-6   (122 GB/s serial, 891 GB/s warp)

decode-bench: OK
[round 409] decode-bench OK
[round 409] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 60.5s
interleave: false
  iter  0: 115.86 ms  151.5 GB/s
  iter  5: 119.00 ms  147.5 GB/s
  iter 10: 120.74 ms  145.4 GB/s
  iter 15: 117.66 ms  149.2 GB/s
  iter 20: 115.25 ms  152.3 GB/s
  iter 25: 115.25 ms  152.3 GB/s
  iter 29: 117.32 ms  149.6 GB/s

per-token weight traffic : 17.555 GB
time per token           : 117.79 ms
achieved bandwidth       : 149.0 GB/s
projected decode         : 8.49 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 65.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260929T103316Z.json
[round 409] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260929T103316Z.json
```
