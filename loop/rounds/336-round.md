# Round 336 — 20260928T020138Z

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
[round 336] batch-parity OK
[round 336] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 336] prefix cache OK
[round 336] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.084       1.552    8.43x     2.00e-6   (123 GB/s serial, 1038 GB/s warp)

decode-bench: OK
[round 336] decode-bench OK
[round 336] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.8s
interleave: false
  iter  0: 116.74 ms  150.4 GB/s
  iter  5: 118.45 ms  148.2 GB/s
  iter 10: 114.90 ms  152.8 GB/s
  iter 15: 117.25 ms  149.7 GB/s
  iter 20: 117.07 ms  150.0 GB/s
  iter 25: 113.38 ms  154.8 GB/s
  iter 29: 117.19 ms  149.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.43 ms
achieved bandwidth       : 150.8 GB/s
projected decode         : 8.59 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.1% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T020138Z.json
[round 336] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T020138Z.json
```
