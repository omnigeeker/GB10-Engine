# Round 402 — 20260928T172413Z

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
[round 402] batch-parity OK
[round 402] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 402] prefix cache OK
[round 402] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.452       1.205   11.17x     1.96e-6   (120 GB/s serial, 1337 GB/s warp)

decode-bench: OK
[round 402] decode-bench OK
[round 402] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.6s
interleave: false
  iter  0: 96.28 ms  182.3 GB/s
  iter  5: 94.33 ms  186.1 GB/s
  iter 10: 96.01 ms  182.9 GB/s
  iter 15: 93.63 ms  187.5 GB/s
  iter 20: 96.12 ms  182.6 GB/s
  iter 25: 95.89 ms  183.1 GB/s
  iter 29: 96.41 ms  182.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 95.70 ms
achieved bandwidth       : 183.4 GB/s
projected decode         : 10.45 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 80.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T172413Z.json
[round 402] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T172413Z.json
```
