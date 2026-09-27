# Round 258 — 20260927T121856Z

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
[round 258] batch-parity OK
[round 258] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 258] prefix cache OK
[round 258] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      15.657       1.734    9.03x     2.00e-6   (103 GB/s serial, 929 GB/s warp)

decode-bench: OK
[round 258] decode-bench OK
[round 258] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 37.3s
interleave: false
  iter  0: 97.60 ms  179.9 GB/s
  iter  5: 97.59 ms  179.9 GB/s
  iter 10: 97.10 ms  180.8 GB/s
  iter 15: 97.46 ms  180.1 GB/s
  iter 20: 98.29 ms  178.6 GB/s
  iter 25: 96.38 ms  182.1 GB/s
  iter 29: 98.04 ms  179.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.59 ms
achieved bandwidth       : 179.9 GB/s
projected decode         : 10.25 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T121856Z.json
[round 258] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260927T121856Z.json
```
