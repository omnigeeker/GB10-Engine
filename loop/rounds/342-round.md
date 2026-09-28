# Round 342 — 20260928T025510Z

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
[round 342] batch-parity OK
[round 342] prefix cache A/B (caching vs --no-prefix-cache)
=== caching server ===
=== non-caching server (--no-prefix-cache) ===

PASS: cache and no-cache produce identical output
  a1: 'It appears you have pasted the same paragraph about the arch'
  b: 'Session bravo acknowledged.\n\nI have received the repeated de'
  c: 'DONE'
[round 342] prefix cache OK
[round 342] gb10-verify decode-bench (warp vs serial reference)
q heads 24, kv heads 4, head_dim 256, n_seq 4
     keys   serial ms     warp ms  speedup     rms rel
     8192      13.241       1.595    8.30x     2.00e-6   (122 GB/s serial, 1009 GB/s warp)

decode-bench: OK
[round 342] decode-bench OK
[round 342] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 58.2s
interleave: false
  iter  0: 117.80 ms  149.0 GB/s
  iter  5: 116.14 ms  151.2 GB/s
  iter 10: 114.42 ms  153.4 GB/s
  iter 15: 117.29 ms  149.7 GB/s
  iter 20: 117.37 ms  149.6 GB/s
  iter 25: 113.78 ms  154.3 GB/s
  iter 29: 113.27 ms  155.0 GB/s

per-token weight traffic : 17.555 GB
time per token           : 116.16 ms
achieved bandwidth       : 151.1 GB/s
projected decode         : 8.61 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 66.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T025510Z.json
[round 342] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260928T025510Z.json
```
