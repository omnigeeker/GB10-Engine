# Round 4 — 20260920T041607Z

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness | pass |
| benchmark | pass |

**status: PASS**

## Log

```
  output     err/scale=3.827e-7  max_rel=4.192e-7  max|y|=7.9739e1  OK
  proj_qkv   err/scale=2.484e-7  max_rel=8.459e-7  max|y|=1.1519e1  OK
  proj_z     err/scale=2.510e-7  max_rel=1.553e-6  max|y|=4.7489e0  OK
  conv_out   err/scale=1.737e-7  max_rel=1.291e-6  max|y|=3.4311e0  OK
  q_ln       err/scale=3.526e-7  max_rel=1.473e-6  max|y|=7.9249e-2  OK
  k_ln       err/scale=2.439e-7  max_rel=1.352e-6  max|y|=9.7734e-1  OK
  decay      err/scale=1.192e-7  max_rel=4.300e-7  max|y|=9.9998e-1  OK
  beta       err/scale=1.954e-7  max_rel=4.554e-7  max|y|=9.1530e-1  OK
  delta_out  err/scale=8.223e-7  max_rel=1.828e-6  max|y|=1.3025e-2  OK
  gnorm      err/scale=5.204e-7  max_rel=1.703e-6  max|y|=2.3825e1  OK
  (14 stage checks)
layer 3 (FullAttention)  fixture 8 tokens x 5120 hidden
  input_ln   err/scale=1.448e-7  max_rel=2.873e-7  max|y|=6.5851e0  OK
  o_proj     err/scale=2.780e-7  max_rel=5.081e-7  max|y|=8.2320e1  OK
  mlp        err/scale=2.587e-7  max_rel=4.713e-7  max|y|=2.2119e1  OK
  output     err/scale=2.466e-7  max_rel=3.601e-7  max|y|=9.2822e1  OK
  q_proj     err/scale=2.327e-7  max_rel=1.509e-6  max|y|=8.1974e0  OK
  v_proj     err/scale=1.728e-7  max_rel=9.378e-7  max|y|=8.2790e0  OK
  q_rope     err/scale=2.676e-7  max_rel=1.316e-6  max|y|=5.3449e0  OK
  k_rope     err/scale=2.790e-7  max_rel=1.428e-6  max|y|=5.1274e0  OK
  gate       err/scale=2.327e-7  max_rel=1.509e-6  max|y|=8.1974e0  OK
  attn_gated err/scale=4.461e-7  max_rel=3.076e-6  max|y|=5.3441e0  OK
  (10 stage checks)

all gates: OK
[round 4] correctness OK
[round 4] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 43.1s

per-token weight traffic : 17.555 GB
time per token           : 74.86 ms
achieved bandwidth       : 234.5 GB/s
projected decode         : 13.36 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 102.9% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T041607Z.json
[round 4] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T041607Z.json
```
