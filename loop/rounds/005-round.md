# Round 5 — 20260920T043154Z

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness (layers) | pass |
| correctness (64-layer) | pass |
| benchmark | pass |

**status: PASS**

## Log

```
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
[round 5] correctness OK
[round 5] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 43.3s  (17.60 GB streamed per token)
TTFT 6859.8 ms (59 prompt tokens)
decoded 16 tokens in 1.862s -> 8.60 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 5] generate OK
[round 5] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.7s

per-token weight traffic : 17.555 GB
time per token           : 74.76 ms
achieved bandwidth       : 234.8 GB/s
projected decode         : 13.38 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 103.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T043154Z.json
[round 5] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T043154Z.json
```
