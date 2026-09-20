# Round 14 — 20260920T054910Z

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
  attn_gated err/scale=4.461e-7  max_rel=3.076e-6  max|y|=5.3441e0  OK
  (10 stage checks)

all gates: OK
[round 14] correctness OK
[round 14] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 42.0s  (17.60 GB streamed per token)
TTFT 2073.2 ms (59 prompt tokens)
decoded 16 tokens in 1.870s -> 8.56 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 14] generate OK
[round 14] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.2s
interleave: false
  iter  0: 96.43 ms  182.0 GB/s
  iter  5: 98.53 ms  178.2 GB/s
  iter 10: 96.59 ms  181.8 GB/s
  iter 15: 97.99 ms  179.2 GB/s
  iter 20: 96.02 ms  182.8 GB/s
  iter 25: 97.29 ms  180.4 GB/s
  iter 29: 98.24 ms  178.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.42 ms
achieved bandwidth       : 180.2 GB/s
projected decode         : 10.26 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T054910Z.json
[round 14] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T054910Z.json
```
