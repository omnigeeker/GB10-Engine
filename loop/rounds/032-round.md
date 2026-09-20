# Round 32 — 20260920T073938Z

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
[round 32] correctness OK
[round 32] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 42.0s  (17.60 GB streamed per token)
TTFT 453.3 ms (59 prompt tokens)
decoded 16 tokens in 1.872s -> 8.55 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 32] generate OK
[round 32] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 41.4s
interleave: false
  iter  0: 97.84 ms  179.4 GB/s
  iter  5: 97.60 ms  179.9 GB/s
  iter 10: 97.34 ms  180.4 GB/s
  iter 15: 99.62 ms  176.2 GB/s
  iter 20: 97.28 ms  180.5 GB/s
  iter 25: 92.67 ms  189.4 GB/s
  iter 29: 94.17 ms  186.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.69 ms
achieved bandwidth       : 181.6 GB/s
projected decode         : 10.34 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T073938Z.json
[round 32] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T073938Z.json
```
