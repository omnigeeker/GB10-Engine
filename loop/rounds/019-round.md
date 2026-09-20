# Round 19 — 20260920T062006Z

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
[round 19] correctness OK
[round 19] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 42.6s  (17.60 GB streamed per token)
TTFT 492.6 ms (59 prompt tokens)
decoded 16 tokens in 1.874s -> 8.54 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 19] generate OK
[round 19] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.3s
interleave: false
  iter  0: 97.48 ms  180.1 GB/s
  iter  5: 98.46 ms  178.3 GB/s
  iter 10: 98.41 ms  178.4 GB/s
  iter 15: 96.99 ms  181.0 GB/s
  iter 20: 98.17 ms  178.8 GB/s
  iter 25: 96.10 ms  182.7 GB/s
  iter 29: 96.05 ms  182.8 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.99 ms
achieved bandwidth       : 181.0 GB/s
projected decode         : 10.31 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.4% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T062006Z.json
[round 19] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T062006Z.json
```
