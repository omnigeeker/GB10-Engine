# Round 28 — 20260920T070309Z

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
[round 28] correctness OK
[round 28] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 41.9s  (17.60 GB streamed per token)
TTFT 452.4 ms (59 prompt tokens)
decoded 16 tokens in 1.859s -> 8.61 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 28] generate OK
[round 28] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.9s
interleave: false
  iter  0: 96.44 ms  182.0 GB/s
  iter  5: 99.06 ms  177.2 GB/s
  iter 10: 97.88 ms  179.3 GB/s
  iter 15: 94.53 ms  185.7 GB/s
  iter 20: 98.23 ms  178.7 GB/s
  iter 25: 99.76 ms  176.0 GB/s
  iter 29: 98.71 ms  177.9 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.14 ms
achieved bandwidth       : 180.7 GB/s
projected decode         : 10.29 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T070309Z.json
[round 28] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T070309Z.json
```
