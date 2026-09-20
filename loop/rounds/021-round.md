# Round 21 — 20260920T063005Z

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
[round 21] correctness OK
[round 21] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 42.7s  (17.60 GB streamed per token)
TTFT 449.8 ms (59 prompt tokens)
decoded 16 tokens in 1.859s -> 8.61 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 21] generate OK
[round 21] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.0s
interleave: false
  iter  0: 97.98 ms  179.2 GB/s
  iter  5: 95.55 ms  183.7 GB/s
  iter 10: 96.72 ms  181.5 GB/s
  iter 15: 95.20 ms  184.4 GB/s
  iter 20: 95.56 ms  183.7 GB/s
  iter 25: 97.68 ms  179.7 GB/s
  iter 29: 97.32 ms  180.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.09 ms
achieved bandwidth       : 180.8 GB/s
projected decode         : 10.30 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T063005Z.json
[round 21] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T063005Z.json
```
