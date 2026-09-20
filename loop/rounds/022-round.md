# Round 22 — 20260920T063458Z

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
[round 22] correctness OK
[round 22] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 42.3s  (17.60 GB streamed per token)
TTFT 478.0 ms (59 prompt tokens)
decoded 16 tokens in 1.943s -> 8.23 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 22] generate OK
[round 22] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.4s
interleave: false
  iter  0: 98.86 ms  177.6 GB/s
  iter  5: 99.04 ms  177.3 GB/s
  iter 10: 97.36 ms  180.3 GB/s
  iter 15: 97.37 ms  180.3 GB/s
  iter 20: 96.38 ms  182.1 GB/s
  iter 25: 98.35 ms  178.5 GB/s
  iter 29: 98.26 ms  178.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 98.15 ms
achieved bandwidth       : 178.9 GB/s
projected decode         : 10.19 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T063458Z.json
[round 22] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T063458Z.json
```
