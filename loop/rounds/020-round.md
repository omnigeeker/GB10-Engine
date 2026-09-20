# Round 20 — 20260920T062602Z

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
[round 20] correctness OK
[round 20] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 41.8s  (17.60 GB streamed per token)
TTFT 456.1 ms (59 prompt tokens)
decoded 16 tokens in 1.844s -> 8.68 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 20] generate OK
[round 20] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 43.6s
interleave: false
  iter  0: 98.10 ms  179.0 GB/s
  iter  5: 97.43 ms  180.2 GB/s
  iter 10: 97.69 ms  179.7 GB/s
  iter 15: 94.68 ms  185.4 GB/s
  iter 20: 95.56 ms  183.7 GB/s
  iter 25: 99.99 ms  175.6 GB/s
  iter 29: 94.23 ms  186.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.61 ms
achieved bandwidth       : 181.7 GB/s
projected decode         : 10.35 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T062602Z.json
[round 20] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T062602Z.json
```
