# Round 11 — 20260920T053303Z

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
[round 11] correctness OK
[round 11] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 42.4s  (17.60 GB streamed per token)
TTFT 6269.4 ms (59 prompt tokens)
decoded 16 tokens in 1.703s -> 9.40 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 11] generate OK
[round 11] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 41.8s
interleave: false
  iter  0: 100.18 ms  175.2 GB/s
  iter  5: 97.02 ms  180.9 GB/s
  iter 10: 95.67 ms  183.5 GB/s
  iter 15: 97.48 ms  180.1 GB/s
  iter 20: 96.88 ms  181.2 GB/s
  iter 25: 98.33 ms  178.5 GB/s
  iter 29: 97.87 ms  179.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.44 ms
achieved bandwidth       : 180.2 GB/s
projected decode         : 10.26 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.0% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T053303Z.json
[round 11] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T053303Z.json
```
