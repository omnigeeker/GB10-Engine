# Round 31 — 20260920T072318Z

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
[round 31] correctness OK
[round 31] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 41.4s  (17.60 GB streamed per token)
TTFT 452.6 ms (59 prompt tokens)
decoded 16 tokens in 1.852s -> 8.64 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 31] generate OK
[round 31] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.1s
interleave: false
  iter  0: 96.60 ms  181.7 GB/s
  iter  5: 97.36 ms  180.3 GB/s
  iter 10: 96.06 ms  182.8 GB/s
  iter 15: 97.88 ms  179.4 GB/s
  iter 20: 96.03 ms  182.8 GB/s
  iter 25: 96.54 ms  181.8 GB/s
  iter 29: 97.36 ms  180.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 96.89 ms
achieved bandwidth       : 181.2 GB/s
projected decode         : 10.32 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 79.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T072318Z.json
[round 31] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T072318Z.json
```
