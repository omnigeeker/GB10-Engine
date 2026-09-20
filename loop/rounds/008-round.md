# Round 8 — 20260920T051439Z

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
[round 8] correctness OK
[round 8] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 42.2s  (17.60 GB streamed per token)
TTFT 6598.0 ms (59 prompt tokens)
decoded 16 tokens in 1.802s -> 8.88 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 8] generate OK
[round 8] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.1s
interleave: false
  iter  0: 101.94 ms  172.2 GB/s
  iter  5: 101.11 ms  173.6 GB/s
  iter 10: 102.70 ms  170.9 GB/s
  iter 15: 102.75 ms  170.9 GB/s
  iter 20: 100.23 ms  175.1 GB/s
  iter 25: 102.12 ms  171.9 GB/s
  iter 29: 102.22 ms  171.7 GB/s

per-token weight traffic : 17.555 GB
time per token           : 101.98 ms
achieved bandwidth       : 172.1 GB/s
projected decode         : 9.81 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 75.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T051439Z.json
[round 8] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T051439Z.json
```
