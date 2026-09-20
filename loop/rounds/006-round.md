# Round 6 — 20260920T045042Z

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
  gate       err/scale=2.327e-7  max_rel=1.509e-6  max|y|=8.1974e0  OK
  attn_gated err/scale=4.461e-7  max_rel=3.076e-6  max|y|=5.3441e0  OK
  (10 stage checks)

all gates: OK
[round 6] correctness OK
[round 6] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 42.5s  (17.60 GB streamed per token)
TTFT 6603.1 ms (59 prompt tokens)
decoded 16 tokens in 1.803s -> 8.87 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 6] generate OK
[round 6] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.2s
  iter  0: 73.37 ms  239.3 GB/s
  iter  5: 72.49 ms  242.2 GB/s
  iter 10: 72.54 ms  242.0 GB/s
  iter 15: 71.17 ms  246.7 GB/s
  iter 20: 73.28 ms  239.6 GB/s
  iter 25: 71.01 ms  247.2 GB/s
  iter 29: 71.93 ms  244.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 72.24 ms
achieved bandwidth       : 243.0 GB/s
projected decode         : 13.84 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 106.6% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T045042Z.json
[round 6] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T045042Z.json
```
