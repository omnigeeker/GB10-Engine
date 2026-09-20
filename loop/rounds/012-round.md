# Round 12 — 20260920T053833Z

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness (layers) | fail |
| correctness (64-layer) | pass |
| benchmark | pass |

**status: FAIL**

## Log

```

thread 'main' (494206) panicked at crates/gb10-verify/src/main.rs:187:26:
copy_from_slice: source slice length (2621440) does not match destination slice length (5120)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
[round 12] correctness FAILED (see /home/wayne/dsh/QWen3.8-27B-GB10/loop/logs/round-012-20260920T053833Z.log)
[round 12] gb10-verify generate (64-layer greedy decode vs full-model oracle)
prompt: 59 tokens
prompt ids match the oracle (59 tokens)
model loaded in 41.5s  (17.60 GB streamed per token)
TTFT 4992.8 ms (59 prompt tokens)
decoded 16 tokens in 1.859s -> 8.61 tok/s
ids:  [1421, 16561, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 8722, 57879, 3296, 13, 21134]
text: "User asks: \"What is the capital of France?\" Simple factual question. Answer"
oracle agreement: 16/16 (100.0%)  reference=Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16
  exact match

generate: OK
[round 12] generate OK
[round 12] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 43.0s
interleave: false
  iter  0: 101.51 ms  172.9 GB/s
  iter  5: 99.37 ms  176.7 GB/s
  iter 10: 99.46 ms  176.5 GB/s
  iter 15: 98.38 ms  178.4 GB/s
  iter 20: 97.16 ms  180.7 GB/s
  iter 25: 98.89 ms  177.5 GB/s
  iter 29: 99.47 ms  176.5 GB/s

per-token weight traffic : 17.555 GB
time per token           : 97.83 ms
achieved bandwidth       : 179.4 GB/s
projected decode         : 10.22 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 78.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T053833Z.json
[round 12] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260920T053833Z.json
```
