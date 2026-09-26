# Round 249 — 20260926T091657Z

**Milestone: the endpoint is deployed, and deploying it found two real bugs.**
No model behaviour changed. Both fixes are in `gb10-server`, and a usage guide
now exists at `docs/USAGE.md`.

## Two bugs found by actually using the service

1. **Every response reported `"model": ""`.** `MODEL_NAME` was a `thread_local!`
   — written once on the main thread at startup and read from every connection
   thread, each of which got its own empty copy. The comment above it said
   "published once so the request handlers do not need the `Engine`", which is
   what a `thread_local!` cannot do. Replaced with a process-wide `OnceLock`.
   `/v1/models` and every completion now report `Qwen3.8-27B-NVFP4`.

2. **`POST /v1/completions` returned `{"error":{"message":"not found"}}`.** The
   route was documented in `README.md` and required by **T8**, but had never been
   implemented. Added: legacy shape (`object: "text_completion"`, bare `text` per
   choice, `logprobs: null`, streaming uses `text` not `delta`). A batch `prompt`
   array is refused with a clear error rather than silently answered with only
   its first element.

Both were invisible to the gates, which test the engine and not the HTTP
surface. Deploying the thing is what found them.

## Verified live

| check | result |
|---|---|
| `GET /v1/models` | `Qwen3.8-27B-NVFP4` |
| `POST /v1/chat/completions` | `"1, 2, 3, 4, 5"` |
| `POST /v1/chat/completions` streaming | SSE deltas, model id populated |
| `POST /v1/completions` | legacy shape, `"The capital of France is **Paris**."` |
| `POST /v1/completions` streaming | `text` chunks |
| `POST /v1/messages` | Anthropic shape, `stop_reason: end_turn`, `"Paris"` |
| `enable_thinking: true` (default) | 56 tokens generated, `</think>` block stripped, content `391` |
| 16 concurrent requests | all 16 correct and distinct in 7.7 s |
| startup | port open after ~45 s |

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
  seq  2 (prompt   13 tok): match
  seq  3 (prompt   17 tok): match
  seq  4 (prompt   21 tok): match
  seq  5 (prompt   25 tok): match
  seq  6 (prompt   29 tok): match
  seq  7 (prompt   33 tok): match
  seq  8 (prompt   37 tok): match
  seq  9 (prompt   41 tok): match
  seq 10 (prompt   45 tok): match
  seq 11 (prompt   49 tok): match
  seq 12 (prompt   53 tok): match
  seq 13 (prompt   57 tok): match
  seq 14 (prompt   61 tok): match
  seq 15 (prompt   65 tok): match
batch parity: 16/16 sequences exact over 16 tokens

batch-parity: OK
[round 249] batch-parity OK
[round 249] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.8s
interleave: false
  iter  0: 80.86 ms  217.1 GB/s
  iter  5: 82.01 ms  214.1 GB/s
  iter 10: 79.91 ms  219.7 GB/s
  iter 15: 79.87 ms  219.8 GB/s
  iter 20: 85.83 ms  204.5 GB/s
  iter 25: 87.68 ms  200.2 GB/s
  iter 29: 81.61 ms  215.1 GB/s

per-token weight traffic : 17.555 GB
time per token           : 82.39 ms
achieved bandwidth       : 213.1 GB/s
projected decode         : 12.14 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 93.5% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260926T091657Z.json
[round 249] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260926T091657Z.json
```
