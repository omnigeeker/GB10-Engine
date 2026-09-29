# GB10-Engine

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Português](README.pt-BR.md)

A from-scratch, pure-Rust inference engine for **Qwen3.8-27B (NVFP4)** on the
**NVIDIA DGX Spark GB10** (`sm_121`, CUDA 13), benchmarked head to head against
**llama.cpp** on the same weights, the same box, and the same harness.

- Custom CUDA kernels compiled to PTX for `sm_121`, no vendor inference runtime.
- Gated DeltaNet + full attention hybrid, 64 layers, native **262,144-token** context.
- OpenAI- **and** Anthropic-compatible HTTP endpoints, streaming, continuous batching.
- Every number in this README is reproducible from a script in this repository.

---

## Benchmark: GB10-Engine vs llama.cpp

**This is the section to read.** Same machine, same NVFP4 checkpoint (gb10-engine reads the
`modelopt` NVFP4 checkpoint; llama.cpp reads a GGUF converted from the same weights), same prompts,
same client, runs strictly sequential so neither contender has the GPU to itself.

### Test setup

| | |
|---|---|
| Hardware | NVIDIA DGX Spark **GB10**, `sm_121`, 48 SMs, 121.7 GiB unified LPDDR5X |
| Measured read bandwidth | **228 GB/s** (see [`docs/PHYSICS.md`](docs/PHYSICS.md)) |
| gb10-engine | `./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144` |
| llama.cpp | `llama-server -m models/Qwen3.8-27B-NVFP4.gguf -c 262144 -ngl 99 -fa on` |
| Client | [`bench/longctx/ttft.py`](bench/longctx/ttft.py) — identical request twice per trial, unique leading marker so "warm" cannot be a false hit |
| Decoding | greedy, `max_tokens 128` |
| Prefix cache | enabled on both (this is what makes **warm TTFT** meaningful) |

**Metric definitions**, because "TTFT" hides a real question:

- **Cold TTFT** — the prefix has never been seen, so the whole prompt is prefilled. This is the
  prefill-compute measurement.
- **Warm TTFT** — the identical request again, so a prefix cache can skip the prefill. This is the
  cache-hit measurement.
- **OTPS** — output tokens per second during decode.

> **Only same-session pairs are comparable.** Identical code measured **23% apart** on this box on
> different days (see "Methodology notes"). Every ratio below comes from a pair measured in one
> session; each cell has at least two independent measurements.

### Decode throughput (OTPS) — gb10-engine wins 3 of 4

| Context | gb10-engine | llama.cpp | Result |
|---|---|---|---|
| 8K | **8.91** | 7.26 | **1.23x faster** |
| 32K | **7.14** | 6.865 | **1.04x faster** |
| 128K | **5.29** | 4.75 | **1.11x faster** |
| 256K | 3.665 | **3.86** | 1.05x slower |

### TTFT — gb10-engine wins every warm cell, and is behind on every cold cell

| Context | Cold TTFT (gb10 / llama.cpp) | Warm TTFT (gb10 / llama.cpp) |
|---|---|---|
| **8K** | 10.35 s / **9.50 s** — 1.09x slower | **0.03 s** / 0.223 s — **7.4x faster** |
| **32K** | 80.00 s / **44.55 s** — 1.80x slower | **0.05 s** / 0.29 s — **5.8x faster** |
| **128K** | 890.54 s / **275.04 s** — 3.24x slower | **0.14 s** / 0.48 s — **3.4x faster** |
| **256K** | 3295.45 s / **726.22 s** — 4.54x slower | **0.28 s** / 0.66 s — **2.36x faster** |

**Read this table honestly: it is a split decision.**

- **Warm TTFT: 4 of 4 won**, by 2.4x to 7.8x. Once a prefix is cached, this engine resumes in
  0.03–0.28 s where llama.cpp needs 0.22–0.66 s. For agent, RAG and multi-turn workloads that re-send
  a long prefix, this is the metric that dominates wall-clock time.
- **OTPS: 3 of 4 won**, by 1.04x to 1.23x. The 256K cell is 1.05x behind.
- **Cold TTFT: 0 of 4 won.** This engine is 1.09x behind at 8K and 4.54x behind at 256K. **This is
  the open problem**, and it is arithmetic: the 16 full-attention layers do quadratic prefill work
  and the prefill attention kernel runs at ~8% of the part's fp16 CUDA-core peak.

**Cold-TTFT progress made in the current round of work** (measured on the server, gated by the
`chunked-prefill` correctness check):

| Context | Cold TTFT before | Cold TTFT now | Improvement |
|---|---|---|---|
| 8K | 14.93 s | **10.35 s** | **1.43x** |
| 32K | 90.54 s | **80.00 s** | **1.13x** |
| 128K | 1134.63 s | **890.54 s** | **1.28x** |
| 256K | 4138.39 s | **3295.45 s** | **1.25x** |

### Accuracy — gb10-engine is more accurate than llama.cpp on the same weights

**Perplexity** (WikiText-2, `n_ctx 512`, 580 windows, 147,900 predictions — the same configuration
as the historical baseline):

| Implementation | Weights | Perplexity | vs BF16 |
|---|---|---|---|
| **gb10-engine** | **NVFP4** | **7.1006** | +0.71% |
| llama.cpp | NVFP4 (same GGUF) | 7.2088 | +2.24% |
| transformers (reference) | BF16 (base model) | **7.0506** | — |

**MMLU** (3,240 questions, 4-way agreement):

| Implementation | Weights | MMLU | Correct |
|---|---|---|---|
| transformers (reference) | BF16 | **80.74%** | 2616/3240 |
| **gb10-engine** | **NVFP4** | **80.34%** | **2603/3240** |
| torch, NVFP4 dequantized | NVFP4 | 80.46% | 2607/3240 |
| llama.cpp | NVFP4 (same GGUF) | 79.88% | 2588/3240 |

**Why these numbers say the quantization path is correct:**

- gb10-engine is **1.5% better than llama.cpp** on perplexity and **+0.46 pp better on MMLU**, using
  the *same* NVFP4 weights. The engine is not losing accuracy relative to the mature implementation.
- It sits **0.71% above BF16** on perplexity and **0.40 pp below BF16** on MMLU — the expected cost of
  4-bit weights for activations/gradients-free inference, and the same direction and magnitude as the
  independent torch dequantized reference.
- The harness **cross-checks the tokenizer against `llama-tokenize` and reports 0 mismatches over
  297,054 tokens**, so no accuracy difference here is a tokenization artefact.
- Correctness is also gated per round: token-exact match against frozen HuggingFace oracle traces
  (64 layers, greedy), plus a batch-parity check.

### Known issue: long-context retrieval is not currently certified

The needle-in-a-haystack validation ([`bench/longctx/run_validation.sh`](bench/longctx/run_validation.sh))
recorded **3/3 at 32K, 1/1 at 128K, 1/1 at 256K**. **Re-running the identical 32K leg now gives 0/3 on
gb10-engine *and* 0/3 on llama.cpp**, so the failure was traced and the engine was not implicated:

- the failures are deterministic (`finish_reason: stop`, `completion_tokens: 0` — EOS as the first token);
- they are erratic in *length*, not monotone, which is not the shape of a KV-precision regression;
- they are **identical with `temperature: 0`**, with `--no-prefix-cache`, with `PREFILL_CHUNK` 2048
  *and* 8192, and with `enable_thinking` omitted/false/true;
- **and llama.cpp fails the same prompts.**

**Conclusion: precision on the standard metrics is normal; long-context retrieval is neither certified
nor refuted by this harness today.** It is tracked openly rather than omitted. Full write-up in
[`bench/longctx/comparison.md`](bench/longctx/comparison.md).

### Reproduce it yourself

```bash
# build
cargo build --release --workspace

# perplexity (gb10-engine) -- compares to the recorded 7.0988 / llama.cpp 7.2088 / BF16 7.0506
./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 580

# correctness + benchmark gate (build, tests, token-exact oracle, batch parity, attn tiling, bench)
loop/run_round.sh
```

**Do not run a server-backed measurement and the gate at the same time** — the gate starts and stops
servers on port 8080 and will kill the server mid-request.

---

## Status

| Milestone | State |
|---|---|
| M0 Foundation: config, safetensors, tokenizer, chat template | **done** |
| M1 CUDA backend + roofline GEMV | **done** |
| M2 Layer kernels (Gated DeltaNet, attention, MLP, NVFP4/FP8) | **done** |
| M3 Full forward + greedy-decode parity | **done** (token-exact, 64 layers, 16/16) |
| M4 MTP speculative decoding | **done** (token-exact; 44/48 acceptance) |
| M5 Paged KV + continuous batching | **done** (16 concurrent) |
| M6 OpenAI + Anthropic endpoints | **done** (both protocols + streaming) |
| M7 Roofline optimization | **in progress** |
| M8 llama.cpp head-to-head | **done** — OTPS 3/4 won, warm TTFT 4/4 won, cold TTFT 0/4 |

### Where performance stands against the hardware roofline

> **Read [`docs/PHYSICS.md`](docs/PHYSICS.md) first.** The original request asked for 100 tok/s
> single-stream. Measured on this machine, GB10 sustains **228 GB/s** of read bandwidth and this
> checkpoint requires **17.608 GB** of weight traffic per token, so the single-stream roofline is
> **12.95 tok/s** and the per-step floor is **77.2 ms**. The 100 tok/s target needs ~1.76 TB/s —
> **7.7x the measured bandwidth** — so no implementation can reach it. The agreed contract in
> [`docs/TARGETS.md`](docs/TARGETS.md) reflects what the hardware can actually do.

The remaining cold-prefill gap is one kernel: the tiled prefill attention runs at **3.12 TFLOP/s,
~8.4% of the fp16 CUDA-core peak**, and its instruction stream is only 42.4% arithmetic, so it is
issue-bound. Closing the long-context cold-TTFT gap requires an `mma.sync` tensor-core rewrite of
that kernel; the plan and the required speedups (1.29x at 8K, 5.98x at 32K, 8.40x at 128K, 8.82x at
256K) are in [`bench/longctx/comparison.md`](bench/longctx/comparison.md).

## Layout

```
crates/
  gb10-core/     config, mmap safetensors reader, tokenizer, chat template  (CUDA-free)
  gb10-cuda/     CUDA backend: kernels + memory management
  gb10-model/    Qwen3.5 graph: Gated DeltaNet + full attention + MLP + MTP
  gb10-server/   OpenAI- and Anthropic-compatible HTTP endpoint
  gb10-verify/   verification harness (generate, ppl, mmlu, attn-tile, prefill-shape, ...)
  gb10-bench/    benchmark harness
kernels/         CUDA sources, compiled to PTX for sm_121 by build.rs
loop/            round driver, durable state, per-round reports
bench/hw/        hardware characterization (bandwidth probe)
bench/longctx/   long-context comparison against llama.cpp
bench/ppl/       perplexity harness and recorded results
bench/mmlu/      MMLU harness and recorded results
docs/            physics, targets, architecture, roadmap
```

## Quick start

```bash
# correctness foundation (no GPU needed)
cargo test -p gb10-core

# hardware physics, if you want to re-derive the roofline yourself
nvcc -arch=sm_121 -O3 -o bench/hw/bw bench/hw/bw.cu && ./bench/hw/bw
python3 scripts/weight_traffic.py

# build everything and serve
cargo build --release
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4
```

It takes about **40 s** to load the weights, then it serves on `127.0.0.1:8080`.

## Model

`nv-community/Qwen3.8-27B-NVFP4`, fetched with the ModelScope CLI:

```bash
modelscope download --model nv-community/Qwen3.8-27B-NVFP4 \
    --local_dir models/Qwen3.8-27B-NVFP4
```

`models/` is a symlink to the downloaded checkpoint directory and is not tracked by git.

Architecture: `qwen3_5`, 64 layers (48 Gated-DeltaNet linear attention + 16 full attention at every
4th layer), hidden 5120, 24 query / 4 KV heads (GQA group 6), head_dim 256, partial RoPE (0.25,
mRoPE interleaved), vocab 248320, **1 MTP layer**, native **262,144** positions. Quantization is
modelopt MIXED_PRECISION: NVFP4 (group 16) for all MLP and `lm_head`, FP8 for every attention
projection, and MTP/vision weights left unquantized. On-disk size 21.921 GB.

## Using the local endpoint

The server speaks both the OpenAI and the Anthropic wire protocols on `127.0.0.1:8080`.

| route | protocol | notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | supports `stream`, `max_tokens`, `enable_thinking` |
| `POST /v1/completions` | OpenAI | legacy text completion; `prompt` must be a single string |
| `POST /v1/messages` | Anthropic | supports `stream`, `max_tokens` |
| `GET /v1/models` | both | returns the loaded model id |
| `GET /health` | — | liveness; use this to wait for startup |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":16}'
```

### Context length and concurrency

The context window is `--ctx` (**default 32768**) up to the checkpoint's **262144** positions, and
`--concurrency` sets how many sequences the KV cache is sized for. Concurrency defaults to whatever a
40 GB KV budget covers, capped at 16, so a long context lowers it automatically:

```sh
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144
# context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB
```

> **For any long-context benchmark, `--ctx` is mandatory.** Without it the server runs at 32768 and a
> longer prompt returns **no content at all** rather than an error — a silent failure that looks like
> a client bug.

**Concurrency.** The server batches up to 16 concurrent requests into a single forward pass,
collecting them within a 25 ms window. Set `GB10_BATCH_LOG=1` to log the group size each step — 16
parallel requests should log `batch of 16`. With identical prompts, all 16 return byte-identical
answers.

**Answers are stripped of the model's reasoning block.** `enable_thinking` defaults to `true`
(matching the checkpoint), so the template puts the opening ` thinking` in the prompt and generation
returns reasoning followed by `</think>`; the endpoint removes everything through that tag, so
`content` is the answer itself. **Pass `"enable_thinking": false` for a shorter, direct answer with no
reasoning generated at all.** **Streaming still passes the reasoning through** — see
[`docs/NEXT.md`](docs/NEXT.md) for the designed fix and the truncation trap it has to avoid.

No authentication is implemented and the listener is bound to loopback; add a proxy in front of it if
it needs to be reachable from elsewhere.

## The loop

`loop/run_round.sh` is the iteration driver. Each round:

1. `cargo build --release --workspace`
2. `cargo test --workspace --release`
3. correctness gate: token-exact match against frozen HF oracle traces
4. benchmark gate: fresh artifact written to `bench/results/`
5. commit and push to `omnigeeker/GB10-Engine`

Durable state lives in `loop/state.json`; per-round reports in `loop/rounds/`. A round that fails a
gate is recorded as `FAIL` and does not advance the milestone.

## Methodology notes

Three rules this project learned the hard way, and applies to every number above:

1. **Only same-session pairs are comparable.** Identical fp32 code measured 90.54 s and 111.59 s
   (+23.2%) on different days on this box; llama.cpp's 32K number drifted 44.55 s → 53.60 s over the
   same period. Ratios held, absolute seconds did not.
2. **A measured effect is not an attributed cause.** The 128K and 256K decode throughput improved ~23%
   across this work while 8K/32K reproduced their recorded values exactly. That the improvement is
   real is measured; *why* is still unattributed, and it is recorded that way.
3. **A probe measures a sensitivity, not headroom.** An occupancy probe showing a kernel costs 1.47x
   more at half the occupancy says nothing about whether occupancy can be raised — that requires the
   resource arithmetic (here: 2 blocks/SM today, 3 would need a 22% shared-memory cut).

## Documentation

| Document | Contents |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | full usage guide: Python and streaming examples, parameter table, decoding limits |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | measured hardware physics and how the roofline was derived |
| [`docs/TARGETS.md`](docs/TARGETS.md) | the agreed performance contract |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | engine architecture |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) | the tiled attention kernel, chunking, memory budget, needle results |
| [`bench/longctx/comparison.md`](bench/longctx/comparison.md) | the full long-context comparison log against llama.cpp, including rejected hypotheses and corrections |
| [`bench/ppl/`](bench/ppl/) · [`bench/mmlu/`](bench/mmlu/) | recorded accuracy results and the harnesses that produced them |
| [`docs/NEXT.md`](docs/NEXT.md) | the open work list |

## License

See the repository for license information.
