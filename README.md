# GB10-Engine

A from-scratch, pure-Rust inference engine for **Qwen3.8-27B (NVFP4)** on the
**NVIDIA DGX Spark GB10** (`sm_121`, CUDA 13).

Targets, and the measured hardware physics behind them, are in
[`docs/TARGETS.md`](docs/TARGETS.md) and [`docs/PHYSICS.md`](docs/PHYSICS.md).

> **Read `docs/PHYSICS.md` first.** The original request asked for 100 tok/s
> single-stream. Measured on this machine, GB10 sustains **228 GB/s** of read
> bandwidth and this checkpoint requires **17.608 GB** of weight traffic per token,
> so the single-stream roofline is **12.95 tok/s** and the per-step floor is
> **77.2 ms**. The 100 tok/s target needs ~1.76 TB/s -- **7.7x the measured
> bandwidth** -- so no implementation can reach it. The agreed contract in
> `docs/TARGETS.md` reflects what the hardware can actually do.

## Status

| Milestone | State |
|---|---|
| M0 Foundation: config, safetensors, tokenizer, chat template | **done** |
| M1 CUDA backend + roofline GEMV | **done** |
| M2 Layer kernels (Gated DeltaNet, attention, MLP, NVFP4/FP8) | **done** |
| M3 Full forward + greedy-decode parity | **done** (token-exact, 64 layers, 16/16) |
| M4 MTP speculative decoding | **done** (token-exact; 44/48 acceptance; ~0.32x batched) |
| M5 Paged KV + continuous batching (16 concurrent) | **done** (47.78 tok/s at B=16) |
| M6 OpenAI + Anthropic endpoints | **done** (both protocols + streaming) |
| M7 Roofline optimization | **in progress** -- see below |
| M8 llama.cpp head-to-head | **done** (otp +11-14%; TTFT rate still 5.9x behind) |

### Where the performance actually stands

| | measured | target | |
|---|---|---|---|
| single-stream decode | **9.66 tok/s** | 12.5 (the agreed substitute) | 73% of the 12.95 roofline |
| engine, 16 concurrent | **47.78 tok/s** | 30 | **met** |
| endpoint, 16 concurrent | **19.8 tok/s** | 30 | prefill-bound |
| otp vs llama.cpp | **+11-14%** | better | **met** |
| TTFT rate vs llama.cpp | 7.36 vs 1.25 ms/token | better | **not met** |
| single-stream 100 tok/s | -- | 100 | **physically impossible (7.7x)** |

The remaining gap is two kernels, both characterized in `docs/NEXT.md`:
`nvfp4_gemv_kernel` reaches 190 GB/s against the 272 GB/s its own access pattern
sustains, and the prefill GEMM runs at **47% occupancy** -- bound by neither
bandwidth (18% of roofline) nor compute (7.4 TFLOPS).

## Layout

```
crates/
  gb10-core/     config, mmap safetensors reader, tokenizer, chat template  (CUDA-free)
  gb10-cuda/     CUDA backend: kernels + memory management
  gb10-model/    Qwen3.5 graph: Gated DeltaNet + full attention + MLP + MTP
  gb10-server/   OpenAI- and Anthropic-compatible HTTP endpoint
  gb10-bench/    benchmark + verification harness
kernels/         CUDA sources, compiled to PTX for sm_121 by build.rs
loop/            round driver, durable state, per-round reports
bench/hw/        hardware characterization (bandwidth probe)
docs/            physics, targets, architecture, roadmap
scripts/         standalone analysis tools
```

## Quick start

```bash
# correctness foundation (no GPU needed)
cargo test -p gb10-core

# hardware physics, if you want to re-derive the roofline yourself
nvcc -arch=sm_121 -O3 -o bench/hw/bw bench/hw/bw.cu && ./bench/hw/bw
python3 scripts/weight_traffic.py

# one iteration of the loop: build, test, gate, benchmark, commit, push
loop/run_round.sh
```

## Model

`nv-community/Qwen3.8-27B-NVFP4`, fetched with the ModelScope CLI:

```bash
modelscope download --model nv-community/Qwen3.8-27B-NVFP4 \
    --local_dir models/Qwen3.8-27B-NVFP4
```

`models/` is a symlink to the downloaded checkpoint directory and is not
tracked by git.

Architecture: `qwen3_5`, 64 layers (48 Gated-DeltaNet linear attention +
16 full attention at every 4th layer), hidden 5120, 24 query / 4 KV heads,
head_dim 256, partial RoPE (0.25, mRoPE interleaved), vocab 248320,
**1 MTP layer**. Quantization is modelopt MIXED_PRECISION: NVFP4 (group 16)
for all MLP and `lm_head`, FP8 for every attention projection, and MTP/vision
weights left unquantized.

## The loop

`loop/run_round.sh` is the iteration driver. Each round:

1. `cargo build --release --workspace`
2. `cargo test --workspace --release`
3. correctness gate: token-exact match against frozen HF oracle traces
4. benchmark gate: fresh artifact written to `bench/results/`
5. commit and push to `omnigeeker/GB10-Engine`

Durable state lives in `loop/state.json`; per-round reports in `loop/rounds/`.
A round that fails a gate is recorded as `FAIL` and does not advance the
milestone.

## Using the local endpoint

The server speaks both the OpenAI and the Anthropic wire protocols on
`127.0.0.1:8080`. Start it with the model directory as the argument:

```sh
cargo build --release
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4
```

It takes about 95 s to load the weights, then it serves:

| route | protocol | notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | supports `stream`, `max_tokens`, `enable_thinking` |
| `POST /v1/messages` | Anthropic | supports `stream`, `max_tokens` |
| `GET /v1/models` | both | returns the loaded model id |

**OpenAI client:**

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":16}'
```

**Anthropic client:**

```sh
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"Count from one to five."}]}'
```

**Concurrency.** The server batches up to 16 concurrent requests into a single
forward pass, collecting them within a 25 ms window. Set `GB10_BATCH_LOG=1` to log
the group size each step -- 16 parallel requests should log `batch of 16`.
Measured on one GB10: **256 completion tokens from 16 concurrent requests in
12.9 s, i.e. 19.8 tok/s aggregate**, with identical prompts returning byte-identical
answers.

**Answers are stripped of the model's reasoning block.** `enable_thinking` defaults to
`true` (matching the checkpoint), so the template puts the opening ` thinking` in the
prompt and generation returns reasoning followed by `</think>`; the endpoint removes
everything through that tag, so `content` is the answer itself. **Pass
`"enable_thinking": false` for a shorter, direct answer with no reasoning generated at
all** -- in that case the stripper is a no-op and the text passes through untouched.
**Streaming still passes the reasoning through** -- see `docs/NEXT.md` for the
designed fix and the truncation trap it has to avoid.

No authentication is implemented and the listener is bound to loopback; add a
proxy in front of it if it needs to be reachable from elsewhere.
