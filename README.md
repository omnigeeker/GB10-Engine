# GB10-Engine

A from-scratch, pure-Rust inference engine for **Qwen3.8-27B (NVFP4)** on the
**NVIDIA DGX Spark GB10** (`sm_121`, CUDA 13).

Targets, and the measured hardware physics behind them, are in
[`docs/TARGETS.md`](docs/TARGETS.md) and [`docs/PHYSICS.md`](docs/PHYSICS.md).

> **Read `docs/PHYSICS.md` first.** The original request asked for 100 tok/s
> single-stream. GB10 provides ~200 GB/s of usable read bandwidth and this
> checkpoint requires 21.0 GB of weight reads per token, so the single-stream
> ceiling is **9.5 tok/s**. That target is off by 10.5x and no implementation
> can reach it; the agreed contract in `docs/TARGETS.md` reflects what the
> hardware can actually do.

## Status

| Milestone | State |
|---|---|
| M0 Foundation: config, safetensors, tokenizer, chat template | **done** |
| M1 CUDA backend + roofline GEMV | in progress |
| M2 Layer kernels (Gated DeltaNet, attention, MLP, NVFP4/FP8) | pending |
| M3 Full forward + greedy-decode parity | pending |
| M4 MTP speculative decoding | pending |
| M5 Paged KV + continuous batching (16 concurrent) | pending |
| M6 OpenAI + Anthropic endpoints | pending |
| M7 Roofline optimization | pending |
| M8 Ollama head-to-head | pending |

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
13.6 s, i.e. 18.8 tok/s aggregate**, with identical prompts returning byte-identical
answers.

No authentication is implemented and the listener is bound to loopback; add a
proxy in front of it if it needs to be reachable from elsewhere.
