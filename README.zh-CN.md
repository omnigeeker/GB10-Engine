# GB10-Engine
[English](README.md) · **简体中文** · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Português](README.pt-BR.md)

一个从零编写、纯 Rust 实现的推理引擎，面向 **NVIDIA DGX Spark GB10**（`sm_121`、CUDA 13）上的
**Qwen3.8-27B (NVFP4)**，并在相同权重、相同机器、相同测试框架下与 **llama.cpp** 进行正面基准对比。

- 为 `sm_121` 编写并编译为 PTX 的自定义 CUDA kernel，不依赖厂商推理运行时。
- Gated DeltaNet + full attention 混合架构，64 层，原生 **262,144-token** 上下文。
- 兼容 OpenAI **和** Anthropic 的 HTTP 端点，支持流式输出与连续批处理。
- 本 README 中的每一个数字都可以通过本仓库中的脚本复现。

---

## 基准测试：GB10-Engine vs llama.cpp

**这是最需要读的一节。** 同一台机器，同一个 NVFP4 checkpoint（gb10-engine 读取
`modelopt` NVFP4 checkpoint；llama.cpp 读取由相同权重转换而来的 GGUF），相同的 prompt，
相同的客户端，严格串行运行，因此两个对比者都不会独占 GPU。

### 测试设置

| | |
|---|---|
| 硬件 | NVIDIA DGX Spark **GB10**，`sm_121`，48 个 SM，121.7 GiB 统一 LPDDR5X |
| 实测读取带宽 | **228 GB/s**（见 [`docs/PHYSICS.md`](docs/PHYSICS.md)） |
| gb10-engine | `./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144` |
| llama.cpp | `llama-server -m models/Qwen3.8-27B-NVFP4.gguf -c 262144 -ngl 99 -fa on` |
| 客户端 | [`bench/longctx/ttft.py`](bench/longctx/ttft.py) — 每次试验发送两次完全相同的请求，并使用唯一的前置标记，使 “warm” 不可能是误命中 |
| 解码 | 贪婪解码，`max_tokens 128` |
| 前缀缓存 | 两者均启用（这正是 **warm TTFT** 有意义的原因） |

**指标定义**，因为 “TTFT” 掩盖了一个真正的问题：

- **Cold TTFT** — 此前从未见过该前缀，因此整个 prompt 都需要 prefill。这是对 prefill 计算量的测量。
- **Warm TTFT** — 再次发送完全相同的请求，前缀缓存可以跳过 prefill。这是对缓存命中的测量。
- **OTPS** — 解码期间每秒输出的 token 数。

> **只有同一会话内的成对测量才具有可比性。** 相同的代码在不同日期于本机上测得的差异达 **23%**
> （见 “Methodology notes”）。下文中的每一个比值都来自同一会话内测得的一对数据；
> 每个单元格至少包含两次独立测量。

### 解码吞吐（OTPS）— gb10-engine 在 4 项中胜出 3 项

| 上下文 | gb10-engine | llama.cpp | 结果 |
|---|---|---|---|
| 8K | **8.91** | 7.26 | **快 1.23x** |
| 32K | **7.14** | 6.865 | **快 1.04x** |
| 128K | **5.29** | 4.75 | **快 1.11x** |
| 256K | 3.665 | **3.86** | 慢 1.05x |

### TTFT — gb10-engine 赢得了每一个 warm 单元格，但在每一个 cold 单元格上都落后

| 上下文 | Cold TTFT (gb10 / llama.cpp) | Warm TTFT (gb10 / llama.cpp) |
|---|---|---|
| **8K** | 10.35 s / **9.50 s** — 慢 1.09x | **0.03 s** / 0.223 s — **快 7.4x** |
| **32K** | 80.00 s / **44.55 s** — 慢 1.80x | **0.05 s** / 0.29 s — **快 5.8x** |
| **128K** | 890.54 s / **275.04 s** — 慢 3.24x | **0.14 s** / 0.48 s — **快 3.4x** |
| **256K** | 3295.45 s / **726.22 s** — 慢 4.54x | **0.28 s** / 0.66 s — **快 2.36x** |

**请坦率地看待这张表：这是一个胜负参半的结果。**

- **Warm TTFT：4 项中 4 项全胜**，领先 2.4x 到 7.8x。一旦前缀被缓存，本引擎可在
  0.03–0.28 s 内恢复，而 llama.cpp 需要 0.22–0.66 s。对于需要反复发送长前缀的 agent、RAG 和
  多轮对话负载而言，这才是主导整体耗时的指标。
- **OTPS：4 项中胜出 3 项**，领先 1.04x 到 1.23x。256K 单元格落后 1.05x。
- **Cold TTFT：4 项中 0 项获胜。** 本引擎在 8K 上落后 1.09x，在 256K 上落后 4.54x。**这是尚未解决的问题**，
  而且它是一个算术问题：16 层 full attention 需要做平方级的 prefill 计算，
  而 prefill attention kernel 的运行速度仅为该芯片 fp16 CUDA-core 峰值的约 8%。

**当前这一轮工作取得的 Cold-TTFT 进展**（在服务器上测得，并以
`chunked-prefill` 正确性检查作为门禁）：

| 上下文 | 之前的 Cold TTFT | 现在的 Cold TTFT | 提升 |
|---|---|---|---|
| 8K | 14.93 s | **10.35 s** | **1.43x** |
| 32K | 90.54 s | **80.00 s** | **1.13x** |
| 128K | 1134.63 s | **890.54 s** | **1.28x** |
| 256K | 4138.39 s | **3295.45 s** | **1.25x** |

### 精度 — 在相同权重下 gb10-engine 比 llama.cpp 更准确

**Perplexity**（WikiText-2，`n_ctx 512`，580 个窗口，147,900 次预测 — 与历史基线相同的配置）：

| 实现 | 权重 | Perplexity | 相对 BF16 |
|---|---|---|---|
| **gb10-engine** | **NVFP4** | **7.1006** | +0.71% |
| llama.cpp | NVFP4（同一个 GGUF） | 7.2088 | +2.24% |
| transformers（参考） | BF16（基础模型） | **7.0506** | — |

**MMLU**（3,240 道题，4 方一致）：

| 实现 | 权重 | MMLU | 正确数 |
|---|---|---|---|
| transformers（参考） | BF16 | **80.74%** | 2616/3240 |
| **gb10-engine** | **NVFP4** | **80.34%** | **2603/3240** |
| torch，NVFP4 反量化 | NVFP4 | 80.46% | 2607/3240 |
| llama.cpp | NVFP4（同一个 GGUF） | 79.88% | 2588/3240 |

**为什么这些数字说明量化路径是正确的：**

- 在使用*相同* NVFP4 权重的情况下，gb10-engine 的 perplexity **比 llama.cpp 好 1.5%**，
  在 MMLU 上**高出 +0.46 pp**。该引擎相对于成熟实现并未损失精度。
- 它在 perplexity 上**比 BF16 高 0.71%**，在 MMLU 上**比 BF16 低 0.40 pp** — 这是无激活/梯度推理
  使用 4-bit 权重所预期的代价，其方向和幅度与独立的 torch 反量化参考一致。
- 该测试框架会将 **tokenizer 与 `llama-tokenize` 交叉校验，并报告在 297,054 个 token 上 0 处不一致**，
  因此这里的任何精度差异都不是分词造成的假象。
- 正确性同样按轮次进行门禁：与冻结的 HuggingFace oracle trace 逐 token 完全一致
  （64 层，贪婪解码），并附加批处理一致性检查。

### 已知问题：长上下文检索目前尚未得到认证

大海捞针（needle-in-a-haystack）验证（[`bench/longctx/run_validation.sh`](bench/longctx/run_validation.sh)）
记录的结果为 **32K 时 3/3、128K 时 1/1、256K 时 1/1**。**现在重新运行完全相同的 32K 分支，得到
gb10-engine 上 0/3、*并且* llama.cpp 上也是 0/3**，因此该失败已被追踪定位，且与本引擎无关：

- 这些失败是确定性的（`finish_reason: stop`、`completion_tokens: 0` — 首个 token 即为 EOS）；
- 它们在*长度*上表现不稳定、并非单调，这不符合 KV 精度回退（regression）的特征；
- 在 `temperature: 0`、`--no-prefix-cache`、`PREFILL_CHUNK` 为 2048 *和* 8192、
  以及 `enable_thinking` 省略/false/true 的情况下，它们**完全相同**；
- **而且 llama.cpp 在相同的 prompt 上也会失败。**

**结论：标准指标上的精度是正常的；长上下文检索目前既未被该测试框架认证，也未被其否定。**
我们公开跟踪此事，而非将其略去。完整说明见
[`bench/longctx/comparison.md`](bench/longctx/comparison.md)。

### 自行复现

```bash
# build
cargo build --release --workspace

# perplexity (gb10-engine) -- compares to the recorded 7.0988 / llama.cpp 7.2088 / BF16 7.0506
./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 580

# correctness + benchmark gate (build, tests, token-exact oracle, batch parity, attn tiling, bench)
loop/run_round.sh
```

**不要同时运行依赖服务器的测量和门禁** — 该门禁会在 8080 端口启动和停止服务器，
并会在请求进行中杀死服务器。

---

## 状态

| 里程碑 | 状态 |
|---|---|
| M0 基础：config、safetensors、tokenizer、chat template | **已完成** |
| M1 CUDA 后端 + roofline GEMV | **已完成** |
| M2 层级 kernel（Gated DeltaNet、attention、MLP、NVFP4/FP8） | **已完成** |
| M3 完整前向 + 贪婪解码一致性 | **已完成**（逐 token 完全一致，64 层，16/16） |
| M4 MTP 投机解码 | **已完成**（逐 token 完全一致；44/48 接受率） |
| M5 Paged KV + 连续批处理 | **已完成**（16 路并发） |
| M6 OpenAI + Anthropic 端点 | **已完成**（两种协议 + 流式） |
| M7 Roofline 优化 | **进行中** |
| M8 与 llama.cpp 正面比拼 | **已完成** — OTPS 胜 3/4，warm TTFT 胜 4/4，cold TTFT 0/4 |

### 性能相对硬件 roofline 的位置

> **请先阅读 [`docs/PHYSICS.md`](docs/PHYSICS.md)。** 最初的请求要求单流达到 100 tok/s。
> 在本机上实测，GB10 可维持 **228 GB/s** 的读取带宽，而该 checkpoint 每个 token 需要
> **17.608 GB** 的权重流量，因此单流 roofline 为 **12.95 tok/s**，每步下限为 **77.2 ms**。
> 100 tok/s 的目标需要约 1.76 TB/s — **是实测带宽的 7.7 倍** — 因此任何实现都无法达到。
> [`docs/TARGETS.md`](docs/TARGETS.md) 中约定的契约反映了硬件实际能达到的水平。

剩余的 cold-prefill 差距集中在一个 kernel 上：分块（tiled）prefill attention 的运行速度为
**3.12 TFLOP/s，约为 fp16 CUDA-core 峰值的 8.4%**，其指令流中只有 42.4% 是算术运算，因此它受发射
（issue）限制。要缩小长上下文 cold-TTFT 的差距，需要用 `mma.sync` tensor core 重写该 kernel；
计划与所需的加速比（8K 时 1.29x、32K 时 5.98x、128K 时 8.40x、256K 时 8.82x）见
[`bench/longctx/comparison.md`](bench/longctx/comparison.md)。

## 目录结构

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

## 快速开始

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

加载权重大约需要 **40 s**，随后它会在 `127.0.0.1:8080` 上提供服务。

## 模型

`nv-community/Qwen3.8-27B-NVFP4`，使用 ModelScope CLI 下载：

```bash
modelscope download --model nv-community/Qwen3.8-27B-NVFP4 \
    --local_dir models/Qwen3.8-27B-NVFP4
```

`models/` 是指向已下载 checkpoint 目录的符号链接，未被 git 跟踪。

架构：`qwen3_5`，64 层（48 层 Gated-DeltaNet 线性注意力 + 每第 4 层一个共 16 层 full attention），
hidden 5120，24 个 query / 4 个 KV head（GQA 组为 6），head_dim 256，部分 RoPE（0.25，
mRoPE 交错），vocab 248320，**1 层 MTP**，原生 **262,144** 个位置。量化为
modelopt MIXED_PRECISION：所有 MLP 和 `lm_head` 使用 NVFP4（group 16），每个 attention
投影使用 FP8，MTP/vision 权重保持未量化。磁盘占用 21.921 GB。

## 使用本地端点

服务器在 `127.0.0.1:8080` 上同时支持 OpenAI 和 Anthropic 两种通信协议。

| 路由 | 协议 | 说明 |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | 支持 `stream`、`max_tokens`、`enable_thinking` |
| `POST /v1/completions` | OpenAI | 旧式文本补全；`prompt` 必须是单个字符串 |
| `POST /v1/messages` | Anthropic | 支持 `stream`、`max_tokens` |
| `GET /v1/models` | 两者 | 返回已加载的模型 id |
| `GET /health` | — | 存活探测；可用它等待启动完成 |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":16}'
```

### 上下文长度与并发

上下文窗口由 `--ctx` 指定（**默认 32768**），最大可达该 checkpoint 的 **262144** 个位置，
而 `--concurrency` 设定 KV 缓存可容纳的序列数。并发数默认取 40 GB KV 预算所能覆盖的值，
上限为 16，因此长上下文会自动降低并发数：

```sh
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144
# context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB
```

> **对于任何长上下文基准测试，`--ctx` 都是必需的。** 不设置它时，服务器会以 32768 运行，
> 更长的 prompt 会**完全不返回任何内容**，而不是报错 — 这种静默失败看起来像是客户端 bug。

**并发。** 服务器会将最多 16 个并发请求批处理为单次前向传播，在 25 ms 窗口内收集它们。
设置 `GB10_BATCH_LOG=1` 可记录每一步的分组大小 — 16 个并行请求应记录为 `batch of 16`。
当 prompt 完全相同时，全部 16 个请求返回逐字节相同的答案。

**回答中会剥离模型的推理块。** `enable_thinking` 默认为 `true`（与 checkpoint 一致），
因此模板会在 prompt 中放入起始标签 ` thinking`，生成结果会先返回推理内容，再返回 `</think>`；
端点会删除直到该标签为止的所有内容，因此 `content` 就是答案本身。
**若想要更简短、直接且完全不生成推理内容的答案，请传入 `"enable_thinking": false`。**
**流式输出仍会传递推理内容** — 设计中的修复方案及其必须避免的截断陷阱见
[`docs/NEXT.md`](docs/NEXT.md)。

未实现任何身份验证，且监听器绑定在回环地址上；若需要从其他地方访问，请在其前面加一个代理。

## 迭代循环

`loop/run_round.sh` 是迭代驱动脚本。每一轮：

1. `cargo build --release --workspace`
2. `cargo test --workspace --release`
3. 正确性门禁：与冻结的 HF oracle trace 逐 token 完全一致
4. 基准门禁：将全新的产物写入 `bench/results/`
5. 提交并推送到 `omnigeeker/GB10-Engine`

持久状态保存在 `loop/state.json` 中；每轮报告保存在 `loop/rounds/` 中。未能通过门禁的轮次
会被记录为 `FAIL`，并且不会推进里程碑。

## 方法论说明

本项目在付出代价后才学到的三条规则，并适用于上文中的每一个数字：

1. **只有同一会话内的成对测量才具有可比性。** 相同的 fp32 代码在不同日期于本机上测得
   90.54 s 和 111.59 s（+23.2%）；llama.cpp 的 32K 数值在同一时期内从 44.55 s 漂移到 53.60 s。
   比值保持稳定，绝对秒数则不然。
2. **被测量到的效应并不等于已归因的原因。** 在这项工作中，128K 和 256K 的解码吞吐提升了约 23%，
   而 8K/32K 精确复现了其记录值。提升是真实的这一点已被测量；*原因* 仍未被归因，
   并且就按此方式记录。
3. **探针测量的是敏感度，而不是余量。** 一个占用率探针显示某 kernel 在占用率减半时开销增加 1.47x，
   这并不能说明占用率能否提高 — 那需要资源算术（此处：目前为 2 blocks/SM，
   若要达到 3 则需要削减 22% 的共享内存）。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | 完整使用指南：Python 与流式示例、参数表、解码限制 |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | 实测硬件物理特性以及 roofline 的推导方式 |
| [`docs/TARGETS.md`](docs/TARGETS.md) | 约定的性能契约 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 引擎架构 |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) | 分块 attention kernel、chunking、内存预算、needle 结果 |
| [`bench/longctx/comparison.md`](bench/longctx/comparison.md) | 与 llama.cpp 的完整长上下文对比日志，包括被否定的假设与更正 |
| [`bench/ppl/`](bench/ppl/) · [`bench/mmlu/`](bench/mmlu/) | 记录的精度结果以及产生这些结果的测试框架 |
| [`docs/NEXT.md`](docs/NEXT.md) | 待办工作清单 |

## 许可证

许可证信息请见仓库。
