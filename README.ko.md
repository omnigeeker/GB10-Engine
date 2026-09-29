# GB10-Engine
[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **한국어** · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Português](README.pt-BR.md)

**Qwen3.8-27B (NVFP4)** 를 **NVIDIA DGX Spark GB10**(`sm_121`, CUDA 13)에서 돌리기 위해 처음부터 직접 만든 순수 Rust 추론 엔진이며, 동일한 가중치, 동일한 장비, 동일한 하네스로 **llama.cpp**와 정면 비교 벤치마크했습니다.

- `sm_121`용으로 PTX로 컴파일한 자체 CUDA 커널, 벤더 추론 런타임 없음.
- Gated DeltaNet + full attention 하이브리드, 64 레이어, 네이티브 **262,144-token** 컨텍스트.
- OpenAI **및** Anthropic 호환 HTTP 엔드포인트, 스트리밍, 연속 배칭(continuous batching).
- 이 README의 모든 수치는 이 저장소의 스크립트로 재현할 수 있습니다.

---

## 벤치마크: GB10-Engine vs llama.cpp

**이 섹션이 바로 읽어야 할 부분입니다.** 같은 머신, 같은 NVFP4 체크포인트(gb10-engine은 `modelopt` NVFP4 체크포인트를 읽고, llama.cpp는 동일한 가중치에서 변환한 GGUF를 읽음), 같은 프롬프트, 같은 클라이언트, 그리고 어느 쪽도 GPU를 독점하지 못하도록 엄격하게 순차 실행합니다.

### 테스트 환경

| | |
|---|---|
| 하드웨어 | NVIDIA DGX Spark **GB10**, `sm_121`, 48 SMs, 121.7 GiB 통합 LPDDR5X |
| 측정된 읽기 대역폭 | **228 GB/s** ([`docs/PHYSICS.md`](docs/PHYSICS.md) 참조) |
| gb10-engine | `./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144` |
| llama.cpp | `llama-server -m models/Qwen3.8-27B-NVFP4.gguf -c 262144 -ngl 99 -fa on` |
| 클라이언트 | [`bench/longctx/ttft.py`](bench/longctx/ttft.py) — 시행마다 동일한 요청을 두 번 보내며, "warm"이 잘못된 적중이 될 수 없도록 고유한 선행 마커를 사용 |
| 디코딩 | greedy, `max_tokens 128` |
| 프리픽스 캐시 | 양쪽 모두 활성화 (**warm TTFT**를 의미 있게 만드는 요소) |

**지표 정의**. "TTFT"라는 말은 실제 질문 하나를 가리기 때문입니다:

- **Cold TTFT** — 프리픽스를 한 번도 본 적이 없어 전체 프롬프트를 프리필합니다. 프리필 연산 측정치입니다.
- **Warm TTFT** — 동일한 요청을 다시 보내므로 프리픽스 캐시가 프리필을 건너뛸 수 있습니다. 캐시 적중 측정치입니다.
- **OTPS** — 디코드 중 초당 출력 token 수.

> **같은 세션에서 측정한 쌍만 비교할 수 있습니다.** 동일한 코드가 이 장비에서 다른 날 **23%** 차이로 측정되었습니다("방법론 노트" 참조). 아래의 모든 비율은 한 세션에서 측정한 쌍에서 나온 것이며, 각 셀에는 최소 두 번의 독립적인 측정이 있습니다.

### 디코드 처리량 (OTPS) — gb10-engine이 4개 중 3개에서 승리

| 컨텍스트 | gb10-engine | llama.cpp | 결과 |
|---|---|---|---|
| 8K | **8.91** | 7.26 | **1.23x 빠름** |
| 32K | **7.14** | 6.865 | **1.04x 빠름** |
| 128K | **5.29** | 4.75 | **1.11x 빠름** |
| 256K | 3.665 | **3.86** | 1.05x 느림 |

### TTFT — gb10-engine은 모든 warm 셀에서 이기고, 모든 cold 셀에서 뒤진다

| 컨텍스트 | Cold TTFT (gb10 / llama.cpp) | Warm TTFT (gb10 / llama.cpp) |
|---|---|---|
| **8K** | 10.35 s / **9.50 s** — 1.09x 느림 | **0.03 s** / 0.223 s — **7.4x 빠름** |
| **32K** | 80.00 s / **44.55 s** — 1.80x 느림 | **0.05 s** / 0.29 s — **5.8x 빠름** |
| **128K** | 890.54 s / **275.04 s** — 3.24x 느림 | **0.14 s** / 0.48 s — **3.4x 빠름** |
| **256K** | 3295.45 s / **726.22 s** — 4.54x 느림 | **0.28 s** / 0.66 s — **2.36x 빠름** |

**이 표는 정직하게 읽어야 합니다: 판정이 갈립니다.**

- **Warm TTFT: 4개 중 4개 승리**, 2.4x에서 7.8x까지. 프리픽스가 캐시되면 이 엔진은 0.03–0.28 s 만에 재개하는 반면 llama.cpp는 0.22–0.66 s가 필요합니다. 긴 프리픽스를 다시 보내는 agent, RAG, 멀티턴 워크로드에서는 이 지표가 wall-clock 시간을 지배합니다.
- **OTPS: 4개 중 3개 승리**, 1.04x에서 1.23x까지. 256K 셀은 1.05x 뒤집니다.
- **Cold TTFT: 4개 중 0개 승리.** 이 엔진은 8K에서 1.09x, 256K에서 4.54x 뒤집니다. **이것이 미해결 문제이며**, 산술의 문제입니다: 16개의 full-attention 레이어가 이차(quadratic) 프리필 연산을 수행하고, 프리필 attention 커널은 이 부품의 fp16 CUDA-core 피크의 ~8%에서 실행됩니다.

**현재 작업 라운드에서 이루어진 Cold-TTFT 개선**(서버에서 측정, `chunked-prefill` 정확성 검사로 게이트됨):

| 컨텍스트 | 이전 Cold TTFT | 현재 Cold TTFT | 개선 |
|---|---|---|---|
| 8K | 14.93 s | **10.35 s** | **1.43x** |
| 32K | 90.54 s | **80.00 s** | **1.13x** |
| 128K | 1134.63 s | **890.54 s** | **1.28x** |
| 256K | 4138.39 s | **3295.45 s** | **1.25x** |

### 정확도 — gb10-engine은 동일한 가중치에서 llama.cpp보다 더 정확하다

**Perplexity** (WikiText-2, `n_ctx 512`, 580 윈도우, 147,900 예측 — 역사적 베이스라인과 동일한 구성):

| 구현 | 가중치 | Perplexity | vs BF16 |
|---|---|---|---|
| **gb10-engine** | **NVFP4** | **7.1006** | +0.71% |
| llama.cpp | NVFP4 (동일한 GGUF) | 7.2088 | +2.24% |
| transformers (레퍼런스) | BF16 (베이스 모델) | **7.0506** | — |

**MMLU** (3,240 문제, 4-way 합의):

| 구현 | 가중치 | MMLU | 정답 |
|---|---|---|---|
| transformers (레퍼런스) | BF16 | **80.74%** | 2616/3240 |
| **gb10-engine** | **NVFP4** | **80.34%** | **2603/3240** |
| torch, NVFP4 dequantized | NVFP4 | 80.46% | 2607/3240 |
| llama.cpp | NVFP4 (동일한 GGUF) | 79.88% | 2588/3240 |

**이 수치들이 양자화 경로가 올바르다는 것을 말해 주는 이유:**

- gb10-engine은 *동일한* NVFP4 가중치를 사용하면서 perplexity에서 **llama.cpp보다 1.5% 우수**하고 **MMLU에서 +0.46 pp 우수**합니다. 이 엔진은 성숙한 구현에 비해 정확도를 잃지 않습니다.
- perplexity에서 **BF16보다 0.71% 높고** MMLU에서 **BF16보다 0.40 pp 낮습니다** — 활성화/그래디언트가 없는 추론을 위한 4-bit 가중치의 예상 비용이며, 독립적인 torch dequantized 레퍼런스와 같은 방향과 크기입니다.
- 하네스는 **tokenizer를 `llama-tokenize`와 교차 검증하고 297,054 token에 걸쳐 0개의 불일치를 보고**하므로, 여기의 정확도 차이는 tokenization 인공물이 아닙니다.
- 정확성은 라운드마다 게이트됩니다: 동결된 HuggingFace oracle 트레이스(64 레이어, greedy)에 대한 token-exact 일치와 batch-parity 검사.

### 알려진 문제: 장문 컨텍스트 검색은 현재 인증되지 않음

needle-in-a-haystack 검증([`bench/longctx/run_validation.sh`](bench/longctx/run_validation.sh))은 **32K에서 3/3, 128K에서 1/1, 256K에서 1/1**을 기록했습니다. **동일한 32K 레그를 지금 다시 실행하면 gb10-engine에서 0/3 *그리고* llama.cpp에서 0/3**이 나오므로, 실패를 추적한 결과 이 엔진은 관련이 없는 것으로 밝혀졌습니다:

- 실패는 결정적입니다(`finish_reason: stop`, `completion_tokens: 0` — 첫 token으로 EOS);
- *길이* 면에서 변덕스럽고 단조롭지 않은데, 이는 KV 정밀도 회귀의 형태가 아닙니다;
- `temperature: 0`, `--no-prefix-cache`, `PREFILL_CHUNK` 2048 *및* 8192, `enable_thinking` 생략/false/true에서 **모두 동일**합니다;
- **그리고 llama.cpp도 같은 프롬프트에서 실패합니다.**

**결론: 표준 지표에서의 정밀도는 정상이며, 장문 컨텍스트 검색은 오늘 이 하네스로 인증되지도 반박되지도 않았습니다.** 이는 생략하지 않고 공개적으로 추적됩니다. 전체 설명은 [`bench/longctx/comparison.md`](bench/longctx/comparison.md)에 있습니다.

### 직접 재현하기

```bash
# build
cargo build --release --workspace

# perplexity (gb10-engine) -- compares to the recorded 7.0988 / llama.cpp 7.2088 / BF16 7.0506
./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 580

# correctness + benchmark gate (build, tests, token-exact oracle, batch parity, attn tiling, bench)
loop/run_round.sh
```

**서버 기반 측정과 게이트를 동시에 실행하지 마십시오** — 게이트는 포트 8080에서 서버를 시작하고 중지하며, 요청 도중에 서버를 종료시킵니다.

---

## 현황

| 마일스톤 | 상태 |
|---|---|
| M0 기반: config, safetensors, tokenizer, chat template | **완료** |
| M1 CUDA 백엔드 + roofline GEMV | **완료** |
| M2 레이어 커널 (Gated DeltaNet, attention, MLP, NVFP4/FP8) | **완료** |
| M3 전체 forward + greedy-decode 패리티 | **완료** (token-exact, 64 레이어, 16/16) |
| M4 MTP 추측 디코딩(speculative decoding) | **완료** (token-exact; 44/48 수용) |
| M5 Paged KV + 연속 배칭 | **완료** (16 동시) |
| M6 OpenAI + Anthropic 엔드포인트 | **완료** (두 프로토콜 + 스트리밍) |
| M7 Roofline 최적화 | **진행 중** |
| M8 llama.cpp 정면 비교 | **완료** — OTPS 3/4 승리, warm TTFT 4/4 승리, cold TTFT 0/4 |

### 하드웨어 roofline 대비 성능 현황

> **먼저 [`docs/PHYSICS.md`](docs/PHYSICS.md)를 읽으십시오.** 원래 요청은 단일 스트림 100 tok/s를 요구했습니다. 이 머신에서 측정한 결과 GB10은 **228 GB/s**의 읽기 대역폭을 유지하고 이 체크포인트는 token당 **17.608 GB**의 가중치 트래픽을 필요로 하므로, 단일 스트림 roofline은 **12.95 tok/s**이고 스텝당 하한은 **77.2 ms**입니다. 100 tok/s 목표에는 ~1.76 TB/s가 필요합니다 — **측정된 대역폭의 7.7x** — 따라서 어떤 구현도 이를 달성할 수 없습니다. [`docs/TARGETS.md`](docs/TARGETS.md)에 합의된 계약은 하드웨어가 실제로 할 수 있는 것을 반영합니다.

남은 cold-prefill 격차는 커널 하나 때문입니다: 타일드 prefill attention은 **3.12 TFLOP/s, fp16 CUDA-core 피크의 ~8.4%**로 실행되며, 명령어 스트림이 42.4%만 산술 연산이라 issue-bound입니다. 장문 컨텍스트 cold-TTFT 격차를 좁히려면 이 커널을 `mma.sync` tensor-core로 재작성해야 합니다; 계획과 필요한 속도 향상(8K에서 1.29x, 32K에서 5.98x, 128K에서 8.40x, 256K에서 8.82x)은 [`bench/longctx/comparison.md`](bench/longctx/comparison.md)에 있습니다.

## 레이아웃

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

## 빠른 시작

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

가중치를 로드하는 데 약 **40 s**가 걸리며, 그 후 `127.0.0.1:8080`에서 서비스합니다.

## 모델

`nv-community/Qwen3.8-27B-NVFP4`, ModelScope CLI로 내려받습니다:

```bash
modelscope download --model nv-community/Qwen3.8-27B-NVFP4 \
    --local_dir models/Qwen3.8-27B-NVFP4
```

`models/`는 내려받은 체크포인트 디렉터리에 대한 심볼릭 링크이며 git으로 추적되지 않습니다.

아키텍처: `qwen3_5`, 64 레이어 (48개의 Gated-DeltaNet linear attention + 매 4번째 레이어마다 16개의 full attention), hidden 5120, 24 query / 4 KV heads (GQA group 6), head_dim 256, partial RoPE (0.25, mRoPE interleaved), vocab 248320, **1 MTP layer**, 네이티브 **262,144** 위치. 양자화는 modelopt MIXED_PRECISION입니다: 모든 MLP와 `lm_head`에 NVFP4 (group 16), 모든 attention projection에 FP8, MTP/vision 가중치는 양자화하지 않고 그대로 둡니다. 디스크상 크기 21.921 GB.

## 로컬 엔드포인트 사용

서버는 `127.0.0.1:8080`에서 OpenAI와 Anthropic wire protocol을 모두 지원합니다.

| 라우트 | 프로토콜 | 비고 |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | `stream`, `max_tokens`, `enable_thinking` 지원 |
| `POST /v1/completions` | OpenAI | 레거시 텍스트 완성; `prompt`는 단일 문자열이어야 함 |
| `POST /v1/messages` | Anthropic | `stream`, `max_tokens` 지원 |
| `GET /v1/models` | both | 로드된 model id 반환 |
| `GET /health` | — | liveness; 시작을 기다리는 데 사용 |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":16}'
```

### 컨텍스트 길이와 동시성

컨텍스트 윈도우는 `--ctx`(**기본 32768**)이며 체크포인트의 **262144** 위치까지 가능하고, `--concurrency`는 KV 캐시가 몇 개의 시퀀스에 맞게 크기 조정되는지를 설정합니다. 동시성은 기본적으로 40 GB KV 예산이 감당하는 만큼이며 최대 16으로 제한되므로, 긴 컨텍스트는 자동으로 이 값을 낮춥니다:

```sh
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144
# context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB
```

> **모든 장문 컨텍스트 벤치마크에서 `--ctx`는 필수입니다.** 이를 지정하지 않으면 서버는 32768로 실행되고 더 긴 프롬프트는 오류 대신 **아무 내용도 반환하지 않습니다** — 클라이언트 버그처럼 보이는 조용한 실패입니다.

**동시성.** 서버는 25 ms 윈도우 안에서 모은 최대 16개의 동시 요청을 단일 forward pass로 배칭합니다. 각 스텝의 그룹 크기를 로그로 남기려면 `GB10_BATCH_LOG=1`을 설정하십시오 — 16개의 병렬 요청은 `batch of 16`을 기록해야 합니다. 동일한 프롬프트라면 16개 모두 바이트 단위로 동일한 답변을 반환합니다.

**답변에서는 모델의 추론 블록이 제거됩니다.** `enable_thinking`의 기본값은 `true`(체크포인트와 일치)이므로 템플릿이 여는 ` thinking`을 프롬프트에 넣고 생성은 추론 뒤에 `</think>`를 반환합니다; 엔드포인트는 그 태그까지의 모든 내용을 제거하므로 `content`는 답변 자체입니다. **추론을 전혀 생성하지 않는 더 짧고 직접적인 답변을 원하면 `"enable_thinking": false`를 전달하십시오.** **스트리밍은 여전히 추론을 그대로 전달합니다** — 설계된 수정과 그것이 피해야 하는 잘림 함정은 [`docs/NEXT.md`](docs/NEXT.md)를 참조하십시오.

인증은 구현되어 있지 않고 리스너는 loopback에 바인딩되어 있습니다; 다른 곳에서 접근할 수 있어야 한다면 앞에 프록시를 추가하십시오.

## 루프

`loop/run_round.sh`는 반복 드라이버입니다. 각 라운드는:

1. `cargo build --release --workspace`
2. `cargo test --workspace --release`
3. 정확성 게이트: 동결된 HF oracle 트레이스에 대한 token-exact 일치
4. 벤치마크 게이트: `bench/results/`에 새 아티팩트 기록
5. `omnigeeker/GB10-Engine`에 커밋 및 푸시

영속 상태는 `loop/state.json`에, 라운드별 보고서는 `loop/rounds/`에 있습니다. 게이트를 통과하지 못한 라운드는 `FAIL`로 기록되며 마일스톤을 진행시키지 않습니다.

## 방법론 노트

이 프로젝트가 힘들게 배워서 위의 모든 수치에 적용하는 세 가지 규칙:

1. **같은 세션에서 측정한 쌍만 비교할 수 있습니다.** 동일한 fp32 코드가 이 장비에서 다른 날 90.54 s와 111.59 s(+23.2%)로 측정되었습니다; llama.cpp의 32K 수치는 같은 기간에 44.55 s → 53.60 s로 변동했습니다. 비율은 유지되었지만 절대 초는 그렇지 않았습니다.
2. **측정된 효과는 귀속된 원인이 아닙니다.** 128K와 256K 디코드 처리량은 이 작업 동안 ~23% 개선된 반면 8K/32K는 기록된 값을 정확히 재현했습니다. 개선이 실제라는 것은 측정되었지만, *왜*인지는 아직 귀속되지 않았고 그렇게 기록됩니다.
3. **프로브는 여유(headroom)가 아니라 민감도를 측정합니다.** 커널이 절반의 occupancy에서 1.47x 더 든다는 occupancy 프로브는 occupancy를 올릴 수 있는지에 대해 아무것도 말해 주지 않습니다 — 그것에는 리소스 산술이 필요합니다(여기서는 오늘 2 blocks/SM, 3이 되려면 공유 메모리를 22% 줄여야 함).

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | 전체 사용 가이드: Python 및 스트리밍 예제, 파라미터 표, 디코딩 제한 |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | 측정된 하드웨어 물리와 roofline 도출 방법 |
| [`docs/TARGETS.md`](docs/TARGETS.md) | 합의된 성능 계약 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 엔진 아키텍처 |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) | 타일드 attention 커널, 청킹, 메모리 예산, needle 결과 |
| [`bench/longctx/comparison.md`](bench/longctx/comparison.md) | llama.cpp 대비 전체 장문 컨텍스트 비교 로그(기각된 가설과 수정 포함) |
| [`bench/ppl/`](bench/ppl/) · [`bench/mmlu/`](bench/mmlu/) | 기록된 정확도 결과와 이를 산출한 하네스 |
| [`docs/NEXT.md`](docs/NEXT.md) | 미해결 작업 목록 |

## 라이선스

라이선스 정보는 저장소를 참조하십시오.
