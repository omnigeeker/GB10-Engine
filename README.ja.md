# GB10-Engine
[English](README.md) · [简体中文](README.zh-CN.md) · **日本語** · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Português](README.pt-BR.md)

**Qwen3.8-27B (NVFP4)** を **NVIDIA DGX Spark GB10**（`sm_121`、CUDA 13）上で動かすための、ゼロから書かれた純粋な **Rust** 製推論エンジンです。同じ重み、同じマシン、同じハーネス上で **llama.cpp** と直接比較してベンチマークしています。

- `sm_121` 向けに PTX へコンパイルされた独自 CUDA カーネル。ベンダーの推論ランタイムは使用しません。
- Gated DeltaNet + full attention のハイブリッド、64 層、ネイティブ **262,144 token** コンテキスト。
- OpenAI **および** Anthropic 互換の HTTP エンドポイント、ストリーミング、continuous batching。
- この README のすべての数値は、このリポジトリ内のスクリプトから再現できます。

---

## ベンチマーク: GB10-Engine 対 llama.cpp

**まず読むべきセクションです。** 同じマシン、同じ NVFP4 チェックポイント（gb10-engine は `modelopt` NVFP4 チェックポイントを読み込み、llama.cpp は同じ重みから変換した GGUF を読み込みます）、同じプロンプト、同じクライアントを使用し、どちらの競合相手も GPU を独占しないよう厳密に逐次実行しています。

### テスト環境

| | |
|---|---|
| ハードウェア | NVIDIA DGX Spark **GB10**、`sm_121`、48 SM、121.7 GiB 統合 LPDDR5X |
| 実測読み出し帯域幅 | **228 GB/s**（[`docs/PHYSICS.md`](docs/PHYSICS.md) 参照） |
| gb10-engine | `./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144` |
| llama.cpp | `llama-server -m models/Qwen3.8-27B-NVFP4.gguf -c 262144 -ngl 99 -fa on` |
| クライアント | [`bench/longctx/ttft.py`](bench/longctx/ttft.py) — 試行ごとに同一リクエストを二回送信し、「warm」が偽のヒットにならないよう一意の先頭マーカーを付与 |
| デコード | greedy、`max_tokens 128` |
| プレフィックスキャッシュ | 両方で有効（これが **warm TTFT** を意味のあるものにします） |

**指標の定義** — 「TTFT」という言葉は本当の問いを覆い隠すため:

- **Cold TTFT** — プレフィックスが一度も見られていない状態で、プロンプト全体を prefill します。これは prefill 演算の測定値です。
- **Warm TTFT** — まったく同じリクエストをもう一度送るため、プレフィックスキャッシュが prefill をスキップできます。これはキャッシュヒットの測定値です。
- **OTPS** — デコード中の毎秒出力 token 数。

> **同一セッション内のペアのみが比較可能です。** 同一のコードが、このマシン上で別の日に **23% の差**で測定されました（「方法論上の注意」を参照）。以下のすべての比率は、一つのセッションで測定されたペアから得られています。各セルには少なくとも二回の独立した測定があります。

### デコードスループット (OTPS) — gb10-engine が 4 本中 3 本で勝利

| コンテキスト | gb10-engine | llama.cpp | 結果 |
|---|---|---|---|
| 8K | **8.91** | 7.26 | **1.23x 高速** |
| 32K | **7.14** | 6.865 | **1.04x 高速** |
| 128K | **5.29** | 4.75 | **1.11x 高速** |
| 256K | 3.665 | **3.86** | 1.05x 低速 |

### TTFT — gb10-engine はすべての warm セルで勝利し、すべての cold セルで後れを取る

| コンテキスト | Cold TTFT (gb10 / llama.cpp) | Warm TTFT (gb10 / llama.cpp) |
|---|---|---|
| **8K** | 10.35 s / **9.50 s** — 1.09x 低速 | **0.03 s** / 0.223 s — **7.4x 高速** |
| **32K** | 80.00 s / **44.55 s** — 1.80x 低速 | **0.05 s** / 0.29 s — **5.8x 高速** |
| **128K** | 890.54 s / **275.04 s** — 3.24x 低速 | **0.14 s** / 0.48 s — **3.4x 高速** |
| **256K** | 3295.45 s / **726.22 s** — 4.54x 低速 | **0.28 s** / 0.66 s — **2.36x 高速** |

**この表は正直に読みましょう。結論は二分されています。**

- **Warm TTFT: 4 本中 4 本勝利**、2.4x ～ 7.8x。プレフィックスが一度キャッシュされれば、このエンジンは 0.03～0.28 s で再開しますが、llama.cpp は 0.22～0.66 s を要します。長いプレフィックスを再送するエージェント、RAG、マルチターンのワークロードでは、実時間を支配するのはこの指標です。
- **OTPS: 4 本中 3 本勝利**、1.04x ～ 1.23x。256K のセルは 1.05x 後れを取っています。
- **Cold TTFT: 4 本中 0 本勝利。** このエンジンは 8K で 1.09x、256K で 4.54x 後れを取っています。**これが未解決の問題です。** そしてそれは算数の問題です。16 個の full attention 層が二次の prefill 処理を行い、prefill attention カーネルはこのパーツの fp16 CUDA-core ピークの約 8% で動作しています。

**今回の作業ラウンドで達成した Cold TTFT の進展**（サーバー上で測定、`chunked-prefill` 正しさチェックでゲート）:

| コンテキスト | 以前の Cold TTFT | 現在の Cold TTFT | 改善 |
|---|---|---|---|
| 8K | 14.93 s | **10.35 s** | **1.43x** |
| 32K | 90.54 s | **80.00 s** | **1.13x** |
| 128K | 1134.63 s | **890.54 s** | **1.28x** |
| 256K | 4138.39 s | **3295.45 s** | **1.25x** |

### 精度 — gb10-engine は同じ重みで llama.cpp より高精度

**perplexity**（WikiText-2、`n_ctx 512`、580 ウィンドウ、147,900 予測 — 過去のベースラインと同じ構成）:

| 実装 | 重み | perplexity | BF16 比 |
|---|---|---|---|
| **gb10-engine** | **NVFP4** | **7.1006** | +0.71% |
| llama.cpp | NVFP4 (同じ GGUF) | 7.2088 | +2.24% |
| transformers (リファレンス) | BF16 (ベースモデル) | **7.0506** | — |

**MMLU**（3,240 問、4 択一致）:

| 実装 | 重み | MMLU | 正解数 |
|---|---|---|---|
| transformers (リファレンス) | BF16 | **80.74%** | 2616/3240 |
| **gb10-engine** | **NVFP4** | **80.34%** | **2603/3240** |
| torch、NVFP4 逆量子化 | NVFP4 | 80.46% | 2607/3240 |
| llama.cpp | NVFP4 (同じ GGUF) | 79.88% | 2588/3240 |

**これらの数値が量子化の経路は正しいと言える理由:**

- gb10-engine は *同じ* NVFP4 重みを使って、perplexity で **llama.cpp より 1.5% 良好**、**MMLU で +0.46 pp 良好**です。このエンジンは成熟した実装と比べて精度を落としていません。
- perplexity で **BF16 より 0.71% 上**、MMLU で **BF16 より 0.40 pp 下**に位置します — 活性化/勾配のない推論における 4-bit 重みの想定されるコストであり、独立した torch 逆量子化リファレンスと同じ方向・同じ大きさです。
- ハーネスは **tokenizer を `llama-tokenize` とクロスチェックし、297,054 token にわたって不一致 0 件を報告**しているため、ここでの精度差が tokenization のアーティファクトであることはありません。
- 正しさはラウンドごとにもゲートされています: 凍結された HuggingFace オラクルトレース（64 層、greedy）に対する token 完全一致、およびバッチ等価性チェック。

### 既知の問題: 長コンテキスト検索は現在認定されていない

needle-in-a-haystack 検証（[`bench/longctx/run_validation.sh`](bench/longctx/run_validation.sh)）は **32K で 3/3、128K で 1/1、256K で 1/1** を記録しました。**同一の 32K レグを今再実行すると、gb10-engine で 0/3、*かつ* llama.cpp でも 0/3** となるため、失敗は追跡調査され、このエンジンは関与していないと判断されました:

- 失敗は決定的です（`finish_reason: stop`、`completion_tokens: 0` — 最初の token が EOS）;
- それらは *長さ* において不規則で単調ではなく、KV 精度の回帰の形ではありません;
- `temperature: 0` でも、`--no-prefix-cache` でも、`PREFILL_CHUNK` 2048 *および* 8192 でも、`enable_thinking` を省略/false/true にしても **同一**です;
- **そして llama.cpp も同じプロンプトで失敗します。**

**結論: 標準指標での精度は正常です。長コンテキスト検索は、今日のこのハーネスによって認定も反証もされていません。** これは省略せずに公然と追跡しています。詳細な記述は [`bench/longctx/comparison.md`](bench/longctx/comparison.md) にあります。

### 自分で再現する

```bash
# build
cargo build --release --workspace

# perplexity (gb10-engine) -- compares to the recorded 7.0988 / llama.cpp 7.2088 / BF16 7.0506
./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 580

# correctness + benchmark gate (build, tests, token-exact oracle, batch parity, attn tiling, bench)
loop/run_round.sh
```

**サーバーを使う測定とゲートを同時に実行しないでください** — ゲートはポート 8080 でサーバーを起動・停止するため、リクエストの途中でサーバーを強制終了します。

---

## ステータス

| マイルストーン | 状態 |
|---|---|
| M0 基盤: config、safetensors、tokenizer、chat template | **完了** |
| M1 CUDA バックエンド + roofline GEMV | **完了** |
| M2 層カーネル（Gated DeltaNet、attention、MLP、NVFP4/FP8） | **完了** |
| M3 完全な forward + greedy デコード等価性 | **完了**（token 完全一致、64 層、16/16） |
| M4 MTP 投機的デコード | **完了**（token 完全一致、44/48 受理） |
| M5 ページド KV + continuous batching | **完了**（16 並行） |
| M6 OpenAI + Anthropic エンドポイント | **完了**（両プロトコル + ストリーミング） |
| M7 Roofline 最適化 | **進行中** |
| M8 llama.cpp 直接対決 | **完了** — OTPS 3/4 勝利、warm TTFT 4/4 勝利、cold TTFT 0/4 |

### ハードウェアの roofline に対する性能の現状

> **まず [`docs/PHYSICS.md`](docs/PHYSICS.md) を読んでください。** 当初の要求はシングルストリームで 100 tok/s でした。このマシンで測定すると、GB10 は読み出し帯域幅 **228 GB/s** を維持し、このチェックポイントは token あたり **17.608 GB** の重みトラフィックを必要とするため、シングルストリームの roofline は **12.95 tok/s**、ステップあたりの下限は **77.2 ms** です。100 tok/s の目標には約 1.76 TB/s が必要で、これは**実測帯域幅の 7.7x**であるため、どの実装でも到達できません。[`docs/TARGETS.md`](docs/TARGETS.md) で合意された契約は、ハードウェアが実際に達成できる内容を反映しています。

残っている cold-prefill のギャップは一つのカーネルに集約されます。タイル化された prefill attention は **3.12 TFLOP/s、fp16 CUDA-core ピークの約 8.4%** で動作し、その命令ストリームは 42.4% しか演算でないため、issue バウンドです。長コンテキストの cold-TTFT ギャップを埋めるには、そのカーネルを `mma.sync` tensor core で書き直す必要があります。計画と必要な高速化（8K で 1.29x、32K で 5.98x、128K で 8.40x、256K で 8.82x）は [`bench/longctx/comparison.md`](bench/longctx/comparison.md) にあります。

## ディレクトリ構成

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

## クイックスタート

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

重みの読み込みには約 **40 s** かかり、その後 `127.0.0.1:8080` でサービスを提供します。

## モデル

`nv-community/Qwen3.8-27B-NVFP4` を ModelScope CLI で取得します:

```bash
modelscope download --model nv-community/Qwen3.8-27B-NVFP4 \
    --local_dir models/Qwen3.8-27B-NVFP4
```

`models/` はダウンロードしたチェックポイントディレクトリへのシンボリックリンクであり、git では追跡されていません。

アーキテクチャ: `qwen3_5`、64 層（48 の Gated-DeltaNet 線形 attention + 4 層ごとの 16 の full attention）、hidden 5120、24 query / 4 KV heads（GQA グループ 6）、head_dim 256、部分 RoPE（0.25、mRoPE インターリーブ）、vocab 248320、**MTP 層 1 つ**、ネイティブ **262,144** 位置。量子化は modelopt MIXED_PRECISION: すべての MLP と `lm_head` に NVFP4（グループ 16）、すべての attention 射影に FP8、MTP/vision の重みは非量子化のままです。ディスク上のサイズは 21.921 GB。

## ローカルエンドポイントの使用

サーバーは `127.0.0.1:8080` で OpenAI と Anthropic の両方のワイヤプロトコルを話します。

| ルート | プロトコル | 備考 |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | `stream`、`max_tokens`、`enable_thinking` をサポート |
| `POST /v1/completions` | OpenAI | レガシーなテキスト補完。`prompt` は単一の文字列である必要があります |
| `POST /v1/messages` | Anthropic | `stream`、`max_tokens` をサポート |
| `GET /v1/models` | 両方 | 読み込まれたモデル id を返します |
| `GET /health` | — | 生存確認。起動待ちにこれを使用します |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":16}'
```

### コンテキスト長と並行性

コンテキストウィンドウは `--ctx`（**デフォルト 32768**）で、チェックポイントの **262144** 位置まで設定できます。`--concurrency` は KV キャッシュをいくつのシーケンス分確保するかを設定します。並行数はデフォルトで 40 GB の KV 予算が賄える数、上限 16 であるため、長いコンテキストでは自動的に下がります:

```sh
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144
# context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB
```

> **長コンテキストのベンチマークでは、`--ctx` は必須です。** これを指定しないとサーバーは 32768 で動作し、より長いプロンプトはエラーではなく**コンテンツをまったく返しません** — クライアントのバグに見える静かな失敗です。

**並行性。** サーバーは最大 16 の並行リクエストを 25 ms のウィンドウ内で収集し、単一の forward パスにバッチ処理します。各ステップのグループサイズをログに出すには `GB10_BATCH_LOG=1` を設定します — 16 の並行リクエストなら `batch of 16` とログされるはずです。同一のプロンプトでは、16 件すべてがバイト単位で同一の回答を返します。

**回答からはモデルの推論ブロックが除去されます。** `enable_thinking` のデフォルトは `true`（チェックポイントに一致）であるため、テンプレートはプロンプトに開始タグ ` thinking` を入れ、生成は推論の後に `</think>` を返します。エンドポイントはそのタグまでのすべてを削除するため、`content` は回答そのものになります。**推論をまったく生成せず、より短く直接的な回答を得るには `"enable_thinking": false` を渡してください。** **ストリーミングは依然として推論をそのまま通します** — 設計上の修正と、それが避けなければならない切り詰めの罠については [`docs/NEXT.md`](docs/NEXT.md) を参照してください。

認証は実装されておらず、リスナーはループバックにバインドされています。他から到達可能にする必要がある場合は、その前にプロキシを追加してください。

## ループ

`loop/run_round.sh` はイテレーションのドライバです。各ラウンド:

1. `cargo build --release --workspace`
2. `cargo test --workspace --release`
3. 正しさゲート: 凍結された HF オラクルトレースに対する token 完全一致
4. ベンチマークゲート: `bench/results/` に新しい成果物を書き出し
5. `omnigeeker/GB10-Engine` にコミットしてプッシュ

永続状態は `loop/state.json` にあり、ラウンドごとのレポートは `loop/rounds/` にあります。ゲートに失敗したラウンドは `FAIL` として記録され、マイルストーンは進みません。

## 方法論上の注意

このプロジェクトが苦労して学び、上記のすべての数値に適用している三つのルール:

1. **同一セッション内のペアのみが比較可能です。** 同一の fp32 コードが、このマシン上の別の日に 90.54 s と 111.59 s（+23.2%）として測定されました。llama.cpp の 32K の数値は同じ期間に 44.55 s → 53.60 s へ変動しました。比率は保たれましたが、絶対秒数は保たれませんでした。
2. **測定された効果は、原因が特定されたことを意味しません。** 128K と 256K のデコードスループットはこの作業を通じて約 23% 改善しましたが、8K/32K は記録された値を正確に再現しました。改善が実在することは測定されていますが、*なぜ* かはまだ特定されておらず、そのように記録されています。
3. **プローブが測るのは感度であり、ヘッドルームではありません。** カーネルが占有率半分で 1.47x のコストになることを示す占有率プローブは、占有率を上げられるかどうかについては何も語りません — それにはリソースの算数が必要です（ここでは: 現在 2 blocks/SM、3 にするには共有メモリを 22% 削減する必要があります）。

## ドキュメント

| ドキュメント | 内容 |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | 完全な使用ガイド: Python とストリーミングの例、パラメータ表、デコードの制限 |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | 実測されたハードウェア物理と roofline の導出方法 |
| [`docs/TARGETS.md`](docs/TARGETS.md) | 合意された性能契約 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | エンジンのアーキテクチャ |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) | タイル化 attention カーネル、チャンキング、メモリ予算、needle の結果 |
| [`bench/longctx/comparison.md`](bench/longctx/comparison.md) | llama.cpp との完全な長コンテキスト比較ログ（棄却された仮説と修正を含む） |
| [`bench/ppl/`](bench/ppl/) · [`bench/mmlu/`](bench/mmlu/) | 記録された精度結果と、それを生成したハーネス |
| [`docs/NEXT.md`](docs/NEXT.md) | 未解決の作業リスト |

## ライセンス

ライセンス情報についてはリポジトリを参照してください。
