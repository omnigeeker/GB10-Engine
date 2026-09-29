# GB10-Engine
[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · **Deutsch** · [Русский](README.ru.md) · [Português](README.pt-BR.md)

Eine von Grund auf neu entwickelte, reine Rust-Inferenz-Engine für **Qwen3.8-27B (NVFP4)** auf der
**NVIDIA DGX Spark GB10** (`sm_121`, CUDA 13), im direkten Vergleich (head to head) gegen
**llama.cpp** mit denselben Gewichten, derselben Maschine und demselben Harness.

- Benutzerdefinierte CUDA-Kernel, für `sm_121` zu PTX kompiliert, keine Inferenz-Laufzeitumgebung eines Anbieters.
- Gated DeltaNet + Full-Attention-Hybrid, 64 Layer, nativer **262,144-Token**-Kontext.
- OpenAI- **und** Anthropic-kompatible HTTP-Endpunkte, Streaming, kontinuierliches Batching.
- Jede Zahl in dieser README ist aus einem Skript in diesem Repository reproduzierbar.

---

## Benchmark: GB10-Engine vs llama.cpp

**Dies ist der Abschnitt, den man lesen sollte.** Dieselbe Maschine, derselbe NVFP4-Checkpoint
(gb10-engine liest den `modelopt`-NVFP4-Checkpoint; llama.cpp liest ein GGUF, das aus denselben
Gewichten konvertiert wurde), dieselben Prompts, derselbe Client, strikt sequenzielle Läufe, damit
kein Kontrahent die GPU für sich allein hat.

### Testaufbau

| | |
|---|---|
| Hardware | NVIDIA DGX Spark **GB10**, `sm_121`, 48 SMs, 121.7 GiB unified LPDDR5X |
| Gemessene Lesebandbreite | **228 GB/s** (siehe [`docs/PHYSICS.md`](docs/PHYSICS.md)) |
| gb10-engine | `./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144` |
| llama.cpp | `llama-server -m models/Qwen3.8-27B-NVFP4.gguf -c 262144 -ngl 99 -fa on` |
| Client | [`bench/longctx/ttft.py`](bench/longctx/ttft.py) — identische Anfrage zweimal pro Versuch, eindeutiger führender Marker, damit „warm“ kein falscher Treffer sein kann |
| Dekodierung | greedy, `max_tokens 128` |
| Präfix-Cache | bei beiden aktiviert (das macht **warm TTFT** aussagekräftig) |

**Definitionen der Metriken**, denn „TTFT“ verbirgt eine echte Frage:

- **Cold TTFT** — der Präfix wurde noch nie gesehen, daher wird der gesamte Prompt vorbefüllt (prefill).
  Dies ist die Messung der Prefill-Rechenlast.
- **Warm TTFT** — dieselbe Anfrage erneut, sodass ein Präfix-Cache den Prefill überspringen kann. Dies ist
  die Cache-Treffer-Messung.
- **OTPS** — Ausgabe-Token pro Sekunde während der Dekodierung.

> **Nur Paare aus derselben Sitzung sind vergleichbar.** Identischer Code wurde auf dieser Maschine an
> verschiedenen Tagen **um 23% abweichend** gemessen (siehe „Methodik-Hinweise“). Jedes Verhältnis unten
> stammt aus einem Paar, das in einer Sitzung gemessen wurde; jede Zelle hat mindestens zwei unabhängige
> Messungen.

### Dekodier-Durchsatz (OTPS) — gb10-engine gewinnt 3 von 4

| Kontext | gb10-engine | llama.cpp | Ergebnis |
|---|---|---|---|
| 8K | **8.91** | 7.26 | **1.23x schneller** |
| 32K | **7.14** | 6.865 | **1.04x schneller** |
| 128K | **5.29** | 4.75 | **1.11x schneller** |
| 256K | 3.665 | **3.86** | 1.05x langsamer |

### TTFT — gb10-engine gewinnt jede Warm-Zelle und liegt bei jeder Cold-Zelle zurück

| Kontext | Cold TTFT (gb10 / llama.cpp) | Warm TTFT (gb10 / llama.cpp) |
|---|---|---|
| **8K** | 10.35 s / **9.50 s** — 1.09x langsamer | **0.03 s** / 0.223 s — **7.4x schneller** |
| **32K** | 80.00 s / **44.55 s** — 1.80x langsamer | **0.05 s** / 0.29 s — **5.8x schneller** |
| **128K** | 890.54 s / **275.04 s** — 3.24x langsamer | **0.14 s** / 0.48 s — **3.4x schneller** |
| **256K** | 3295.45 s / **726.22 s** — 4.54x langsamer | **0.28 s** / 0.66 s — **2.36x schneller** |

**Lesen Sie diese Tabelle ehrlich: Es ist eine geteilte Entscheidung.**

- **Warm TTFT: 4 von 4 gewonnen**, um das 2.4x- bis 7.8x-Fache. Sobald ein Präfix im Cache liegt, nimmt
  diese Engine die Arbeit in 0.03–0.28 s wieder auf, während llama.cpp 0.22–0.66 s benötigt. Für Agent-,
  RAG- und Multi-Turn-Workloads, die einen langen Präfix erneut senden, ist dies die Metrik, die die
  Wall-Clock-Zeit dominiert.
- **OTPS: 3 von 4 gewonnen**, um das 1.04x- bis 1.23x-Fache. Die 256K-Zelle liegt 1.05x zurück.
- **Cold TTFT: 0 von 4 gewonnen.** Diese Engine liegt bei 8K um 1.09x und bei 256K um 4.54x zurück. **Dies
  ist das offene Problem**, und es ist Arithmetik: Die 16 Full-Attention-Layer verrichten quadratische
  Prefill-Arbeit, und der Prefill-Attention-Kernel läuft bei ~8% der fp16-Spitzenleistung der CUDA-Cores
  des Chips.

**Beim Cold TTFT erzielter Fortschritt in der aktuellen Arbeitsrunde** (auf dem Server gemessen,
abgesichert durch die Korrektheitsprüfung `chunked-prefill`):

| Kontext | Cold TTFT vorher | Cold TTFT jetzt | Verbesserung |
|---|---|---|---|
| 8K | 14.93 s | **10.35 s** | **1.43x** |
| 32K | 90.54 s | **80.00 s** | **1.13x** |
| 128K | 1134.63 s | **890.54 s** | **1.28x** |
| 256K | 4138.39 s | **3295.45 s** | **1.25x** |

### Genauigkeit — gb10-engine ist bei denselben Gewichten genauer als llama.cpp

**Perplexity** (WikiText-2, `n_ctx 512`, 580 Fenster, 147,900 Vorhersagen — dieselbe Konfiguration wie
die historische Baseline):

| Implementierung | Gewichte | Perplexity | vs BF16 |
|---|---|---|---|
| **gb10-engine** | **NVFP4** | **7.1006** | +0.71% |
| llama.cpp | NVFP4 (dasselbe GGUF) | 7.2088 | +2.24% |
| transformers (Referenz) | BF16 (Basismodell) | **7.0506** | — |

**MMLU** (3,240 Fragen, 4-Wege-Übereinstimmung):

| Implementierung | Gewichte | MMLU | Korrekt |
|---|---|---|---|
| transformers (Referenz) | BF16 | **80.74%** | 2616/3240 |
| **gb10-engine** | **NVFP4** | **80.34%** | **2603/3240** |
| torch, NVFP4 dequantisiert | NVFP4 | 80.46% | 2607/3240 |
| llama.cpp | NVFP4 (dasselbe GGUF) | 79.88% | 2588/3240 |

**Warum diese Zahlen belegen, dass der Quantisierungspfad korrekt ist:**

- gb10-engine ist bei der Perplexity **1.5% besser als llama.cpp** und bei MMLU **+0.46 pp besser**, und
  zwar mit den *selben* NVFP4-Gewichten. Die Engine verliert also keine Genauigkeit gegenüber der
  ausgereiften Implementierung.
- Sie liegt bei der Perplexity **0.71% über BF16** und bei MMLU **0.40 pp unter BF16** — die erwarteten
  Kosten von 4-Bit-Gewichten für Inferenz ohne Aktivierungen/Gradienten, in derselben Richtung und
  Größenordnung wie die unabhängige dequantisierte torch-Referenz.
- Der Harness **gleicht den Tokenizer gegen `llama-tokenize` ab und meldet 0 Abweichungen über
  297,054 Token**, sodass kein Genauigkeitsunterschied hier ein Tokenisierungsartefakt ist.
- Die Korrektheit wird außerdem pro Runde geprüft: token-exakter Abgleich gegen eingefrorene
  HuggingFace-Oracle-Traces (64 Layer, greedy) sowie eine Batch-Paritätsprüfung.

### Bekanntes Problem: Long-Context-Retrieval ist derzeit nicht zertifiziert

Die Needle-in-a-Haystack-Validierung ([`bench/longctx/run_validation.sh`](bench/longctx/run_validation.sh))
verzeichnete **3/3 bei 32K, 1/1 bei 128K, 1/1 bei 256K**. **Der identische 32K-Durchlauf ergibt jetzt 0/3
bei gb10-engine *und* 0/3 bei llama.cpp**, sodass der Fehler nachverfolgt wurde und die Engine nicht
betroffen war:

- die Fehler sind deterministisch (`finish_reason: stop`, `completion_tokens: 0` — EOS als erstes Token);
- sie sind in der *Länge* unregelmäßig, nicht monoton, was nicht die Form einer KV-Präzisionsregression ist;
- sie sind **identisch mit `temperature: 0`**, mit `--no-prefix-cache`, mit `PREFILL_CHUNK` 2048
  *und* 8192 sowie mit `enable_thinking` weggelassen/false/true;
- **und llama.cpp scheitert an denselben Prompts.**

**Fazit: Die Präzision bei den Standardmetriken ist normal; Long-Context-Retrieval wird von diesem Harness
heute weder zertifiziert noch widerlegt.** Es wird offen verfolgt statt ausgelassen. Vollständige
Ausführung in [`bench/longctx/comparison.md`](bench/longctx/comparison.md).

### Selbst reproduzieren

```bash
# build
cargo build --release --workspace

# perplexity (gb10-engine) -- compares to the recorded 7.0988 / llama.cpp 7.2088 / BF16 7.0506
./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 580

# correctness + benchmark gate (build, tests, token-exact oracle, batch parity, attn tiling, bench)
loop/run_round.sh
```

**Führen Sie eine servergestützte Messung und das Gate nicht gleichzeitig aus** — das Gate startet und
stoppt Server auf Port 8080 und beendet den Server mitten in der Anfrage.

---

## Status

| Meilenstein | Status |
|---|---|
| M0 Grundlage: Config, safetensors, Tokenizer, Chat-Template | **fertig** |
| M1 CUDA-Backend + Roofline-GEMV | **fertig** |
| M2 Layer-Kernel (Gated DeltaNet, Attention, MLP, NVFP4/FP8) | **fertig** |
| M3 Vollständiger Forward-Pass + Greedy-Decode-Parität | **fertig** (token-exakt, 64 Layer, 16/16) |
| M4 MTP spekulative Dekodierung | **fertig** (token-exakt; 44/48 Akzeptanz) |
| M5 Paged KV + kontinuierliches Batching | **fertig** (16 gleichzeitig) |
| M6 OpenAI- + Anthropic-Endpunkte | **fertig** (beide Protokolle + Streaming) |
| M7 Roofline-Optimierung | **in Arbeit** |
| M8 Direktvergleich mit llama.cpp | **fertig** — OTPS 3/4 gewonnen, Warm-TTFT 4/4 gewonnen, Cold-TTFT 0/4 |

### Wo die Performance gegenüber der Hardware-Roofline steht

> **Lesen Sie zuerst [`docs/PHYSICS.md`](docs/PHYSICS.md).** Die ursprüngliche Anforderung verlangte
> 100 tok/s für einen einzelnen Stream. Auf dieser Maschine gemessen hält GB10 **228 GB/s** Lesebandbreite
> aufrecht, und dieser Checkpoint erfordert **17.608 GB** Gewichtsverkehr pro Token, sodass die
> Single-Stream-Roofline **12.95 tok/s** und die Untergrenze pro Schritt **77.2 ms** beträgt. Das Ziel von
> 100 tok/s erfordert ~1.76 TB/s — **das 7.7x-Fache der gemessenen Bandbreite** —, sodass keine
> Implementierung es erreichen kann. Der vereinbarte Vertrag in
> [`docs/TARGETS.md`](docs/TARGETS.md) spiegelt wider, wozu die Hardware tatsächlich in der Lage ist.

Die verbleibende Cold-Prefill-Lücke ist ein einziger Kernel: Die getilete Prefill-Attention läuft mit
**3.12 TFLOP/s, ~8.4% der fp16-Spitzenleistung der CUDA-Cores**, und ihr Instruktionsstrom ist nur zu
42.4% Arithmetik, weshalb sie issue-bound ist. Um die Cold-TTFT-Lücke bei langem Kontext zu schließen,
ist eine `mma.sync`-Tensor-Core-Neufassung dieses Kernels erforderlich; der Plan und die erforderlichen
Beschleunigungen (1.29x bei 8K, 5.98x bei 32K, 8.40x bei 128K, 8.82x bei 256K) stehen in
[`bench/longctx/comparison.md`](bench/longctx/comparison.md).

## Struktur

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

## Schnellstart

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

Das Laden der Gewichte dauert etwa **40 s**, danach wird auf `127.0.0.1:8080` bedient.

## Modell

`nv-community/Qwen3.8-27B-NVFP4`, heruntergeladen mit der ModelScope-CLI:

```bash
modelscope download --model nv-community/Qwen3.8-27B-NVFP4 \
    --local_dir models/Qwen3.8-27B-NVFP4
```

`models/` ist ein Symlink auf das heruntergeladene Checkpoint-Verzeichnis und wird nicht von git verfolgt.

Architektur: `qwen3_5`, 64 Layer (48 Gated-DeltaNet-Linear-Attention + 16 Full-Attention in jedem 4.
Layer), hidden 5120, 24 Query- / 4 KV-Heads (GQA-Gruppe 6), head_dim 256, partielles RoPE (0.25,
mRoPE verschachtelt), vocab 248320, **1 MTP-Layer**, nativ **262,144** Positionen. Die Quantisierung ist
modelopt MIXED_PRECISION: NVFP4 (Gruppe 16) für alle MLP und `lm_head`, FP8 für jede
Attention-Projektion, und MTP-/Vision-Gewichte bleiben unquantisiert. Größe auf der Platte 21.921 GB.

## Verwendung des lokalen Endpunkts

Der Server spricht sowohl das OpenAI- als auch das Anthropic-Wire-Protokoll auf `127.0.0.1:8080`.

| Route | Protokoll | Hinweise |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | unterstützt `stream`, `max_tokens`, `enable_thinking` |
| `POST /v1/completions` | OpenAI | Legacy-Text-Completion; `prompt` muss ein einzelner String sein |
| `POST /v1/messages` | Anthropic | unterstützt `stream`, `max_tokens` |
| `GET /v1/models` | beide | gibt die ID des geladenen Modells zurück |
| `GET /health` | — | Lebendigkeit; damit auf den Start warten |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":16}'
```

### Kontextlänge und Nebenläufigkeit

Das Kontextfenster ist `--ctx` (**Standard 32768**) bis zu den **262144** Positionen des Checkpoints, und
`--concurrency` legt fest, für wie viele Sequenzen der KV-Cache dimensioniert wird. Die Nebenläufigkeit
ist standardmäßig so hoch, wie ein KV-Budget von 40 GB abdeckt, begrenzt auf 16, sodass ein langer
Kontext sie automatisch senkt:

```sh
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144
# context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB
```

> **Für jeden Long-Context-Benchmark ist `--ctx` zwingend erforderlich.** Ohne diese Angabe läuft der
> Server mit 32768, und ein längerer Prompt liefert **überhaupt keinen Inhalt** statt eines Fehlers —
> ein stiller Fehler, der wie ein Client-Bug aussieht.

**Nebenläufigkeit.** Der Server bündelt bis zu 16 gleichzeitige Anfragen in einem einzigen Forward-Pass
und sammelt sie innerhalb eines Fensters von 25 ms. Setzen Sie `GB10_BATCH_LOG=1`, um die Gruppengröße
bei jedem Schritt zu protokollieren — 16 parallele Anfragen sollten `batch of 16` protokollieren. Bei
identischen Prompts liefern alle 16 byte-identische Antworten.

**Antworten werden vom Reasoning-Block des Modells befreit.** `enable_thinking` ist standardmäßig `true`
(entsprechend dem Checkpoint), daher setzt das Template das öffnende ` thinking` in den Prompt und die
Generierung liefert Reasoning, gefolgt von `</think>`; der Endpunkt entfernt alles bis einschließlich
dieses Tags, sodass `content` die Antwort selbst ist. **Übergeben Sie `"enable_thinking": false` für eine
kürzere, direkte Antwort, bei der überhaupt kein Reasoning erzeugt wird.** **Beim Streaming wird das
Reasoning weiterhin durchgereicht** — siehe [`docs/NEXT.md`](docs/NEXT.md) für den geplanten Fix und die
Truncation-Falle, die er vermeiden muss.

Es ist keine Authentifizierung implementiert und der Listener ist an Loopback gebunden; setzen Sie einen
Proxy davor, wenn er von anderer Stelle erreichbar sein muss.

## Die Schleife

`loop/run_round.sh` ist der Iterationstreiber. Jede Runde:

1. `cargo build --release --workspace`
2. `cargo test --workspace --release`
3. Korrektheits-Gate: token-exakter Abgleich gegen eingefrorene HF-Oracle-Traces
4. Benchmark-Gate: frisches Artefakt geschrieben nach `bench/results/`
5. Commit und Push nach `omnigeeker/GB10-Engine`

Der dauerhafte Zustand liegt in `loop/state.json`; Berichte pro Runde in `loop/rounds/`. Eine Runde, die
ein Gate nicht besteht, wird als `FAIL` aufgezeichnet und bringt den Meilenstein nicht voran.

## Methodik-Hinweise

Drei Regeln, die dieses Projekt auf die harte Tour gelernt hat und auf jede Zahl oben anwendet:

1. **Nur Paare aus derselben Sitzung sind vergleichbar.** Identischer fp32-Code wurde auf dieser Maschine
   an verschiedenen Tagen mit 90.54 s und 111.59 s (+23.2%) gemessen; der 32K-Wert von llama.cpp driftete
   im selben Zeitraum von 44.55 s → 53.60 s. Die Verhältnisse blieben stabil, die absoluten Sekunden nicht.
2. **Ein gemessener Effekt ist keine zugeordnete Ursache.** Der Dekodier-Durchsatz bei 128K und 256K
   verbesserte sich über diese Arbeit hinweg um ~23%, während 8K/32K ihre aufgezeichneten Werte exakt
   reproduzierten. Dass die Verbesserung real ist, ist gemessen; *warum*, ist weiterhin nicht zugeordnet,
   und so wird es auch aufgezeichnet.
3. **Eine Sonde misst eine Sensitivität, nicht den Spielraum (headroom).** Eine Occupancy-Sonde, die
   zeigt, dass ein Kernel bei halber Occupancy 1.47x mehr kostet, sagt nichts darüber aus, ob die
   Occupancy erhöht werden kann — dafür ist die Ressourcen-Arithmetik nötig (hier: heute 2 Blöcke/SM,
   für 3 wäre eine Kürzung des Shared Memory um 22% nötig).

## Dokumentation

| Dokument | Inhalt |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | vollständiger Nutzungsleitfaden: Python- und Streaming-Beispiele, Parametertabelle, Dekodiergrenzen |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | gemessene Hardware-Physik und wie die Roofline abgeleitet wurde |
| [`docs/TARGETS.md`](docs/TARGETS.md) | der vereinbarte Performance-Vertrag |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Engine-Architektur |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) | der getilete Attention-Kernel, Chunking, Speicherbudget, Needle-Ergebnisse |
| [`bench/longctx/comparison.md`](bench/longctx/comparison.md) | das vollständige Long-Context-Vergleichsprotokoll gegen llama.cpp, einschließlich verworfener Hypothesen und Korrekturen |
| [`bench/ppl/`](bench/ppl/) · [`bench/mmlu/`](bench/mmlu/) | aufgezeichnete Genauigkeitsergebnisse und die Harnesses, die sie erzeugt haben |
| [`docs/NEXT.md`](docs/NEXT.md) | die offene Arbeitsliste |

## Lizenz

Lizenzinformationen finden Sie im Repository.
