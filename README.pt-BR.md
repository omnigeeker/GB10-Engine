# GB10-Engine
[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · **Português**

Um motor de inferência em Rust puro, escrito do zero, para o **Qwen3.8-27B (NVFP4)** no
**NVIDIA DGX Spark GB10** (`sm_121`, CUDA 13), comparado frente a frente com o
**llama.cpp** usando os mesmos pesos, a mesma máquina e o mesmo harness.

- Kernels CUDA personalizados compilados para PTX visando `sm_121`, sem runtime de inferência de fornecedor.
- Híbrido de Gated DeltaNet + full attention, 64 camadas, contexto nativo de **262,144 tokens**.
- Endpoints HTTP compatíveis com OpenAI **e** Anthropic, streaming, continuous batching.
- Todos os números deste README são reprodutíveis a partir de um script neste repositório.

---

## Benchmark: GB10-Engine vs llama.cpp

**Esta é a seção que você deve ler.** Mesma máquina, mesmo checkpoint NVFP4 (o gb10-engine lê o
checkpoint NVFP4 `modelopt`; o llama.cpp lê um GGUF convertido dos mesmos pesos), mesmos prompts,
mesmo cliente, execuções estritamente sequenciais para que nenhum dos concorrentes tenha a GPU
só para si.

### Configuração do teste

| | |
|---|---|
| Hardware | NVIDIA DGX Spark **GB10**, `sm_121`, 48 SMs, 121.7 GiB de LPDDR5X unificada |
| Largura de banda de leitura medida | **228 GB/s** (veja [`docs/PHYSICS.md`](docs/PHYSICS.md)) |
| gb10-engine | `./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144` |
| llama.cpp | `llama-server -m models/Qwen3.8-27B-NVFP4.gguf -c 262144 -ngl 99 -fa on` |
| Cliente | [`bench/longctx/ttft.py`](bench/longctx/ttft.py) — requisição idêntica duas vezes por tentativa, marcador inicial único para que "warm" não possa ser um falso acerto |
| Decodificação | greedy, `max_tokens 128` |
| Cache de prefixo | habilitado em ambos (é isso que torna o **warm TTFT** significativo) |

**Definições das métricas**, porque "TTFT" esconde uma questão real:

- **Cold TTFT** — o prefixo nunca foi visto, então todo o prompt passa por prefill. Esta é a
  medição de computação de prefill.
- **Warm TTFT** — a requisição idêntica novamente, para que um cache de prefixo possa pular o prefill. Esta é a
  medição de acerto de cache.
- **OTPS** — tokens de saída por segundo durante a decodificação.

> **Somente pares da mesma sessão são comparáveis.** Código idêntico mediu **23% de diferença** nesta máquina em
> dias diferentes (veja "Notas de metodologia"). Toda razão abaixo vem de um par medido em uma única
> sessão; cada célula tem pelo menos duas medições independentes.

### Throughput de decodificação (OTPS) — o gb10-engine vence 3 de 4

| Contexto | gb10-engine | llama.cpp | Resultado |
|---|---|---|---|
| 8K | **8.91** | 7.26 | **1.23x mais rápido** |
| 32K | **7.14** | 6.865 | **1.04x mais rápido** |
| 128K | **5.29** | 4.75 | **1.11x mais rápido** |
| 256K | 3.665 | **3.86** | 1.05x mais lento |

### TTFT — o gb10-engine vence todas as células warm e fica atrás em todas as células cold

| Contexto | Cold TTFT (gb10 / llama.cpp) | Warm TTFT (gb10 / llama.cpp) |
|---|---|---|
| **8K** | 10.35 s / **9.50 s** — 1.09x mais lento | **0.03 s** / 0.223 s — **7.4x mais rápido** |
| **32K** | 80.00 s / **44.55 s** — 1.80x mais lento | **0.05 s** / 0.29 s — **5.8x mais rápido** |
| **128K** | 890.54 s / **275.04 s** — 3.24x mais lento | **0.14 s** / 0.48 s — **3.4x mais rápido** |
| **256K** | 3295.45 s / **726.22 s** — 4.54x mais lento | **0.28 s** / 0.66 s — **2.36x mais rápido** |

**Leia esta tabela com honestidade: é uma decisão dividida.**

- **Warm TTFT: 4 de 4 vencidas**, por 2.4x a 7.8x. Depois que um prefixo está em cache, este motor retoma em
  0.03–0.28 s, enquanto o llama.cpp precisa de 0.22–0.66 s. Para cargas de trabalho de agentes, RAG e múltiplos turnos que reenviam
  um prefixo longo, esta é a métrica que domina o tempo de relógio de parede.
- **OTPS: 3 de 4 vencidas**, por 1.04x a 1.23x. A célula de 256K fica 1.05x atrás.
- **Cold TTFT: 0 de 4 vencidas.** Este motor fica 1.09x atrás em 8K e 4.54x atrás em 256K. **Este é
  o problema em aberto**, e é aritmética: as 16 camadas de full attention fazem trabalho de prefill quadrático
  e o kernel de atenção de prefill roda a ~8% do pico de CUDA-core fp16 da peça.

**Progresso de Cold TTFT obtido na rodada atual de trabalho** (medido no servidor, validado pela
checagem de correção `chunked-prefill`):

| Contexto | Cold TTFT antes | Cold TTFT agora | Melhoria |
|---|---|---|---|
| 8K | 14.93 s | **10.35 s** | **1.43x** |
| 32K | 90.54 s | **80.00 s** | **1.13x** |
| 128K | 1134.63 s | **890.54 s** | **1.28x** |
| 256K | 4138.39 s | **3295.45 s** | **1.25x** |

### Precisão — o gb10-engine é mais preciso que o llama.cpp com os mesmos pesos

**Perplexity** (WikiText-2, `n_ctx 512`, 580 janelas, 147,900 predições — a mesma configuração
da linha de base histórica):

| Implementação | Pesos | Perplexity | vs BF16 |
|---|---|---|---|
| **gb10-engine** | **NVFP4** | **7.1006** | +0.71% |
| llama.cpp | NVFP4 (mesmo GGUF) | 7.2088 | +2.24% |
| transformers (referência) | BF16 (modelo base) | **7.0506** | — |

**MMLU** (3,240 perguntas, concordância de 4 vias):

| Implementação | Pesos | MMLU | Acertos |
|---|---|---|---|
| transformers (referência) | BF16 | **80.74%** | 2616/3240 |
| **gb10-engine** | **NVFP4** | **80.34%** | **2603/3240** |
| torch, NVFP4 desquantizado | NVFP4 | 80.46% | 2607/3240 |
| llama.cpp | NVFP4 (mesmo GGUF) | 79.88% | 2588/3240 |

**Por que esses números indicam que o caminho de quantização está correto:**

- O gb10-engine é **1.5% melhor que o llama.cpp** em perplexity e **+0.46 pp melhor no MMLU**, usando
  os *mesmos* pesos NVFP4. O motor não está perdendo precisão em relação à implementação madura.
- Ele fica **0.71% acima do BF16** em perplexity e **0.40 pp abaixo do BF16** no MMLU — o custo esperado de
  pesos de 4 bits para inferência sem ativações/gradientes, e a mesma direção e magnitude da
  referência independente de torch desquantizado.
- O harness **verifica o tokenizador em cruzamento com o `llama-tokenize` e reporta 0 divergências em
  297,054 tokens**, então nenhuma diferença de precisão aqui é artefato de tokenização.
- A correção também é validada a cada rodada: correspondência exata de tokens com traços oráculo congelados do HuggingFace
  (64 camadas, greedy), além de uma checagem de paridade de lote.

### Problema conhecido: a recuperação em contexto longo não está certificada no momento

A validação needle-in-a-haystack ([`bench/longctx/run_validation.sh`](bench/longctx/run_validation.sh))
registrou **3/3 em 32K, 1/1 em 128K, 1/1 em 256K**. **Repetir a mesma etapa de 32K agora dá 0/3 no
gb10-engine *e* 0/3 no llama.cpp**, então a falha foi rastreada e o motor não foi implicado:

- as falhas são determinísticas (`finish_reason: stop`, `completion_tokens: 0` — EOS como o primeiro token);
- elas são erráticas em *comprimento*, não monotônicas, o que não é a forma de uma regressão de precisão de KV;
- elas são **idênticas com `temperature: 0`**, com `--no-prefix-cache`, com `PREFILL_CHUNK` 2048
  *e* 8192, e com `enable_thinking` omitido/false/true;
- **e o llama.cpp falha nos mesmos prompts.**

**Conclusão: a precisão nas métricas padrão está normal; a recuperação em contexto longo não é certificada
nem refutada por este harness hoje.** Ela é acompanhada abertamente em vez de omitida. Relato completo em
[`bench/longctx/comparison.md`](bench/longctx/comparison.md).

### Reproduza você mesmo

```bash
# build
cargo build --release --workspace

# perplexity (gb10-engine) -- compares to the recorded 7.0988 / llama.cpp 7.2088 / BF16 7.0506
./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 580

# correctness + benchmark gate (build, tests, token-exact oracle, batch parity, attn tiling, bench)
loop/run_round.sh
```

**Não execute uma medição baseada em servidor e o gate ao mesmo tempo** — o gate inicia e para
servidores na porta 8080 e matará o servidor no meio de uma requisição.

---

## Status

| Marco | Estado |
|---|---|
| M0 Fundação: config, safetensors, tokenizer, chat template | **concluído** |
| M1 Backend CUDA + roofline GEMV | **concluído** |
| M2 Kernels de camada (Gated DeltaNet, atenção, MLP, NVFP4/FP8) | **concluído** |
| M3 Forward completo + paridade de decodificação greedy | **concluído** (exato em tokens, 64 camadas, 16/16) |
| M4 Decodificação especulativa MTP | **concluído** (exato em tokens; 44/48 de aceitação) |
| M5 KV paginado + continuous batching | **concluído** (16 concorrentes) |
| M6 Endpoints OpenAI + Anthropic | **concluído** (ambos os protocolos + streaming) |
| M7 Otimização de roofline | **em andamento** |
| M8 Comparação frente a frente com llama.cpp | **concluído** — OTPS 3/4 vencidas, warm TTFT 4/4 vencidas, cold TTFT 0/4 |

### Onde o desempenho está em relação ao roofline de hardware

> **Leia [`docs/PHYSICS.md`](docs/PHYSICS.md) primeiro.** O pedido original era 100 tok/s
> em fluxo único. Medido nesta máquina, o GB10 sustenta **228 GB/s** de largura de banda de leitura e este
> checkpoint exige **17.608 GB** de tráfego de pesos por token, então o roofline de fluxo único é
> **12.95 tok/s** e o piso por passo é **77.2 ms**. A meta de 100 tok/s exigiria ~1.76 TB/s —
> **7.7x a largura de banda medida** — então nenhuma implementação consegue alcançá-la. O contrato acordado em
> [`docs/TARGETS.md`](docs/TARGETS.md) reflete o que o hardware realmente pode fazer.

A lacuna restante de cold prefill é um único kernel: a atenção de prefill com tiling roda a **3.12 TFLOP/s,
~8.4% do pico de CUDA-core fp16**, e seu fluxo de instruções é apenas 42.4% aritmético, então ele é
limitado por emissão (issue-bound). Fechar a lacuna de cold TTFT em contexto longo exige uma reescrita com tensor cores
`mma.sync` desse kernel; o plano e os ganhos de velocidade necessários (1.29x em 8K, 5.98x em 32K, 8.40x em 128K, 8.82x em
256K) estão em [`bench/longctx/comparison.md`](bench/longctx/comparison.md).

## Estrutura

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

## Início rápido

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

Leva cerca de **40 s** para carregar os pesos, depois ele serve em `127.0.0.1:8080`.

## Modelo

`nv-community/Qwen3.8-27B-NVFP4`, baixado com a CLI do ModelScope:

```bash
modelscope download --model nv-community/Qwen3.8-27B-NVFP4 \
    --local_dir models/Qwen3.8-27B-NVFP4
```

`models/` é um symlink para o diretório do checkpoint baixado e não é rastreado pelo git.

Arquitetura: `qwen3_5`, 64 camadas (48 de atenção linear Gated-DeltaNet + 16 de full attention a cada
4ª camada), hidden 5120, 24 heads de query / 4 de KV (grupo GQA 6), head_dim 256, RoPE parcial (0.25,
mRoPE interleaved), vocab 248320, **1 camada MTP**, **262,144** posições nativas. A quantização é
modelopt MIXED_PRECISION: NVFP4 (grupo 16) para todos os MLP e o `lm_head`, FP8 para cada projeção de atenção,
e pesos de MTP/vision deixados sem quantização. Tamanho em disco 21.921 GB.

## Usando o endpoint local

O servidor fala tanto o protocolo de rede da OpenAI quanto o da Anthropic em `127.0.0.1:8080`.

| rota | protocolo | observações |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | suporta `stream`, `max_tokens`, `enable_thinking` |
| `POST /v1/completions` | OpenAI | completude de texto legada; `prompt` deve ser uma única string |
| `POST /v1/messages` | Anthropic | suporta `stream`, `max_tokens` |
| `GET /v1/models` | ambos | retorna o id do modelo carregado |
| `GET /health` | — | liveness; use isto para aguardar a inicialização |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":16}'
```

### Comprimento de contexto e concorrência

A janela de contexto é `--ctx` (**padrão 32768**) até as **262144** posições do checkpoint, e
`--concurrency` define para quantas sequências o cache KV é dimensionado. A concorrência padrão é o que um
orçamento de KV de 40 GB cobre, limitada a 16, então um contexto longo a reduz automaticamente:

```sh
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144
# context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB
```

> **Para qualquer benchmark de contexto longo, `--ctx` é obrigatório.** Sem ele, o servidor roda em 32768 e um
> prompt mais longo retorna **nenhum conteúdo** em vez de um erro — uma falha silenciosa que parece
> um bug do cliente.

**Concorrência.** O servidor agrupa até 16 requisições concorrentes em um único forward pass,
coletando-as dentro de uma janela de 25 ms. Defina `GB10_BATCH_LOG=1` para registrar o tamanho do grupo a cada passo — 16
requisições paralelas devem registrar `batch of 16`. Com prompts idênticos, todas as 16 retornam
respostas idênticas byte a byte.

**As respostas são despojadas do bloco de raciocínio do modelo.** `enable_thinking` tem padrão `true`
(igual ao checkpoint), então o template coloca o ` thinking` de abertura no prompt e a geração
retorna o raciocínio seguido de `</think>`; o endpoint remove tudo até essa tag, então
`content` é a própria resposta. **Passe `"enable_thinking": false` para uma resposta mais curta e direta, sem
raciocínio gerado.** **O streaming ainda repassa o raciocínio** — veja
[`docs/NEXT.md`](docs/NEXT.md) para a correção planejada e a armadilha de truncamento que ela precisa evitar.

Nenhuma autenticação é implementada e o listener está vinculado ao loopback; adicione um proxy na frente dele se
precisar ser acessível de outros lugares.

## O loop

`loop/run_round.sh` é o driver de iteração. Cada rodada:

1. `cargo build --release --workspace`
2. `cargo test --workspace --release`
3. gate de correção: correspondência exata de tokens com traços oráculo congelados do HF
4. gate de benchmark: artefato novo gravado em `bench/results/`
5. commit e push para `omnigeeker/GB10-Engine`

O estado durável fica em `loop/state.json`; os relatórios por rodada em `loop/rounds/`. Uma rodada que falha em um
gate é registrada como `FAIL` e não avança o marco.

## Notas de metodologia

Três regras que este projeto aprendeu na prática, e aplica a todos os números acima:

1. **Somente pares da mesma sessão são comparáveis.** Código fp32 idêntico mediu 90.54 s e 111.59 s
   (+23.2%) em dias diferentes nesta máquina; o número de 32K do llama.cpp variou de 44.55 s → 53.60 s no
   mesmo período. As razões se mantiveram, os segundos absolutos não.
2. **Um efeito medido não é uma causa atribuída.** O throughput de decodificação em 128K e 256K melhorou ~23%
   ao longo deste trabalho, enquanto 8K/32K reproduziram exatamente seus valores registrados. Que a melhoria é
   real está medido; *por quê* ainda não foi atribuído, e é registrado assim.
3. **Uma sonda mede uma sensibilidade, não a folga.** Uma sonda de ocupação mostrando que um kernel custa 1.47x
   mais com metade da ocupação não diz nada sobre se a ocupação pode ser aumentada — isso exige a
   aritmética de recursos (aqui: 2 blocos/SM hoje, 3 exigiriam um corte de 22% na memória compartilhada).

## Documentação

| Documento | Conteúdo |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | guia completo de uso: exemplos em Python e de streaming, tabela de parâmetros, limites de decodificação |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | física de hardware medida e como o roofline foi derivado |
| [`docs/TARGETS.md`](docs/TARGETS.md) | o contrato de desempenho acordado |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | arquitetura do motor |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) | o kernel de atenção com tiling, chunking, orçamento de memória, resultados de needle |
| [`bench/longctx/comparison.md`](bench/longctx/comparison.md) | o log completo de comparação de contexto longo contra o llama.cpp, incluindo hipóteses rejeitadas e correções |
| [`bench/ppl/`](bench/ppl/) · [`bench/mmlu/`](bench/mmlu/) | resultados de precisão registrados e os harnesses que os produziram |
| [`docs/NEXT.md`](docs/NEXT.md) | a lista de trabalho em aberto |

## Licença

Consulte o repositório para informações de licença.
