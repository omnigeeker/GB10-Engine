# GB10-Engine
[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Español** · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Português](README.pt-BR.md)

Un motor de inferencia escrito desde cero y en Rust puro para **Qwen3.8-27B (NVFP4)** sobre la
**NVIDIA DGX Spark GB10** (`sm_121`, CUDA 13), medido cara a cara contra
**llama.cpp** con los mismos pesos, la misma máquina y el mismo harness.

- Kernels CUDA personalizados compilados a PTX para `sm_121`, sin runtime de inferencia de proveedor.
- Híbrido de Gated DeltaNet + atención completa, 64 capas, contexto nativo de **262,144 tokens**.
- Endpoints HTTP compatibles con OpenAI **y** con Anthropic, streaming, batching continuo.
- Todos los números de este README son reproducibles a partir de un script de este repositorio.

---

## Benchmark: GB10-Engine vs llama.cpp

**Esta es la sección que hay que leer.** La misma máquina, el mismo checkpoint NVFP4 (gb10-engine lee el
checkpoint NVFP4 de `modelopt`; llama.cpp lee un GGUF convertido a partir de los mismos pesos), los mismos
prompts, el mismo cliente, ejecuciones estrictamente secuenciales para que ninguno de los contendientes
tenga la GPU para sí solo.

### Configuración de la prueba

| | |
|---|---|
| Hardware | NVIDIA DGX Spark **GB10**, `sm_121`, 48 SMs, 121.7 GiB de LPDDR5X unificada |
| Ancho de banda de lectura medido | **228 GB/s** (véase [`docs/PHYSICS.md`](docs/PHYSICS.md)) |
| gb10-engine | `./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144` |
| llama.cpp | `llama-server -m models/Qwen3.8-27B-NVFP4.gguf -c 262144 -ngl 99 -fa on` |
| Cliente | [`bench/longctx/ttft.py`](bench/longctx/ttft.py) — petición idéntica dos veces por ensayo, con un marcador inicial único para que «warm» no pueda ser un falso acierto |
| Decodificación | greedy, `max_tokens 128` |
| Caché de prefijo | habilitada en ambos (esto es lo que da sentido al **warm TTFT**) |

**Definiciones de las métricas**, porque «TTFT» esconde una pregunta real:

- **Cold TTFT** — el prefijo no se ha visto nunca, así que se hace prefill de todo el prompt. Esta es la
  medición del cómputo de prefill.
- **Warm TTFT** — la misma petición otra vez, de modo que una caché de prefijo puede saltarse el prefill. Esta es la
  medición del acierto de caché.
- **OTPS** — tokens de salida por segundo durante la decodificación.

> **Solo son comparables los pares medidos en la misma sesión.** El mismo código midió con **23% de
> diferencia** en esta máquina en días distintos (véase «Notas de metodología»). Todos los ratios de
> abajo provienen de un par medido en una sola sesión; cada celda tiene al menos dos mediciones
> independientes.

### Rendimiento de decodificación (OTPS) — gb10-engine gana 3 de 4

| Contexto | gb10-engine | llama.cpp | Resultado |
|---|---|---|---|
| 8K | **8.91** | 7.26 | **1.23x más rápido** |
| 32K | **7.14** | 6.865 | **1.04x más rápido** |
| 128K | **5.29** | 4.75 | **1.11x más rápido** |
| 256K | 3.665 | **3.86** | 1.05x más lento |

### TTFT — gb10-engine gana todas las celdas warm y va por detrás en todas las celdas cold

| Contexto | Cold TTFT (gb10 / llama.cpp) | Warm TTFT (gb10 / llama.cpp) |
|---|---|---|
| **8K** | 10.35 s / **9.50 s** — 1.09x más lento | **0.03 s** / 0.223 s — **7.4x más rápido** |
| **32K** | 80.00 s / **44.55 s** — 1.80x más lento | **0.05 s** / 0.29 s — **5.8x más rápido** |
| **128K** | 890.54 s / **275.04 s** — 3.24x más lento | **0.14 s** / 0.48 s — **3.4x más rápido** |
| **256K** | 3295.45 s / **726.22 s** — 4.54x más lento | **0.28 s** / 0.66 s — **2.36x más rápido** |

**Hay que leer esta tabla con honestidad: el resultado está dividido.**

- **Warm TTFT: 4 de 4 ganadas**, por un factor de 2.4x a 7.8x. Una vez que un prefijo está en caché, este motor
  reanuda en 0.03–0.28 s donde llama.cpp necesita 0.22–0.66 s. Para cargas de trabajo de agentes, RAG y
  multi-turno que reenvían un prefijo largo, esta es la métrica que domina el tiempo de reloj.
- **OTPS: 3 de 4 ganadas**, por un factor de 1.04x a 1.23x. La celda de 256K está 1.05x por detrás.
- **Cold TTFT: 0 de 4 ganadas.** Este motor está 1.09x por detrás en 8K y 4.54x por detrás en 256K. **Este es
  el problema abierto**, y es aritmética: las 16 capas de atención completa hacen trabajo de prefill
  cuadrático y el kernel de atención de prefill se ejecuta al ~8% del pico fp16 de CUDA-core de la pieza.

**Progreso en Cold TTFT logrado en la ronda de trabajo actual** (medido en el servidor, validado por la
comprobación de corrección `chunked-prefill`):

| Contexto | Cold TTFT antes | Cold TTFT ahora | Mejora |
|---|---|---|---|
| 8K | 14.93 s | **10.35 s** | **1.43x** |
| 32K | 90.54 s | **80.00 s** | **1.13x** |
| 128K | 1134.63 s | **890.54 s** | **1.28x** |
| 256K | 4138.39 s | **3295.45 s** | **1.25x** |

### Precisión — gb10-engine es más preciso que llama.cpp con los mismos pesos

**Perplexity** (WikiText-2, `n_ctx 512`, 580 ventanas, 147,900 predicciones — la misma configuración
que la línea base histórica):

| Implementación | Pesos | Perplexity | vs BF16 |
|---|---|---|---|
| **gb10-engine** | **NVFP4** | **7.1006** | +0.71% |
| llama.cpp | NVFP4 (mismo GGUF) | 7.2088 | +2.24% |
| transformers (referencia) | BF16 (modelo base) | **7.0506** | — |

**MMLU** (3,240 preguntas, acuerdo de 4 vías):

| Implementación | Pesos | MMLU | Correctas |
|---|---|---|---|
| transformers (referencia) | BF16 | **80.74%** | 2616/3240 |
| **gb10-engine** | **NVFP4** | **80.34%** | **2603/3240** |
| torch, NVFP4 dequantizado | NVFP4 | 80.46% | 2607/3240 |
| llama.cpp | NVFP4 (mismo GGUF) | 79.88% | 2588/3240 |

**Por qué estos números indican que la ruta de cuantización es correcta:**

- gb10-engine es **1.5% mejor que llama.cpp** en perplexity y **+0.46 pp mejor en MMLU**, usando
  los *mismos* pesos NVFP4. El motor no está perdiendo precisión respecto a la implementación madura.
- Se sitúa **0.71% por encima de BF16** en perplexity y **0.40 pp por debajo de BF16** en MMLU — el coste esperado de
  los pesos de 4 bits para inferencia sin activaciones ni gradientes, y la misma dirección y magnitud que la
  referencia independiente de torch dequantizado.
- El harness **verifica de forma cruzada el tokenizer contra `llama-tokenize` y reporta 0 discrepancias sobre
  297,054 tokens**, así que ninguna diferencia de precisión aquí es un artefacto de tokenización.
- La corrección también se valida en cada ronda: coincidencia exacta a nivel de token contra trazas oráculo
  congeladas de HuggingFace (64 capas, greedy), más una comprobación de paridad de batch.

### Problema conocido: la recuperación en contexto largo no está certificada actualmente

La validación de aguja en un pajar ([`bench/longctx/run_validation.sh`](bench/longctx/run_validation.sh))
registró **3/3 en 32K, 1/1 en 128K, 1/1 en 256K**. **Volver a ejecutar ahora el mismo tramo de 32K da 0/3 en
gb10-engine *y* 0/3 en llama.cpp**, así que el fallo se rastreó y el motor no quedó implicado:

- los fallos son deterministas (`finish_reason: stop`, `completion_tokens: 0` — EOS como primer token);
- son erráticos en *longitud*, no monótonos, lo que no es la forma de una regresión de precisión de KV;
- son **idénticos con `temperature: 0`**, con `--no-prefix-cache`, con `PREFILL_CHUNK` 2048
  *y* 8192, y con `enable_thinking` omitido/false/true;
- **y llama.cpp falla con los mismos prompts.**

**Conclusión: la precisión en las métricas estándar es normal; la recuperación en contexto largo ni está
certificada ni refutada por este harness a día de hoy.** Se documenta abiertamente en lugar de omitirse.
El análisis completo está en [`bench/longctx/comparison.md`](bench/longctx/comparison.md).

### Reprodúcelo tú mismo

```bash
# build
cargo build --release --workspace

# perplexity (gb10-engine) -- compares to the recorded 7.0988 / llama.cpp 7.2088 / BF16 7.0506
./target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --text bench/ppl/wiki.test.raw --ctx 512 --chunks 580

# correctness + benchmark gate (build, tests, token-exact oracle, batch parity, attn tiling, bench)
loop/run_round.sh
```

**No ejecutes a la vez una medición respaldada por servidor y el gate** — el gate arranca y para
servidores en el puerto 8080 y matará el servidor a mitad de petición.

---

## Estado

| Hito | Estado |
|---|---|
| M0 Base: config, safetensors, tokenizer, plantilla de chat | **hecho** |
| M1 Backend CUDA + GEMV roofline | **hecho** |
| M2 Kernels de capa (Gated DeltaNet, atención, MLP, NVFP4/FP8) | **hecho** |
| M3 Forward completo + paridad de decodificación greedy | **hecho** (token-exact, 64 capas, 16/16) |
| M4 Decodificación especulativa MTP | **hecho** (token-exact; 44/48 de aceptación) |
| M5 KV paginado + batching continuo | **hecho** (16 concurrentes) |
| M6 Endpoints OpenAI + Anthropic | **hecho** (ambos protocolos + streaming) |
| M7 Optimización de roofline | **en curso** |
| M8 Cara a cara con llama.cpp | **hecho** — OTPS 3/4 ganadas, warm TTFT 4/4 ganadas, cold TTFT 0/4 |

### Dónde está el rendimiento respecto al roofline del hardware

> **Lee primero [`docs/PHYSICS.md`](docs/PHYSICS.md).** La petición original pedía 100 tok/s
> en un solo stream. Medido en esta máquina, GB10 sostiene **228 GB/s** de ancho de banda de lectura y este
> checkpoint requiere **17.608 GB** de tráfico de pesos por token, así que el roofline de un solo stream es
> **12.95 tok/s** y el suelo por paso es **77.2 ms**. El objetivo de 100 tok/s necesita ~1.76 TB/s —
> **7.7x el ancho de banda medido** — así que ninguna implementación puede alcanzarlo. El contrato acordado en
> [`docs/TARGETS.md`](docs/TARGETS.md) refleja lo que el hardware puede hacer realmente.

La brecha restante de cold-prefill es un solo kernel: la atención de prefill por bloques (tiled) se ejecuta a
**3.12 TFLOP/s, ~8.4% del pico fp16 de CUDA-core**, y su flujo de instrucciones es solo 42.4% aritmético,
por lo que está limitado por el issue. Cerrar la brecha de cold TTFT en contexto largo requiere reescribir ese
kernel con tensor core y `mma.sync`; el plan y las aceleraciones necesarias (1.29x en 8K, 5.98x en 32K, 8.40x en 128K, 8.82x en
256K) están en [`bench/longctx/comparison.md`](bench/longctx/comparison.md).

## Estructura

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

## Inicio rápido

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

Tarda unos **40 s** en cargar los pesos y luego sirve en `127.0.0.1:8080`.

## Modelo

`nv-community/Qwen3.8-27B-NVFP4`, descargado con la CLI de ModelScope:

```bash
modelscope download --model nv-community/Qwen3.8-27B-NVFP4 \
    --local_dir models/Qwen3.8-27B-NVFP4
```

`models/` es un enlace simbólico al directorio del checkpoint descargado y no está rastreado por git.

Arquitectura: `qwen3_5`, 64 capas (48 de atención lineal Gated-DeltaNet + 16 de atención completa, una
de cada 4 capas), hidden 5120, 24 cabezas de query / 4 de KV (grupo GQA 6), head_dim 256, RoPE parcial (0.25,
mRoPE intercalado), vocab 248320, **1 capa MTP**, **262,144** posiciones nativas. La cuantización es
modelopt MIXED_PRECISION: NVFP4 (grupo 16) para todos los MLP y `lm_head`, FP8 para cada proyección de
atención, y los pesos de MTP/vision sin cuantizar. Tamaño en disco 21.921 GB.

## Uso del endpoint local

El servidor habla tanto el protocolo de OpenAI como el de Anthropic en `127.0.0.1:8080`.

| ruta | protocolo | notas |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | soporta `stream`, `max_tokens`, `enable_thinking` |
| `POST /v1/completions` | OpenAI | completado de texto heredado; `prompt` debe ser una sola cadena |
| `POST /v1/messages` | Anthropic | soporta `stream`, `max_tokens` |
| `GET /v1/models` | ambos | devuelve el id del modelo cargado |
| `GET /health` | — | liveness; usa esto para esperar al arranque |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":16}'
```

### Longitud de contexto y concurrencia

La ventana de contexto es `--ctx` (**por defecto 32768**) hasta las **262144** posiciones del checkpoint, y
`--concurrency` establece para cuántas secuencias se dimensiona la caché KV. La concurrencia por defecto es la
que cubre un presupuesto de 40 GB de KV, con un tope de 16, así que un contexto largo la reduce automáticamente:

```sh
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4 --ctx 262144
# context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB
```

> **Para cualquier benchmark de contexto largo, `--ctx` es obligatorio.** Sin él, el servidor funciona a 32768 y un
> prompt más largo devuelve **ningún contenido en absoluto** en lugar de un error — un fallo silencioso que parece
> un bug del cliente.

**Concurrencia.** El servidor agrupa hasta 16 peticiones concurrentes en un solo forward pass,
recogiéndolas dentro de una ventana de 25 ms. Activa `GB10_BATCH_LOG=1` para registrar el tamaño del grupo en
cada paso — 16 peticiones en paralelo deberían registrar `batch of 16`. Con prompts idénticos, las 16 devuelven
respuestas idénticas byte a byte.

**Las respuestas se limpian del bloque de razonamiento del modelo.** `enable_thinking` es `true` por defecto
(igual que el checkpoint), así que la plantilla pone la apertura ` thinking` en el prompt y la generación
devuelve el razonamiento seguido de `</think>`; el endpoint elimina todo hasta esa etiqueta, de modo que
`content` es la respuesta en sí. **Pasa `"enable_thinking": false` para obtener una respuesta más corta y directa, sin
generar razonamiento en absoluto.** **El streaming sigue dejando pasar el razonamiento** — véase
[`docs/NEXT.md`](docs/NEXT.md) para ver el arreglo diseñado y la trampa de truncado que debe evitar.

No hay autenticación implementada y el listener está vinculado a loopback; añade un proxy delante si
necesita ser accesible desde otros sitios.

## El bucle

`loop/run_round.sh` es el driver de iteración. Cada ronda:

1. `cargo build --release --workspace`
2. `cargo test --workspace --release`
3. gate de corrección: coincidencia exacta a nivel de token contra trazas oráculo congeladas de HF
4. gate de benchmark: artefacto recién escrito en `bench/results/`
5. commit y push a `omnigeeker/GB10-Engine`

El estado duradero vive en `loop/state.json`; los informes por ronda, en `loop/rounds/`. Una ronda que falla un
gate se registra como `FAIL` y no avanza el hito.

## Notas de metodología

Tres reglas que este proyecto aprendió a base de golpes y que aplica a todos los números anteriores:

1. **Solo son comparables los pares de la misma sesión.** El mismo código fp32 midió 90.54 s y 111.59 s
   (+23.2%) en días distintos en esta máquina; el número de 32K de llama.cpp derivó de 44.55 s → 53.60 s en el
   mismo periodo. Los ratios se mantuvieron, los segundos absolutos no.
2. **Un efecto medido no es una causa atribuida.** El rendimiento de decodificación de 128K y 256K mejoró ~23%
   a lo largo de este trabajo, mientras que 8K/32K reprodujeron exactamente sus valores registrados. Que la
   mejora es real está medido; *por qué* sigue sin atribuirse, y así se registra.
3. **Una sonda mide una sensibilidad, no un margen.** Una sonda de ocupación que muestra que un kernel cuesta 1.47x
   más con la mitad de ocupación no dice nada sobre si se puede aumentar la ocupación — eso requiere la
   aritmética de recursos (aquí: 2 bloques/SM hoy, 3 necesitaría un recorte del 22% de memoria compartida).

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | guía de uso completa: ejemplos de Python y streaming, tabla de parámetros, límites de decodificación |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | física del hardware medida y cómo se derivó el roofline |
| [`docs/TARGETS.md`](docs/TARGETS.md) | el contrato de rendimiento acordado |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | arquitectura del motor |
| [`docs/LONG_CONTEXT.md`](docs/LONG_CONTEXT.md) | el kernel de atención por bloques (tiled), chunking, presupuesto de memoria, resultados de aguja |
| [`bench/longctx/comparison.md`](bench/longctx/comparison.md) | el registro completo de comparación de contexto largo contra llama.cpp, incluidas las hipótesis rechazadas y las correcciones |
| [`bench/ppl/`](bench/ppl/) · [`bench/mmlu/`](bench/mmlu/) | resultados de precisión registrados y los harnesses que los produjeron |
| [`docs/NEXT.md`](docs/NEXT.md) | la lista de trabajo abierta |

## Licencia

Consulta el repositorio para obtener información sobre la licencia.
