# Using the local endpoint

`gb10-server` serves the NVFP4 checkpoint over both the **OpenAI** and the
**Anthropic** wire protocols, on loopback, with no authentication.

Everything below was run against a live server on this machine. The one thing
that was *not* exercised here is the official Python SDKs — neither `openai` nor
`anthropic` is installed on this host — so those snippets are the standard client
form for these protocols rather than transcripts.

## Start it

```sh
cargo build --release
./target/release/gb10-server --model models/Qwen3.8-27B-NVFP4
```

| flag | default | meaning |
|---|---|---|
| `--model DIR` | `models/Qwen3.8-27B-NVFP4` | checkpoint directory |
| `--host HOST` | `127.0.0.1` | listen address |
| `--port PORT` | `8080` | listen port |
| `--name NAME` | `Qwen3.8-27B-NVFP4` | id reported by `/v1/models` and in responses |
| `--ctx N` | `32768` | context window in tokens (prompt + generation), max `262144` |
| `--concurrency N` | auto | sequence slots to size the KV cache for, max `16` |

`--ctx` above `262144` is refused: that is the checkpoint's
`max_position_embeddings`, and past it the model is extrapolating outside its
RoPE range.

`--concurrency` defaults to the largest count that fits a 40 GB KV budget,
capped at 16, so a long context automatically reduces concurrency rather than
failing to allocate. Passing it explicitly is honoured but refused if it does
not fit. The startup line reports the decision:

```
context 262144 tokens, 1 concurrent sequence(s), KV cache 34.4 GB (34360 MB per sequence)
```

Two things scale with the context window:

| per token | scales with | size |
|---|---|---|
| KV cache — 16 full-attention layers, K and V, 4 KV heads × 256 f32 dims | `--concurrency` | **128 KiB** per sequence |
| `a`, `b`, `normed` — residual buffers, `hidden × ctx × 4` each | no | **60 KiB** total |

So the window costs `128 KiB × concurrency + 60 KiB` per token. At 256K that is
32 GiB of KV and 15 GiB of residual buffers. The 48 Gated-DeltaNet layers hold
a fixed-size recurrent state and do **not** grow with the window.

`a`/`b`/`normed` are the odd ones out: they are sized to the whole window, but
prefill only ever touches one 2048-token chunk at a time, so at 256K that 15 GiB
is slack rather than a requirement.

### How much context fits

GB10 has no separate VRAM — it is one 121.7 GiB pool of unified LPDDR5X shared
with the CPU. Measured on this machine, with everything else idle:

```
total                     121.7 GiB
system resident           - 8.0 GiB
weights (NVFP4)           -20.4 GiB
Scratch (2048 rows)       - 1.1 GiB   (chunk-sized, not ctx-sized)
CUDA context and misc     - 1.5 GiB
--------------------------------------
left for the context        90.7 GiB
```

| concurrency | per token | physical max | at the native 262144 | total |
|---|---|---|---|---|
| 1 | 188 KiB | **~506 K** | 47 GiB | 78 GiB |
| 2 | 316 KiB | ~301 K | 79 GiB | 110 GiB |
| 4 | 572 KiB | ~166 K | 143 GiB | does not fit |
| 8 | 1084 KiB | ~88 K | 271 GiB | does not fit |
| 16 | 2108 KiB | ~45 K | 527 GiB | does not fit |

The model's own `max_position_embeddings` is **262144**, and `--ctx` above that
is refused because the model would be extrapolating outside its RoPE range.
Memory is not the binding constraint: one sequence at the full 262144 uses only
47 GiB of the 90.7 GiB available, so there is room for roughly twice the native
window if the model had the positions to use.

Two consequences worth knowing:

* **`--concurrency 2` at 262144 fits physically but is refused.** `KV_BUDGET_BYTES`
  is a 40 GiB *policy* ceiling, and two sequences would need 64 GiB of KV. The
  ceiling is a constant in `crates/gb10-server/src/main.rs`, not a hardware
  limit — raise it if two 256K sequences are wanted and the box is otherwise free.
* **The real cost of a long window is time, not memory.** Prefill grows
  quadratically; see the next section.

The weights take **~40–95 s** to load, during which the port is not yet open.
There is no `--help`; unknown arguments are a hard error.

Check it is up:

```sh
curl -s http://127.0.0.1:8080/v1/models
# {"data":[{"id":"Qwen3.8-27B-NVFP4","object":"model","owned_by":"gb10-engine"}],"object":"list"}
```

## Routes

| route | protocol | notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | `stream`, `max_tokens`, `enable_thinking` |
| `POST /v1/completions` | OpenAI | legacy text completion; `prompt` must be a **single** string |
| `POST /v1/messages` | Anthropic | `stream`, `max_tokens` |
| `GET /v1/models` | both | the loaded model id |
| `GET /health` | — | liveness; use this to wait for startup |

## Parameters

| field | default | notes |
|---|---|---|
| `max_tokens` | **512** | also accepts `max_completion_tokens` |
| `enable_thinking` | **true** | matches the checkpoint; also accepted as `chat_template_kwargs.enable_thinking` |
| `stream` | false | SSE for both protocols |
| `model` | ignored | any value works; the server always serves the loaded model |

**Decoding is greedy.** There is no `temperature`, `top_p` or `seed` — they are
accepted in the body but do not change the output. Identical prompts return
byte-identical answers.

## `enable_thinking` is the setting that matters

The checkpoint's template puts the opening ` thinking` into the *prompt*, so the
model emits a reasoning block, then `</think>`, then the answer. **The endpoint
strips everything through `</think>`**, so `content` is the answer alone.

* `enable_thinking: true` (default) — better on hard questions, costs tokens and
  latency. A 300-token budget for `17*23` produced 56 tokens of reasoning and
  returned `391`.
* `enable_thinking: false` — no reasoning generated at all; shorter and faster,
  and the stripper becomes a no-op.

Measured on 10 ordinary questions at the **default `max_tokens: 512`**, all 10
finished with `stop` and used 13–188 completion tokens, so the default is
adequate for normal use.

**The failure mode to know about:** on an ambiguous or open-ended question the
model can reason indefinitely and never emit `</think>`. `In one short sentence,
what is GB10?` did exactly that — 800 completion tokens, still no closing tag,
`finish_reason: "length"`, and the raw reasoning returned as `content`. If you
see the model's inner monologue in `content`, that is what happened: either raise
`max_tokens` or set `enable_thinking: false`. The stripper deliberately returns
truncated reasoning unstripped rather than emptying the response, since an empty
answer would be worse than a verbose one.

## curl

```sh
# chat, thinking off
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count from one to five."}],
       "max_tokens":32,"enable_thinking":false}'
# {"choices":[{"finish_reason":"stop","index":0,"message":{"content":"1, 2, 3, 4, 5",
#  "role":"assistant"}}],...,"model":"Qwen3.8-27B-NVFP4",...}

# streaming
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Say hello in three words."}],
       "max_tokens":16,"stream":true,"enable_thinking":false}'

# legacy text completion (single string prompt; a batch is refused)
curl -s http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","prompt":"The capital of France is","max_tokens":8,"enable_thinking":false}'
# {"choices":[{"finish_reason":"length","index":0,"logprobs":null,
#  "text":"The capital of France is **Paris**."}],...,"object":"text_completion",...}

# Anthropic protocol
curl -s http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","max_tokens":64,
       "messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}]}'
# {"content":[{"text":"Paris","type":"text"}],"id":"msg_...","model":"Qwen3.8-27B-NVFP4",
#  "role":"assistant","stop_reason":"end_turn",...,"usage":{"input_tokens":59,"output_tokens":...}}
```

`/v1/completions` returns the **legacy** shape: `object: "text_completion"`, a bare
`text` per choice, no `message`, and `logprobs` always `null`. Streaming uses
`text` too, not chat's `delta`. It wraps the prompt as a single user turn, since
the checkpoint is a chat model.

Note the Anthropic response uses `stop_reason`/`input_tokens`/`output_tokens`
rather than OpenAI's `finish_reason`/`prompt_tokens`/`completion_tokens`.

## Python

With `requests` (installed here):

```python
import requests

r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={
    "model": "m",
    "messages": [{"role": "user", "content": "What is 17*23? Just the number."}],
    "max_tokens": 300,
    "enable_thinking": False,
})
print(r.json()["choices"][0]["message"]["content"])
```

Streaming:

```python
with requests.post("http://127.0.0.1:8080/v1/chat/completions", stream=True, json={
    "model": "m", "messages": [{"role": "user", "content": "Count to five."}],
    "max_tokens": 32, "stream": True, "enable_thinking": False,
}) as r:
    for line in r.iter_lines():
        if line and line.startswith(b"data: "):
            print(line[6:].decode(), flush=True)
```

With the official SDKs (not installed on this host — standard client form):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="Qwen3.8-27B-NVFP4",
    messages=[{"role": "user", "content": "hi"}],
    extra_body={"enable_thinking": False},
).choices[0].message.content)

from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:8080", api_key="not-needed")
print(client.messages.create(
    model="Qwen3.8-27B-NVFP4", max_tokens=64,
    messages=[{"role": "user", "content": "hi"}],
).content[0].text)
```

`api_key` is required by the SDK constructors but ignored by the server.

## Concurrency

The server batches up to **16** concurrent requests into one forward pass,
collecting them within a 25 ms window. 16 simultaneous requests, each asking for
a different number, all returned the correct distinct answers in **7.7 s** wall.
Set `GB10_BATCH_LOG=1` to log the group size per step.

## Limits and caveats

* **Context is `--ctx` tokens** (default 32768, max the checkpoint's 262144),
  prompt **plus** generation. A longer prompt is an error, not a truncation,
  and generation is clamped so the two together stay inside the window.
* **Prefill cost grows quadratically with the prompt.** The 48 Gated-DeltaNet
  layers are linear, but the 16 full-attention layers attend over the whole
  prefix, so a long prompt is priced by that term. Measured on this box
  (prompts of repeated filler, needle recovered at 10/50/90% depth):

  | prompt | prefill |
  |---|---|
  | 5 K | ~50 s |
  | 12 K | ~170 s |
  | 24 K | ~9 min |
  | 32 K | ~16 min |
  | 128 K | ~3 h |
  | 256 K | ~12 h |

  Fitting those gives `≈ 8.8 ms·T + 6.0e-7·T²` seconds. The quadratic term is
  the attention kernel's key range and is inherent to dense attention at this
  scale, not an artefact of chunking; chunking only bounds *memory*, not time.
  Short and medium prompts are unaffected — the quadratic term is under 10% of
  the total below ~16 K tokens.
* **Greedy only** — no sampling controls, no `n`/`best_of`, no tool calling, no
  JSON mode, no vision.
* **No authentication**, bound to loopback. Put a proxy in front if it needs to
  be reachable from elsewhere.
* **Streaming still passes the reasoning block through** when thinking is on;
  only the non-streaming path strips it. Pass `enable_thinking: false` if you
  stream and do not want reasoning.
* Multi-turn works by sending the full `messages` array; there is no
  server-side session state.
* **Concurrency 1 at 256K.** The KV budget means a 256K window leaves room for
  exactly one sequence; requests beyond the slot count wait for the batcher
  rather than being refused.
* `ModelState` also holds context-sized residual buffers (`a`, `b`, `normed`),
  ~16 GB at 256K. They only ever need a prefill chunk's worth of rows, so this
  is slack rather than a requirement.

## Stopping it

```sh
pkill -x gb10-server
```

Use `pkill -x` (exact name). `pkill -f gb10-server` also matches the invoking
shell's own command line and kills your session.
