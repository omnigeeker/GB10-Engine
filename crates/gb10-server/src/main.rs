//! GB10-Engine local endpoint: OpenAI- and Anthropic-compatible HTTP APIs.
//!
//! The HTTP layer is deliberately dependency-free (`std::net` + `serde_json`).
//! The surface actually needed is four routes and one SSE framing format;
//! pulling in an async runtime and a web framework for that would add far more
//! moving parts than it removes, and would make the engine's only network
//! dependency a third-party stack rather than the CUDA driver.
//!
//! Generation is currently serialised: one request at a time, greedy decode.
//! Continuous batching is a separate milestone; see `docs/ROADMAP.md`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{mpsc, OnceLock};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use gb10_core::chat::{text_message, ChatMessage, ChatTemplate, ChatTemplateOptions};
use gb10_core::config::ModelConfig;
use gb10_core::QwenTokenizer;
use gb10_cuda::Device;
use gb10_model::layer::Scratch;
use gb10_model::model::{Model, ModelState};
use serde_json::{json, Value};

/// Positions the checkpoint was trained for (`max_position_embeddings` in
/// `config.json`). Beyond this the model is extrapolating outside its RoPE
/// range, so it is refused rather than silently served.
const NATIVE_MAX_POSITIONS: usize = 262144;

/// Concurrent sequence slots the state can be sized for. The scheduler sizes
/// itself below this when the KV budget cannot cover it.
const MAX_CONCURRENT: usize = 16;

/// Tokens per prefill pass.
///
/// `Scratch` holds one row per token, so this -- not the context window -- sets
/// how much scratch memory exists. A prompt longer than this is prefilled in
/// successive chunks that carry the KV cache and recurrent state forward. That
/// decoupling is what makes a long context possible at all: scratch sized to a
/// 256K context would be ~150 GB.
const PREFILL_CHUNK: usize = 2048;

/// KV cache bytes per (token, sequence).
///
/// Only the 16 full-attention layers cache K and V; each holds 4 KV heads of
/// 256 f32 dimensions, so 16 * 2 * 4 * 256 * 4 = 131072 bytes.
const KV_BYTES_PER_TOKEN_SEQ: usize = 131072;

/// How much of the unified pool to spend on KV cache. The weights are 17.6 GB
/// against 121 GB total, so this leaves headroom for activations and for
/// whatever else the machine is doing.
const KV_BUDGET_BYTES: usize = 40 * 1024 * 1024 * 1024;

/// Pick the number of sequence slots to allocate.
///
/// With no explicit request this is the largest count the KV budget covers,
/// capped at [`MAX_CONCURRENT`]. An explicit `--concurrency` is honoured but
/// refused if it does not fit, because the alternative is an allocation
/// failure partway through loading.
fn resolve_concurrency(ctx: usize, requested: Option<usize>) -> Result<usize> {
    let per_seq = (ctx * KV_BYTES_PER_TOKEN_SEQ) as u64;
    match requested {
        Some(n) => {
            let bytes = per_seq * n as u64;
            anyhow::ensure!(
                bytes <= KV_BUDGET_BYTES as u64,
                "--concurrency {n} at --ctx {ctx} needs {:.1} GB of KV cache, over the {:.0} GB \
                 budget; lower --concurrency or --ctx",
                bytes as f64 / 1e9,
                KV_BUDGET_BYTES as f64 / 1e9,
            );
            Ok(n)
        }
        None => Ok(((KV_BUDGET_BYTES as u64 / per_seq) as usize).clamp(1, MAX_CONCURRENT)),
    }
}

struct Args {
    model: PathBuf,
    host: String,
    port: u16,
    model_name: String,
    /// Context window in tokens (prompt + generation).
    ctx: usize,
    /// Concurrent sequences to size the KV cache for. `None` means "as many as
    /// the KV budget allows, up to `MAX_CONCURRENT`".
    concurrency: Option<usize>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model = PathBuf::from("models/Qwen3.8-27B-NVFP4");
        let mut host = "127.0.0.1".to_string();
        let mut port = 8080u16;
        let mut model_name = "Qwen3.8-27B-NVFP4".to_string();
        let mut ctx = 32768usize;
        let mut concurrency = None;
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--model" => model = PathBuf::from(it.next().context("--model needs a value")?),
                "--host" => host = it.next().context("--host needs a value")?,
                "--port" => port = it.next().context("--port needs a value")?.parse()?,
                "--name" => model_name = it.next().context("--name needs a value")?,
                "--ctx" => ctx = it.next().context("--ctx needs a value")?.parse()?,
                "--concurrency" => {
                    concurrency = Some(it.next().context("--concurrency needs a value")?.parse()?)
                }
                other => anyhow::bail!("unknown argument: {other}"),
            }
        }
        anyhow::ensure!(ctx >= 2, "--ctx must be at least 2");
        anyhow::ensure!(
            ctx <= NATIVE_MAX_POSITIONS,
            "--ctx {ctx} exceeds the checkpoint's {NATIVE_MAX_POSITIONS} positions"
        );
        if let Some(c) = concurrency {
            anyhow::ensure!(c >= 1, "--concurrency must be at least 1");
        }
        Ok(Self { model, host, port, model_name, ctx, concurrency })
    }
}

// ---------------------------------------------------------------------------
// Minimal HTTP/1.1
// ---------------------------------------------------------------------------

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

impl Request {
    fn json(&self) -> Result<Value> {
        if self.body.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&self.body).context("parsing request body as JSON")
    }
}

fn read_request(stream: &mut TcpStream) -> Result<Option<Request>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Some(Request { method, path, body }))
}

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

fn write_head(stream: &mut TcpStream, code: u16, content_type: &str, extra: &str) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\n{}Connection: close\r\n\r\n",
        code,
        status_text(code),
        content_type,
        extra
    )?;
    Ok(())
}

fn respond_json(stream: &mut TcpStream, code: u16, v: &Value) -> Result<()> {
    let body = serde_json::to_vec(v)?;
    write_head(
        stream,
        code,
        "application/json",
        &format!("Content-Length: {}\r\n", body.len()),
    )?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

/// One chunk of an HTTP/1.1 chunked body.
fn write_chunk(stream: &mut TcpStream, data: &[u8]) -> Result<()> {
    write!(stream, "{:x}\r\n", data.len())?;
    stream.write_all(data)?;
    stream.write_all(b"\r\n")?;
    stream.flush()?;
    Ok(())
}

fn end_chunked(stream: &mut TcpStream) -> Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

struct Engine {
    dev: Device,
    model: Model,
    tok: QwenTokenizer,
    tmpl: ChatTemplate,
    state: ModelState,
    sc: Scratch,
    name: String,
    /// Context window in tokens, for the request-time length check.
    ctx: usize,
    /// Sequence slots actually allocated. The batcher must not group more
    /// requests than this, or `prefill_seq` indexes past the state.
    n_seq: usize,
}

/// Qwen's chat template puts the opening ` thinking` into the *prompt*, so generation
/// returns the reasoning block, then `</think>`, then the answer. Passing that through
/// verbatim means a short response shows only reasoning and never the answer -- see the
/// 200-token example recorded in round 152 of `docs/NEXT.md`.
///
/// Strip everything through the closing tag. If there is no closing tag the reasoning
/// was truncated (or the model answered directly), and the text is returned unchanged
/// rather than emptied.
fn visible(text: &str) -> &str {
    match text.find("</think") {
        Some(i) => match text[i..].find('>') {
            Some(j) => text[i + j + 1..].trim_start(),
            None => text,
        },
        None => text,
    }
}

/// Holds pieces until the model's reasoning block is closed, then yields everything
/// after `</think>`.
///
/// **It deliberately does not hold the emitter.** An earlier design stored a
/// `&mut F` callback, which failed with `E0631` (`send` is `FnMut(&Value)`, the gate
/// wanted `FnMut(&str)`) and then forced an adapter closure that had to drop before the
/// trailing finish chunks reused `send` -- a lifetime problem solved only by moving a
/// closing brace. Returning the text instead means the caller keeps ownership of both
/// the writer and the gate, and neither depends on the other's scope.
///
/// Pieces are `tok.decode(&[next], true)` and `</think>` spans several of them, so the
/// tag has to be matched once complete. If generation is truncated before the tag
/// closes, `flush` yields the held text through `visible()` -- **otherwise the stream
/// would deliver nothing at all, which is worse than not gating.**
struct ThinkGate {
    held: String,
    opened: bool,
}

impl ThinkGate {
    fn new() -> Self {
        Self { held: String::new(), opened: false }
    }
    /// `Some(text)` when there is something to emit, `None` while still holding.
    fn push(&mut self, piece: &str) -> Option<String> {
        if self.opened {
            return Some(piece.to_string());
        }
        self.held.push_str(piece);
        // If no tag can still be forming, release immediately. Without this the gate
        // holds the ENTIRE response whenever the model emits no thinking block, which
        // makes streaming non-incremental and destroys TTFT.
        let i = match self.held.find("</think") {
            Some(i) => i,
            None => {
                if self.held.len() >= 64 || !self.held.contains('<') {
                    let out = std::mem::take(&mut self.held);
                    return Some(out);
                }
                return None;
            }
        };
        let j = self.held[i..].find('>')?;
        self.opened = true;
        let rest = self.held[i + j + 1..].trim_start().to_string();
        self.held.clear();
        if rest.is_empty() { None } else { Some(rest) }
    }
    /// Emit whatever is still held when generation ends without a closing tag.
    fn flush(&mut self) -> Option<String> {
        if self.opened || self.held.is_empty() {
            return None;
        }
        let rest = visible(&self.held).to_string();
        self.held.clear();
        if rest.is_empty() { None } else { Some(rest) }
    }
}

struct GenResult {
    ids: Vec<u32>,
    text: String,
    prompt_tokens: usize,
    finish: &'static str,
}

impl Engine {
    fn new(args: &Args) -> Result<Self> {
        let p = args.model.join("config.json");
        let raw = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let cfg: ModelConfig = serde_json::from_str(&raw).context("parsing config.json")?;
        let text = cfg.text_config.clone();

        let dev = Device::new(0)?;
        let model = Model::load_from(&dev, cfg, &args.model)?;
        let tok = QwenTokenizer::from_model_dir(&args.model)?;
        let tmpl = ChatTemplate::from_model_dir(&args.model)?;
        // Slots for concurrent sequences. `step_batch` takes its batch size from
        // `tokens.len()` and requires it to be <= this, so a single request
        // still runs with n_seq = 1 and costs nothing extra.
        let n_seq = resolve_concurrency(args.ctx, args.concurrency)?;
        // The KV cache is the only structure that scales with the context
        // window, so it is what decides whether this fits.
        let kv_gb = (args.ctx * n_seq * KV_BYTES_PER_TOKEN_SEQ) as f64 / 1e9;
        println!(
            "context {} tokens, {} concurrent sequence(s), KV cache {:.1} GB \
             ({:.0} MB per sequence)",
            args.ctx,
            n_seq,
            kv_gb,
            (args.ctx * KV_BYTES_PER_TOKEN_SEQ) as f64 / 1e6,
        );
        if n_seq < MAX_CONCURRENT && args.concurrency.is_none() {
            println!(
                "  (capped from {MAX_CONCURRENT} by the {:.0} GB KV budget; pass \
                 --concurrency to override)",
                KV_BUDGET_BYTES as f64 / 1e9
            );
        }
        let state = ModelState::new(&dev, &model, args.ctx, n_seq)?;
        // Scratch holds one row per token, so it is sized to a prefill chunk
        // rather than to the context: at 256K it would otherwise be ~150 GB.
        let sc = Scratch::new(&dev, &text, PREFILL_CHUNK)?;
        Ok(Self {
            dev,
            model,
            tok,
            tmpl,
            state,
            sc,
            name: args.model_name.clone(),
            ctx: args.ctx,
            n_seq,
        })
    }

    /// Prefill the prompt in [`PREFILL_CHUNK`]-sized passes, returning the next
    /// token after the last one.
    ///
    /// Chunk boundaries are invisible in the result. `attn_prefill` streams the
    /// whole key range on every call rather than only the rows it is given, so
    /// a later chunk starting from a non-empty cache computes exactly what a
    /// single pass over the whole prompt would; and the recurrent layers carry
    /// their state across calls. `gb10-verify chunked-prefill` gates precisely
    /// this, comparing one-shot against split prefill token for token.
    ///
    /// This is what decouples scratch memory from the context window: without
    /// it, `Scratch` and `ModelState`'s residual buffers would both have to
    /// hold the entire prompt.
    fn prefill_chunked(&mut self, ids: &[u32]) -> Result<u32> {
        debug_assert!(!ids.is_empty());
        let timing = std::env::var_os("GB10_TIMING").is_some();
        let t_all = std::time::Instant::now();
        let mut next = 0u32;
        for (i, c) in ids.chunks(PREFILL_CHUNK).enumerate() {
            let t0 = std::time::Instant::now();
            next = self
                .model
                .prefill_seq(&self.dev, c, &mut self.state, &mut self.sc, 0)?;
            if timing {
                eprintln!(
                    "  chunk {i} ({} tok, start {}): {:.2}s",
                    c.len(),
                    i * PREFILL_CHUNK,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        if timing {
            eprintln!(
                "  prefill {} tok in {:.2}s ({:.1} ms/tok)",
                ids.len(),
                t_all.elapsed().as_secs_f64(),
                t_all.elapsed().as_secs_f64() * 1e3 / ids.len() as f64
            );
        }
        Ok(next)
    }

    /// Render a conversation, prefill it, then decode greedily.
    ///
    /// `on_token` receives each decoded piece as it is produced, which is what
    /// makes streaming work without a second code path: the non-streaming
    /// route just passes a closure that ignores its argument.
    fn generate<F>(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        enable_thinking: bool,
        mut on_token: F,
    ) -> Result<GenResult>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let opts = ChatTemplateOptions { enable_thinking, ..Default::default() };
        let prompt = self.tmpl.render(messages, &opts)?;
        let ids = self.tok.encode(&prompt, true)?;
        anyhow::ensure!(!ids.is_empty(), "empty prompt");
        // The prompt must leave room for at least one generated token, so the
        // window is checked against prompt + 1 rather than the prompt alone.
        anyhow::ensure!(
            ids.len() + 1 <= self.ctx,
            "prompt is {} tokens but the context window is {} (raise --ctx)",
            ids.len(),
            self.ctx
        );

        let t_reset = std::time::Instant::now();
        self.state.reset(&self.dev)?;
        if std::env::var_os("GB10_TIMING").is_some() {
            eprintln!("  reset {:.2}s", t_reset.elapsed().as_secs_f64());
        }
        let mut next = self.prefill_chunked(&ids)?;

        // Generation appends to the same cache, so the prompt plus everything
        // generated has to stay inside the window. Clamping rather than
        // erroring matches how the `length` finish reason already behaves.
        let max_tokens = max_tokens.min(self.ctx - ids.len());

        let mut out = Vec::with_capacity(max_tokens);
        let mut text = String::new();
        let mut finish = "length";
        for _ in 0..max_tokens {
            if self.tok.is_eos(next) {
                finish = "stop";
                break;
            }
            out.push(next);
            let piece = self.tok.decode(&[next], true)?;
            text.push_str(&piece);
            on_token(&piece)?;
            next = self.model.step(&self.dev, next, &mut self.state, &mut self.sc)?;
        }
        Ok(GenResult { ids: out, text, prompt_tokens: ids.len(), finish })
    }
}

// ---------------------------------------------------------------------------
// Request decoding shared by both protocols
// ---------------------------------------------------------------------------

/// Flatten OpenAI/Anthropic content, which may be a string or a parts array.
///
/// Returns a plain `String` and goes through `text_message` rather than
/// constructing `ChatMessage` directly, because its `content` field is a
/// `minijinja::Value` (the template engine's type), not a `serde_json::Value`.
fn flatten_content(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn message(role: &str, content: String) -> ChatMessage {
    text_message(role, &content)
}

// ---------------------------------------------------------------------------
// OpenAI-compatible routes
// ---------------------------------------------------------------------------

fn openai_messages(body: &Value) -> Result<Vec<ChatMessage>> {
    let arr = body
        .get("messages")
        .and_then(|m| m.as_array())
        .context("`messages` must be an array")?;
    Ok(arr
        .iter()
        .map(|m| {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            message(role, flatten_content(m.get("content")))
        })
        .collect())
}

/// Legacy OpenAI text completion.
///
/// The checkpoint is a chat model, so the prompt is wrapped as a single user
/// turn and the reply is the assistant's text. The wire shape is the legacy one
/// -- `object: "text_completion"` with a bare `text` per choice, no `message`
/// and no `delta` -- because clients that ask for `/v1/completions` are parsing
/// that shape, not the chat one.
fn handle_completions(tx: &mpsc::Sender<Job>, body: &Value, stream: &mut TcpStream) -> Result<()> {
    let prompt = match body.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        // The legacy API accepts a batch. This server serves one sequence at a
        // time, so a batch is refused rather than silently answered with only
        // its first element.
        Some(Value::Array(a)) => {
            anyhow::ensure!(a.len() == 1, "`prompt` array must have exactly one element");
            a[0].as_str()
                .context("`prompt` array element must be a string")?
                .to_string()
        }
        _ => anyhow::bail!("`prompt` is required and must be a string"),
    };
    let messages = vec![text_message("user", &prompt)];
    let max_tokens = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;
    let want_stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let thinking = body
        .get("enable_thinking")
        .or_else(|| body.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let id = format!("cmpl-{}", now_secs());
    let created = now_secs();
    let name = model_name();

    if !want_stream {
        let r = remote_generate(tx, &messages, max_tokens, thinking, |_| Ok(()))?;
        return respond_json(
            stream,
            200,
            &json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": name,
                "choices": [{
                    "index": 0,
                    "text": visible(&r.text),
                    "logprobs": Value::Null,
                    "finish_reason": r.finish,
                }],
                "usage": {
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.ids.len(),
                    "total_tokens": r.prompt_tokens + r.ids.len(),
                },
            }),
        );
    }

    write_head(
        stream,
        200,
        "text/event-stream",
        "Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n",
    )?;
    let mut send = |v: &Value| -> Result<()> {
        let line = format!("data: {}\n\n", serde_json::to_string(v)?);
        write_chunk(stream, line.as_bytes())
    };

    let mut err: Option<anyhow::Error> = None;
    let mut finish = "length";
    let mut n_out = 0usize;
    let mut prompt_tokens = 0usize;
    {
        let (id2, name2) = (id.clone(), name.clone());
        let mut gate = ThinkGate::new();
        let res = remote_generate(tx, &messages, max_tokens, thinking, |piece| {
            match gate.push(piece) {
                Some(t) => send(&json!({
                    "id": id2, "object": "text_completion", "created": created,
                    "model": name2,
                    "choices": [{"index": 0, "text": t, "logprobs": Value::Null,
                                 "finish_reason": Value::Null}],
                })),
                None => Ok(()),
            }
        });
        if let Some(t) = gate.flush() {
            send(&json!({
                "id": id2, "object": "text_completion", "created": created,
                "model": name2,
                "choices": [{"index": 0, "text": t, "logprobs": Value::Null,
                             "finish_reason": Value::Null}],
            }))?;
        }
        match res {
            Ok(r) => {
                finish = r.finish;
                n_out = r.ids.len();
                prompt_tokens = r.prompt_tokens;
            }
            Err(e) => err = Some(e),
        }
    }
    if let Some(e) = err {
        send(&json!({"error": {"message": format!("{e:#}"), "type": "server_error"}}))?;
        end_chunked(stream)?;
        return Ok(());
    }

    send(&json!({
        "id": id, "object": "text_completion", "created": created, "model": name,
        "choices": [{"index": 0, "text": "", "logprobs": Value::Null,
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n_out,
                  "total_tokens": prompt_tokens + n_out},
    }))?;
    write_chunk(stream, b"data: [DONE]\n\n")?;
    end_chunked(stream)?;
    Ok(())
}

fn handle_chat_completions(tx: &mpsc::Sender<Job>, body: &Value, stream: &mut TcpStream) -> Result<()> {
    let messages = openai_messages(body)?;
    let max_tokens = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;
    let want_stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    // Qwen's template gates the reasoning block on `enable_thinking`; the
    // engine defaults it on, matching the checkpoint's own default.
    let thinking = body
        .get("enable_thinking")
        .or_else(|| body.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let id = format!("chatcmpl-{}", now_secs());
    let created = now_secs();
    let name = model_name();

    if !want_stream {
        let r = remote_generate(tx, &messages, max_tokens, thinking, |_| Ok(()))?;
        return respond_json(
            stream,
            200,
            &json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": visible(&r.text)},
                    "finish_reason": r.finish,
                }],
                "usage": {
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.ids.len(),
                    "total_tokens": r.prompt_tokens + r.ids.len(),
                },
            }),
        );
    }

    write_head(
        stream,
        200,
        "text/event-stream",
        "Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n",
    )?;
    let mut send = |v: &Value| -> Result<()> {
        let line = format!("data: {}\n\n", serde_json::to_string(v)?);
        write_chunk(stream, line.as_bytes())
    };

    send(&json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": name,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                     "finish_reason": null}],
    }))?;

    let mut err: Option<anyhow::Error> = None;
    let mut finish = "length";
    let mut n_out = 0usize;
    let mut prompt_tokens = 0usize;
    {
        let (id2, name2) = (id.clone(), name.clone());
        let mut gate = ThinkGate::new();
        let res = remote_generate(tx, &messages, max_tokens, thinking, |piece| {
            match gate.push(piece) {
                Some(t) => send(&json!({
                    "id": id2, "object": "chat.completion.chunk", "created": created,
                    "model": name2,
                    "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": null}],
                })),
                None => Ok(()),
            }
        });
        if let Some(t) = gate.flush() {
            send(&json!({
                "id": id2, "object": "chat.completion.chunk", "created": created,
                "model": name2,
                "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": null}],
            }))?;
        }
        match res {
            Ok(r) => {
                finish = r.finish;
                n_out = r.ids.len();
                prompt_tokens = r.prompt_tokens;
            }
            Err(e) => err = Some(e),
        }
    }
    if let Some(e) = err {
        // The 200 is already on the wire, so the only honest thing left is to
        // surface the failure in-band before terminating the stream.
        send(&json!({"error": {"message": format!("{e:#}"), "type": "server_error"}}))?;
        end_chunked(stream)?;
        return Ok(());
    }

    send(&json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n_out,
                  "total_tokens": prompt_tokens + n_out},
    }))?;
    write_chunk(stream, b"data: [DONE]\n\n")?;
    end_chunked(stream)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Anthropic-compatible routes
// ---------------------------------------------------------------------------

fn anthropic_messages(body: &Value) -> Result<Vec<ChatMessage>> {
    let arr = body
        .get("messages")
        .and_then(|m| m.as_array())
        .context("`messages` must be an array")?;
    let mut out = Vec::new();
    // Anthropic carries the system prompt as a top-level field, not a message.
    if let Some(sys) = body.get("system") {
        let s = flatten_content(Some(sys));
        if !s.is_empty() {
            out.push(text_message("system", &s));
        }
    }
    for m in arr {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        out.push(message(role, flatten_content(m.get("content"))));
    }
    Ok(out)
}

fn anthropic_stop_reason(finish: &str) -> &'static str {
    match finish {
        "stop" => "end_turn",
        _ => "max_tokens",
    }
}

fn handle_messages(tx: &mpsc::Sender<Job>, body: &Value, stream: &mut TcpStream) -> Result<()> {
    let messages = anthropic_messages(body)?;
    let max_tokens = body.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
    let want_stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let thinking = body
        .get("thinking")
        .map(|t| t.get("type").and_then(|v| v.as_str()) != Some("disabled"))
        .unwrap_or(true);
    let id = format!("msg_{}", now_secs());
    let name = model_name();

    if !want_stream {
        let r = remote_generate(tx, &messages, max_tokens, thinking, |_| Ok(()))?;
        return respond_json(
            stream,
            200,
            &json!({
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": name,
                "content": [{"type": "text", "text": visible(&r.text)}],
                "stop_reason": anthropic_stop_reason(r.finish),
                "stop_sequence": Value::Null,
                "usage": {"input_tokens": r.prompt_tokens, "output_tokens": r.ids.len()},
            }),
        );
    }

    write_head(
        stream,
        200,
        "text/event-stream",
        "Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n",
    )?;
    let mut ev = |name: &str, v: &Value| -> Result<()> {
        let line = format!("event: {name}\ndata: {}\n\n", serde_json::to_string(v)?);
        write_chunk(stream, line.as_bytes())
    };

    ev(
        "message_start",
        &json!({
            "type": "message_start",
            "message": {"id": id, "type": "message", "role": "assistant", "model": name,
                        "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null,
                        "usage": {"input_tokens": 0, "output_tokens": 0}},
        }),
    )?;
    ev(
        "content_block_start",
        &json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}),
    )?;

    let mut err = None;
    let mut finish = "length";
    let mut n_out = 0usize;
    {
        let mut gate = ThinkGate::new();
        let res = remote_generate(tx, &messages, max_tokens, thinking, |piece| {
            match gate.push(piece) {
                Some(t) => ev(
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": t}}),
                ),
                None => Ok(()),
            }
        });
        if let Some(t) = gate.flush() {
            ev(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": t}}),
            )?;
        }
        match res {
            Ok(r) => {
                finish = r.finish;
                n_out = r.ids.len();
            }
            Err(e) => err = Some(e),
        }
    }
    if let Some(e) = err {
        ev(
            "error",
            &json!({"type": "error", "error": {"type": "api_error",
                                               "message": format!("{e:#}")}}),
        )?;
        end_chunked(stream)?;
        return Ok(());
    }

    ev("content_block_stop", &json!({"type": "content_block_stop", "index": 0}))?;
    ev(
        "message_delta",
        &json!({"type": "message_delta",
                "delta": {"stop_reason": anthropic_stop_reason(finish),
                          "stop_sequence": Value::Null},
                "usage": {"output_tokens": n_out}}),
    )?;
    ev("message_stop", &json!({"type": "message_stop"}))?;
    end_chunked(stream)?;
    Ok(())
}

// ---------------------------------------------------------------------------

fn handle(tx: &mpsc::Sender<Job>, name: &str, req: &Request, stream: &mut TcpStream) -> Result<()> {
    let path = req.path.split('?').next().unwrap_or("/");
    match (req.method.as_str(), path) {
        ("GET", "/health") => {
            write_head(stream, 200, "text/plain", "Content-Length: 3\r\n")?;
            stream.write_all(b"ok\n")?;
            stream.flush()?;
            Ok(())
        }
        ("GET", "/v1/models") => respond_json(
            stream,
            200,
            &json!({"object": "list", "data": [{
                "id": name, "object": "model", "owned_by": "gb10-engine",
            }]}),
        ),
        ("POST", "/v1/chat/completions") => {
            let body = req.json()?;
            handle_chat_completions(tx, &body, stream)
        }
        ("POST", "/v1/completions") => {
            let body = req.json()?;
            handle_completions(tx, &body, stream)
        }
        ("POST", "/v1/messages") => {
            let body = req.json()?;
            handle_messages(tx, &body, stream)
        }
        (_, p) if p == "/v1/chat/completions" || p == "/v1/messages" => {
            respond_json(stream, 405, &json!({"error": {"message": "method not allowed"}}))
        }
        _ => respond_json(stream, 404, &json!({"error": {"message": "not found"}})),
    }
}


// ---------------------------------------------------------------------------
// Batching scheduler
// ---------------------------------------------------------------------------

/// The model name, published once at startup so request handlers do not need
/// the `Engine` (which lives on the scheduler thread).
///
/// This must be a process-wide `OnceLock`, not a `thread_local!`: it is written
/// on the main thread and read on every connection thread, and a thread-local
/// would give each reader its own empty copy -- which is exactly what happened,
/// and why `model` was `""` in every response.
static MODEL_NAME: OnceLock<String> = OnceLock::new();

fn model_name() -> String {
    MODEL_NAME.get().cloned().unwrap_or_default()
}

enum Msg {
    Token(String),
    Done(GenResult),
    Fail(String),
}

/// One queued request. `out` carries tokens back to the connection thread.
struct Job {
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    enable_thinking: bool,
    out: mpsc::Sender<Msg>,
}

/// Same signature and return type as `Engine::generate`, so the handlers below
/// did not have to change: it hands the work to the scheduler thread and
/// forwards whatever comes back.
fn remote_generate<F>(
    tx: &mpsc::Sender<Job>,
    messages: &[ChatMessage],
    max_tokens: usize,
    enable_thinking: bool,
    mut on_token: F,
) -> Result<GenResult>
where
    F: FnMut(&str) -> Result<()>,
{
    let (out, rx) = mpsc::channel();
    tx.send(Job { messages: messages.to_vec(), max_tokens, enable_thinking, out })
        .map_err(|_| anyhow::anyhow!("engine stopped"))?;
    loop {
        match rx.recv().map_err(|_| anyhow::anyhow!("engine stopped"))? {
            Msg::Token(p) => on_token(&p)?,
            Msg::Done(r) => return Ok(r),
            Msg::Fail(e) => anyhow::bail!("{e}"),
        }
    }
}

fn scheduler(mut eng: Engine, rx: mpsc::Receiver<Job>) {
    loop {
        // Block for one, then take whatever else is already waiting, so that
        // requests that arrive together decode together.
        let first = match rx.recv() {
            Ok(j) => j,
            Err(_) => return,
        };
        let mut group = vec![first];
        // Batching window. Without it the group is whatever happened to be
        // queued at the instant the first request arrived, which for clients
        // started together is 1 or 2 -- and then the whole group decodes at
        // roughly the single-stream rate instead of the batched one. Measured
        // 815 ms/step against the 287 ms/step that a full batch of 16 costs.
        std::thread::sleep(std::time::Duration::from_millis(25));
        while group.len() < eng.n_seq {
            match rx.try_recv() {
                Ok(j) => group.push(j),
                Err(_) => break,
            }
        }
        let k = group.len();
        if let Err(e) = run_group(&mut eng, group) {
            eprintln!("batch: {e:#}");
        } else if std::env::var_os("GB10_BATCH_LOG").is_some() {
            eprintln!("batch of {k}");
        }
    }
}

/// Prefill every request in the group into its own slot, then step them
/// together. A finished slot keeps being stepped with its own last token so the
/// slot indices stay put; compacting would mean moving per-slot recurrent state
/// and KV cache for no benefit.
fn run_group(eng: &mut Engine, group: Vec<Job>) -> Result<()> {
    let k = group.len();
    let Engine { dev, model, tok, tmpl, state, sc, ctx, .. } = eng;
    let timing = std::env::var_os("GB10_TIMING").is_some();
    let t_reset = std::time::Instant::now();
    state.reset(dev)?;
    if timing {
        eprintln!("  reset {:.2}s", t_reset.elapsed().as_secs_f64());
    }

    let mut next: Vec<u32> = Vec::with_capacity(k);
    let mut prompt_tokens = Vec::with_capacity(k);
    let mut texts = vec![String::new(); k];
    let mut ids: Vec<Vec<u32>> = vec![Vec::new(); k];
    let mut emitted = vec![0usize; k];
    let mut done = vec![false; k];
    let mut finish = vec!["length"; k];
    // Per-slot generation cap: what the request asked for, further limited by
    // what is left of the window after this slot's prompt. The second term is
    // load-bearing -- without it a long prompt plus a large `max_tokens`
    // appends past this slot's KV cache and into the next sequence's.
    let mut limit = vec![0usize; k];

    for (s, job) in group.iter().enumerate() {
        let opts =
            ChatTemplateOptions { enable_thinking: job.enable_thinking, ..Default::default() };
        let prompt = tmpl.render(&job.messages, &opts)?;
        let p = tok.encode(&prompt, true)?;
        if p.is_empty() || p.len() + 1 > *ctx {
            let _ = job.out.send(Msg::Fail(format!(
                "prompt is {} tokens but the context window is {ctx}",
                p.len()
            )));
            done[s] = true;
            prompt_tokens.push(p.len());
            next.push(0);
            continue;
        }
        prompt_tokens.push(p.len());
        limit[s] = job.max_tokens.min(*ctx - p.len());
        let mut nx = 0u32;
        let t_all = std::time::Instant::now();
        for (i, c) in p.chunks(PREFILL_CHUNK).enumerate() {
            let t0 = std::time::Instant::now();
            nx = model.prefill_seq(dev, c, state, sc, s)?;
            if timing {
                eprintln!(
                    "  slot {s} chunk {i} ({} tok, start {}): {:.2}s",
                    c.len(),
                    i * PREFILL_CHUNK,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        if timing {
            eprintln!(
                "  slot {s}: prefill {} tok in {:.2}s ({:.1} ms/tok)",
                p.len(),
                t_all.elapsed().as_secs_f64(),
                t_all.elapsed().as_secs_f64() * 1e3 / p.len() as f64
            );
        }
        next.push(nx);
    }

    let mut live = done.iter().filter(|d| !**d).count();
    while live > 0 {
        for s in 0..k {
            if done[s] {
                continue;
            }
            let t = next[s];
            if tok.is_eos(t) {
                finish[s] = "stop";
                done[s] = true;
                live -= 1;
                continue;
            }
            emitted[s] += 1;
            ids[s].push(t);
            let piece = tok.decode(&[t], true)?;
            texts[s].push_str(&piece);
            let _ = group[s].out.send(Msg::Token(piece));
            if emitted[s] >= limit[s] {
                done[s] = true;
                live -= 1;
            }
        }
        if live == 0 {
            break;
        }
        next = model.step_batch(dev, &next, state, sc)?;
    }

    for (s, job) in group.into_iter().enumerate() {
        let _ = job.out.send(Msg::Done(GenResult {
            ids: std::mem::take(&mut ids[s]),
            text: std::mem::take(&mut texts[s]),
            prompt_tokens: prompt_tokens[s],
            finish: finish[s],
        }));
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    eprintln!("gb10-server: loading {}", args.model.display());
    let t = std::time::Instant::now();
    let eng = Engine::new(&args)?;
    let _ = MODEL_NAME.set(eng.name.clone());
    let n_seq = eng.n_seq;
    eprintln!("gb10-server: ready in {:.1}s", t.elapsed().as_secs_f64());

    let (job_tx, job_rx) = mpsc::channel::<Job>();
    std::thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || scheduler(eng, job_rx))
        .context("spawning scheduler")?;

    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr).with_context(|| format!("binding {addr}"))?;
    eprintln!("gb10-server: listening on http://{addr}");
    eprintln!("  OpenAI:    POST http://{addr}/v1/chat/completions");
    eprintln!("  Anthropic: POST http://{addr}/v1/messages");
    eprintln!("  concurrent slots: {n_seq}");

    for conn in listener.incoming() {
        let mut stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let tx = job_tx.clone();
        std::thread::spawn(move || {
            let req = match read_request(&mut stream) {
                Ok(Some(r)) => r,
                Ok(None) => return,
                Err(e) => {
                    eprintln!("read: {e:#}");
                    return;
                }
            };
            let name = model_name();
            if let Err(e) = handle(&tx, &name, &req, &mut stream) {
                eprintln!("{} {}: {e:#}", req.method, req.path);
                let _ = respond_json(
                    &mut stream,
                    500,
                    &json!({"error": {"message": format!("{e:#}")}}),
                );
            }
        });
    }
    Ok(())
}
