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
use std::cell::RefCell;
use std::sync::mpsc;
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

/// Prompt tokens the KV cache and recurrent state are sized for.
const MAX_SEQ: usize = 2048;
/// Concurrent sequence slots the state is sized for.
const MAX_CONCURRENT: usize = 16;

struct Args {
    model: PathBuf,
    host: String,
    port: u16,
    model_name: String,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model = PathBuf::from("models/Qwen3.8-27B-NVFP4");
        let mut host = "127.0.0.1".to_string();
        let mut port = 8080u16;
        let mut model_name = "Qwen3.8-27B-NVFP4".to_string();
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--model" => model = PathBuf::from(it.next().context("--model needs a value")?),
                "--host" => host = it.next().context("--host needs a value")?,
                "--port" => port = it.next().context("--port needs a value")?.parse()?,
                "--name" => model_name = it.next().context("--name needs a value")?,
                other => anyhow::bail!("unknown argument: {other}"),
            }
        }
        Ok(Self { model, host, port, model_name })
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
        // Slots for concurrent sequences. The scheduler that uses them is not
        // wired up yet (the accept loop is still serial), but the state must be
        // sized for it first. `step_batch` takes its batch size from
        // `tokens.len()` and requires it to be <= this, so a single request
        // still runs with n_seq = 1 and costs nothing extra.
        let state = ModelState::new(&dev, &model, MAX_SEQ, MAX_CONCURRENT)?;
        let sc = Scratch::new(&dev, &text, MAX_SEQ)?;
        Ok(Self { dev, model, tok, tmpl, state, sc, name: args.model_name.clone() })
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
        anyhow::ensure!(
            ids.len() < MAX_SEQ,
            "prompt is {} tokens but the context window is {MAX_SEQ}",
            ids.len()
        );

        self.state.reset(&self.dev)?;
        let mut next = self.model.prefill(&self.dev, &ids, &mut self.state, &mut self.sc)?;

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
    let name = MODEL_NAME.with(|n| n.borrow().clone());

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
        let res = remote_generate(tx, &messages, max_tokens, thinking, |piece| {
            send(&json!({
                "id": id2, "object": "chat.completion.chunk", "created": created,
                "model": name2,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}],
            }))
        });
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
    let name = MODEL_NAME.with(|n| n.borrow().clone());

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
        let res = remote_generate(tx, &messages, max_tokens, thinking, |piece| {
            ev(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": piece}}),
            )
        });
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

thread_local! {
    /// The model name, published once so the request handlers do not need the
    /// `Engine` (which now lives on the scheduler thread).
    static MODEL_NAME: RefCell<String> = RefCell::new(String::new());
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
        while group.len() < MAX_CONCURRENT {
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
    let Engine { dev, model, tok, tmpl, state, sc, .. } = eng;
    state.reset(dev)?;

    let mut next: Vec<u32> = Vec::with_capacity(k);
    let mut prompt_tokens = Vec::with_capacity(k);
    let mut texts = vec![String::new(); k];
    let mut ids: Vec<Vec<u32>> = vec![Vec::new(); k];
    let mut emitted = vec![0usize; k];
    let mut done = vec![false; k];
    let mut finish = vec!["length"; k];

    for (s, job) in group.iter().enumerate() {
        let opts =
            ChatTemplateOptions { enable_thinking: job.enable_thinking, ..Default::default() };
        let prompt = tmpl.render(&job.messages, &opts)?;
        let p = tok.encode(&prompt, true)?;
        if p.is_empty() || p.len() >= MAX_SEQ {
            let _ = job.out.send(Msg::Fail(format!("prompt is {} tokens", p.len())));
            done[s] = true;
            prompt_tokens.push(p.len());
            next.push(0);
            continue;
        }
        prompt_tokens.push(p.len());
        next.push(model.prefill_seq(dev, &p, state, sc, s)?);
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
            if emitted[s] >= group[s].max_tokens {
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
    MODEL_NAME.with(|n| *n.borrow_mut() = eng.name.clone());
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
    eprintln!("  concurrent slots: {MAX_CONCURRENT}");

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
            let name = MODEL_NAME.with(|n| n.borrow().clone());
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
