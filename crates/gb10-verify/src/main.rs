//! # gb10-verify
//!
//! The correctness gate. Compares engine output against the frozen reference
//! traces in `fixtures/oracle/`, which were produced by the real
//! `transformers` Qwen3.5 implementation running on weights dequantized from
//! this same NVFP4 checkpoint (`tools/make_layer_fixtures.py`).
//!
//! The fixtures are a flat float32 blob plus a JSON index, so this binary
//! needs no zip/npy dependency.
//!
//! Exit status is non-zero on any failed comparison, which is what the loop's
//! correctness gate keys on.

use anyhow::{bail, Context, Result};
use gb10_core::chat::{text_message, ChatTemplate};
use gb10_core::config::ModelConfig;
use gb10_core::tokenizer::QwenTokenizer;
use gb10_cuda::Device;
use gb10_model::mtp::{Mtp, MtpState};
use gb10_model::{LayerState, Model, ModelState, Scratch, Store};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const LAYER_PREFIX: &str = "model.language_model.";

#[derive(serde::Deserialize, Debug, Clone)]
struct Entry {
    shape: Vec<usize>,
    offset: usize,
    count: usize,
}

/// A flat float32 fixture blob with a JSON index.
struct Fixture {
    data: Vec<f32>,
    index: HashMap<String, Entry>,
}

impl Fixture {
    fn load(stem: &Path) -> Result<Self> {
        let json = stem.with_extension("json");
        let bin = stem.with_extension("bin");
        let index: HashMap<String, Entry> = serde_json::from_str(
            &std::fs::read_to_string(&json)
                .with_context(|| format!("reading {}", json.display()))?,
        )
        .with_context(|| format!("parsing {}", json.display()))?;

        let raw = std::fs::read(&bin).with_context(|| format!("reading {}", bin.display()))?;
        if raw.len() % 4 != 0 {
            bail!("{}: length {} is not a multiple of 4", bin.display(), raw.len());
        }
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Self { data, index })
    }

    fn get(&self, name: &str) -> Result<&[f32]> {
        let e = self
            .index
            .get(name)
            .with_context(|| format!("fixture has no tensor {name:?}"))?;
        // `offset` is a byte offset into the blob; `data` is an f32 slice.
        let start = e.offset / 4;
        let end = start + e.count;
        if end > self.data.len() {
            bail!("{name}: index says {end} but blob has {}", self.data.len());
        }
        Ok(&self.data[start..end])
    }

    fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self
            .index
            .get(name)
            .with_context(|| format!("fixture has no tensor {name:?}"))?
            .shape)
    }
}

/// Error of `got` against `want`, measured against the reference's own scale.
///
/// A per-element relative error is useless here: dot products cancel, so
/// individual outputs land near zero while carrying fp32 rounding.
struct Diff {
    max_abs: f32,
    scale: f32,
    rel_to_scale: f32,
    max_rel_large: f32,
    worst: usize,
}

fn compare(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut max_abs = 0.0f32;
    let mut max_rel_large = 0.0f32;
    let mut worst = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        if w.abs() > 0.1 * scale {
            max_rel_large = max_rel_large.max(d / w.abs());
        }
    }
    Diff {
        max_abs,
        scale,
        rel_to_scale: if scale > 0.0 { max_abs / scale } else { max_abs },
        max_rel_large,
        worst,
    }
}

/// Run `n_seq` sequences both batched and one-at-a-time and require the
/// generated tokens to agree exactly.
///
/// The prompts are deliberately of *different lengths*. With identical prompts
/// every sequence would hold identical state at identical positions, and a
/// wrong per-sequence `base_stride` would read equivalent data and still
/// produce correct output -- the gate would pass while the batching was broken.
/// Different lengths put the sequences at different positions and in different
/// cache slots, which is what makes cross-talk observable.
fn batch_parity(args: &Args, n_seq: usize, n_new: usize) -> Result<bool> {
    let cfg = load_config(&args.model)?;
    let text = cfg.text_config.clone();
    let dev = Device::new(0)?;
    let model = Model::load_from(&dev, cfg, &args.model)?;
    let tok = QwenTokenizer::from_model_dir(&args.model)?;

    let mut prompts = Vec::with_capacity(n_seq);
    for i in 0..n_seq {
        let p = format!("{}The capital of France is", "token ".repeat(i * 4));
        prompts.push(tok.encode(&p, true)?);
    }

    // Reference: each sequence alone, one token at a time.
    let mut refs: Vec<Vec<u32>> = Vec::with_capacity(n_seq);
    {
        let mut st = ModelState::new(&dev, &model, args.max_seq, 1)?;
        let mut sc = Scratch::new(&dev, &text, args.max_seq)?;
        for p in &prompts {
            st.reset(&dev)?;
            let mut next = model.prefill(&dev, p, &mut st, &mut sc)?;
            let mut out = vec![next];
            for _ in 1..n_new {
                next = model.step(&dev, next, &mut st, &mut sc)?;
                out.push(next);
            }
            refs.push(out);
        }
    }

    // Batched: prefill each prompt into its own slot, then step all together.
    let mut st = ModelState::new(&dev, &model, args.max_seq, n_seq)?;
    let mut sc = Scratch::new(&dev, &text, args.max_seq)?;
    let mut next: Vec<u32> = Vec::with_capacity(n_seq);
    for (s, p) in prompts.iter().enumerate() {
        next.push(model.prefill_seq(&dev, p, &mut st, &mut sc, s)?);
    }
    let mut got: Vec<Vec<u32>> = vec![Vec::with_capacity(n_new); n_seq];
    let t0 = std::time::Instant::now();
    for i in 0..n_new {
        for s in 0..n_seq {
            got[s].push(next[s]);
        }
        if i + 1 == n_new {
            break;
        }
        next = model.step_batch(&dev, &next, &mut st, &mut sc)?;
    }
    dev.stream().synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    let steps = (n_new.saturating_sub(1)) as f64;
    if steps > 0.0 {
        println!(
            "batched decode: {:.2} tok/s aggregate ({} seq x {:.1} tok/s each), {:.1} ms/step",
            steps * n_seq as f64 / dt,
            n_seq,
            steps / dt,
            dt / steps * 1000.0
        );
    }

    let mut all_ok = true;
    for s in 0..n_seq {
        let agree = refs[s] == got[s];
        if agree {
            println!("  seq {s:2} (prompt {:4} tok): match", prompts[s].len());
        } else {
            let w = refs[s].iter().zip(&got[s]).position(|(a, b)| a != b).unwrap_or(0);
            println!(
                "  seq {s:2} (prompt {:4} tok): MISMATCH at token {w}\n    ref {:?}\n    got {:?}",
                prompts[s].len(),
                &refs[s][..(w + 4).min(refs[s].len())],
                &got[s][..(w + 4).min(got[s].len())]
            );
        }
        all_ok &= agree;
    }
    let n_match = (0..n_seq).filter(|&s| refs[s] == got[s]).count();
    println!("batch parity: {n_match}/{n_seq} sequences exact over {n_new} tokens");
    Ok(all_ok)
}

fn load_config(model_dir: &Path) -> Result<ModelConfig> {
    let p = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    let cfg: ModelConfig = serde_json::from_str(&raw).context("parsing config.json")?;
    Ok(cfg)
}

/// Run one decoder layer over the fixture prompt, token by token, and compare
/// the layer output *and* each internal stage the fixture captured.
///
/// Comparing stages rather than just the output is what makes a failure
/// actionable: `input_ln` isolates the norm convention, `out_proj`/`o_proj`
/// isolates the token mixer, `mlp` isolates SwiGLU.
fn layer_parity(args: &Args) -> Result<bool> {
    let cfg = load_config(&args.model)?;
    let text = cfg.text_config.clone();
    let dev = Device::new(0)?;
    let store = Store::open(&args.model, LAYER_PREFIX)?;

    let stem = args.fixtures.join(format!("layer{:02}", args.layer));
    let fx = Fixture::load(&stem)?;

    let x = fx.get("input")?;
    let shape = fx.shape("input")?;
    let (tokens, hidden) = (shape[0], shape[1]);
    if hidden != text.hidden_size {
        bail!("fixture hidden {hidden} != config hidden {}", text.hidden_size);
    }
    println!(
        "layer {} ({:?})  fixture {tokens} tokens x {hidden} hidden",
        args.layer, text.layer_types[args.layer]
    );

    let layer = store.layer(&dev, &text, args.layer)?;
    let mut state = LayerState::new(&dev, &text, &layer, args.max_seq, 1)?;
    let mut sc = Scratch::new_single(&dev, &text)?;

    let mut xbuf = dev.stream().alloc_zeros::<f32>(hidden)?;
    let mut obuf = dev.stream().alloc_zeros::<f32>(hidden)?;

    let n = tokens * hidden;
    let mut got = vec![0.0f32; n];
    let mut st_ln = vec![0.0f32; n];
    let mut st_proj = vec![0.0f32; n];
    let mut st_mlp = vec![0.0f32; n];
    let proj_name = if layer.is_delta() { "out_proj" } else { "o_proj" };
    let mut st_gnorm = vec![0.0f32; tokens * text.linear_value_dim()];
    // Engine-side stages, captured per token. The DeltaNet scratch buffers
    // retain every intermediate the reference stage dump provides, which is
    // what lets a failure be attributed to one kernel rather than the block.
    let qk_dim = text.linear_qk_dim();
    let conv_dim = qk_dim * 2 + text.linear_value_dim();
    let n_gnorm = text.linear_value_dim();
    let nv = text.linear_num_value_heads;

    let mut cap: HashMap<&str, Vec<f32>> = HashMap::new();
    let mut push = |k: &'static str, len: usize, v: Vec<f32>, t: usize| {
        cap.entry(k).or_insert_with(|| vec![0.0; tokens * len])[t * len..(t + 1) * len]
            .copy_from_slice(&v[..len]);
    };

    for t in 0..tokens {
        dev.stream()
            .memcpy_htod(&x[t * hidden..(t + 1) * hidden], &mut xbuf)?;
        layer.forward(&dev, &text, &xbuf, &mut obuf, &mut state, &mut sc, 0)?;
        dev.synchronize()?;
        let r = t * hidden..(t + 1) * hidden;
        got[r.clone()].copy_from_slice(&dev.stream().memcpy_dtov(&obuf)?);
        st_ln[r.clone()].copy_from_slice(&dev.stream().memcpy_dtov(&sc.hidden)?);
        st_proj[r.clone()].copy_from_slice(&dev.stream().memcpy_dtov(&sc.proj)?);
        st_mlp[r.clone()].copy_from_slice(&dev.stream().memcpy_dtov(&sc.down)?);
        if layer.is_delta() {
            let g = dev.stream().memcpy_dtov(&sc.gnorm)?;
            st_gnorm[t * n_gnorm..(t + 1) * n_gnorm].copy_from_slice(&g[..n_gnorm]);
        }

        if layer.is_delta() {
            push("proj_qkv", conv_dim, dev.stream().memcpy_dtov(&sc.qkv)?, t);
            push("proj_z", n_gnorm, dev.stream().memcpy_dtov(&sc.z)?, t);
            push("conv_out", conv_dim, dev.stream().memcpy_dtov(&sc.conv)?, t);
            let ln = dev.stream().memcpy_dtov(&sc.conv_ln)?;
            push("q_ln", qk_dim, ln[..qk_dim].to_vec(), t);
            push("k_ln", qk_dim, ln[qk_dim..2 * qk_dim].to_vec(), t);
            push("decay", nv, dev.stream().memcpy_dtov(&sc.decay)?, t);
            push("beta", nv, dev.stream().memcpy_dtov(&sc.beta)?, t);
            push("delta_out", n_gnorm, dev.stream().memcpy_dtov(&sc.attn)?, t);
            push("gnorm", n_gnorm, dev.stream().memcpy_dtov(&sc.gnorm)?, t);
        } else {
            let q_dim = text.num_attention_heads * text.head_dim;
            let kv_dim = text.num_key_value_heads * text.head_dim;
            push("q_proj", 2 * q_dim, dev.stream().memcpy_dtov(&sc.fused)?, t);
            push("v_proj", kv_dim, dev.stream().memcpy_dtov(&sc.vb)?, t);
            push("q_rope", q_dim, dev.stream().memcpy_dtov(&sc.q)?, t);
            push("k_rope", kv_dim, dev.stream().memcpy_dtov(&sc.kb_ln)?, t);
            push("gate", q_dim, dev.stream().memcpy_dtov(&sc.gate)?, t);
            push("attn_gated", q_dim, dev.stream().memcpy_dtov(&sc.attn)?, t);
        }
    }

    let mut ok = true;
    let mut stages: Vec<(&str, &Vec<f32>)> = vec![("input_ln", &st_ln)];
    if layer.is_delta() {
        stages.push(("gated_norm", &st_gnorm));
    }
    stages.push((proj_name, &st_proj));
    stages.push(("mlp", &st_mlp));
    stages.push(("output", &got));

    // If the fine-grained stage dump exists, compare those too — they localise
    // a failure to a single kernel.
    let stages_stem = args
        .fixtures
        .join(format!("layer{:02}_stages", args.layer));
    let fine = if stages_stem.with_extension("json").exists() {
        Some(Fixture::load(&stages_stem)?)
    } else {
        None
    };

    let mut checks = 0usize;
    let mut report = |name: &str, g: &[f32], w: &[f32]| {
        checks += 1;
        let d = compare(g, w);
        let good = d.rel_to_scale < args.tol_norm && d.max_rel_large < args.tol_rel;
        ok &= good;
        println!(
            "  {name:10} err/scale={:.3e}  max_rel={:.3e}  max|y|={:.4e}  {}",
            d.rel_to_scale,
            d.max_rel_large,
            d.scale,
            if good { "OK" } else { "FAIL" }
        );
    };

    for (name, g) in stages {
        if let Ok(w) = fx.get(name) {
            report(name, g, w);
        }
    }
    if let Some(fine) = &fine {
        for name in [
            "proj_qkv", "proj_z", "conv_out", "q_ln", "k_ln", "decay", "beta", "delta_out",
            "gnorm", "q_proj", "v_proj", "q_rope", "k_rope", "gate", "attn_gated",
        ] {
            if let (Some(g), Ok(w)) = (cap.get(name), fine.get(name)) {
                report(name, g, w);
            }
        }
    }

    // A gate that silently compares nothing would report success forever.
    if checks == 0 {
        bail!(
            "no stages compared for layer {} — fixtures missing under {}?",
            args.layer,
            args.fixtures.display()
        );
    }
    if !ok {
        println!(
            "  (tolerance: err/scale < {:e}, max_rel < {:e})",
            args.tol_norm, args.tol_rel
        );
    }
    println!("  ({checks} stage checks)");
    Ok(ok)
}


/// End-to-end greedy decode with the real 64-layer model.
///
/// This is the M3 gate: it exercises the whole stack (embedding gather, all 48
/// DeltaNet + 16 attention layers, final norm, NVFP4 lm_head, argmax) rather
/// than a single layer.
///
/// If `fixtures/oracle/greedy_tokens.json` exists it is used as the reference:
/// the prompt token ids must match exactly and the greedy continuation must
/// agree token for token.
fn generate(args: &Args, prompt: &str, n_new: usize) -> Result<bool> {
    let cfg = load_config(&args.model)?;
    let text = cfg.text_config.clone();
    let dev = Device::new(0)?;

    let tok = QwenTokenizer::from_model_dir(&args.model)?;
    let tmpl = ChatTemplate::from_model_dir(&args.model)?;

    // The oracle fixes the prompt and the number of steps, so both runs are
    // compared on identical input.
    let oracle_path = args.fixtures.join("greedy_tokens.json");
    let oracle: Option<serde_json::Value> = if oracle_path.exists() {
        Some(serde_json::from_str(&std::fs::read_to_string(&oracle_path)?)?)
    } else {
        None
    };
    let (prompt, n_new) = match &oracle {
        Some(o) => (
            o["prompt"].as_str().unwrap_or(prompt).to_string(),
            o["n_new"].as_u64().unwrap_or(n_new as u64) as usize,
        ),
        None => (prompt.to_string(), n_new),
    };

    let ids = if args.raw {
        tok.encode(&prompt, false)?
    } else {
        let rendered = tmpl.render(
            &[text_message("user", &prompt)],
            &gb10_core::chat::ChatTemplateOptions::default(),
        )?;
        tok.encode(&rendered, false)?
    };
    println!("prompt: {} tokens", ids.len());

    if let Some(o) = &oracle {
        let want: Vec<u32> = o["prompt_ids"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
            .unwrap_or_default();
        if !want.is_empty() && want != ids {
            bail!(
                "prompt tokenisation differs from the oracle\n  engine: {ids:?}\n  oracle: {want:?}"
            );
        }
        println!("prompt ids match the oracle ({} tokens)", ids.len());
    }

    let t0 = std::time::Instant::now();
    let model = Model::load_from(&dev, cfg.clone(), &args.model)?;
    println!(
        "model loaded in {:.1}s  ({:.2} GB streamed per token)",
        t0.elapsed().as_secs_f64(),
        model.traffic_bytes() as f64 / 1e9
    );

    let mut state = ModelState::new(&dev, &model, args.max_seq, 1)?;
    let mut sc = Scratch::new(&dev, &text, 512)?;

    let t1 = std::time::Instant::now();
    let mut next = model.prefill(&dev, &ids, &mut state, &mut sc)?;
    println!(
        "TTFT {:.1} ms ({} prompt tokens)",
        t1.elapsed().as_secs_f64() * 1e3,
        ids.len()
    );

    let mut out = Vec::with_capacity(n_new);
    let mut acc = gb10_model::model::PhaseTimes::default();
    let t2 = std::time::Instant::now();
    for _ in 0..n_new {
        if tok.is_eos(next) {
            break;
        }
        out.push(next);
        if args.profile {
            let (n, pt) = model.step_timed(&dev, next, &mut state, &mut sc)?;
            next = n;
            acc.embed += pt.embed;
            acc.delta_layers += pt.delta_layers;
            acc.attn_layers += pt.attn_layers;
            acc.final_norm += pt.final_norm;
            acc.lm_head += pt.lm_head;
            acc.argmax += pt.argmax;
        } else {
            next = model.step(&dev, next, &mut state, &mut sc)?;
        }
    }
    let dt = t2.elapsed();
    if args.profile && !out.is_empty() {
        let n = out.len() as f64;
        println!("--- per-phase breakdown (mean ms/token) ---");
        println!("  embed        {:7.3}", acc.embed / n);
        println!("  delta layers {:7.3}   (48 layers)", acc.delta_layers / n);
        println!("  attn layers  {:7.3}   (16 layers)", acc.attn_layers / n);
        println!("  final norm   {:7.3}", acc.final_norm / n);
        println!("  lm_head      {:7.3}", acc.lm_head / n);
        println!("  argmax       {:7.3}", acc.argmax / n);
        println!("  sum          {:7.3}", acc.total() / n);
        println!("  wall         {:7.3}", dt.as_secs_f64() * 1e3 / n);
    }
    println!(
        "decoded {} tokens in {:.3}s -> {:.2} tok/s",
        out.len(),
        dt.as_secs_f64(),
        out.len() as f64 / dt.as_secs_f64()
    );
    println!("ids:  {:?}", &out[..out.len().min(24)]);
    println!("text: {:?}", tok.decode(&out, false)?);

    // High-power determinism check: the same prompt, the same weights, fresh
    // state every time. A single run cannot distinguish "deterministic" from
    // "got lucky"; this can, and it is the only way to tell a real numerical
    // flip from a race.
    if args.repeat > 1 {
        let mut identical = 0usize;
        let t_r = std::time::Instant::now();
        for rep in 1..args.repeat {
            let mut st = ModelState::new(&dev, &model, args.max_seq, 1)?;
            let mut sc2 = Scratch::new(&dev, &text, 512)?;
            let mut nx = model.prefill(&dev, &ids, &mut st, &mut sc2)?;
            let mut o2 = Vec::with_capacity(n_new);
            for _ in 0..n_new {
                if tok.is_eos(nx) {
                    break;
                }
                o2.push(nx);
                nx = model.step(&dev, nx, &mut st, &mut sc2)?;
            }
            if o2 == out {
                identical += 1;
            } else {
                let k = o2.iter().zip(&out).position(|(a, b)| a != b);
                println!(
                    "  rep {rep}: DIFFERS first at {:?}  got {:?}",
                    k,
                    &o2[..o2.len().min(24)]
                );
            }
        }
        println!(
            "determinism: {}/{} extra repeats identical to run 0 ({:.1}s total)",
            identical,
            args.repeat - 1,
            t_r.elapsed().as_secs_f64()
        );
        if identical != args.repeat - 1 {
            bail!("non-deterministic: {} of {} repeats differed", args.repeat - 1 - identical, args.repeat - 1);
        }
    }

    let Some(o) = &oracle else {
        println!("no {} — nothing to compare against", oracle_path.display());
        return Ok(true);
    };
    let want: Vec<u32> = o["tokens"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
        .unwrap_or_default();
    let n = out.len().min(want.len());
    let agree = (0..n).filter(|&i| out[i] == want[i]).count();
    let first_bad = (0..n).find(|&i| out[i] != want[i]);
    let rate = if n == 0 { 0.0 } else { agree as f64 / n as f64 };
    println!(
        "oracle agreement: {agree}/{n} ({:.1}%)  reference={}",
        rate * 100.0,
        o["reference"].as_str().unwrap_or("?")
    );
    match first_bad {
        None if out.len() == want.len() => {
            println!("  exact match");
            Ok(true)
        }
        None => {
            println!(
                "  matched on the first {n}, engine produced {} vs oracle {}",
                out.len(),
                want.len()
            );
            Ok(out.len() >= want.len())
        }
        Some(i) => {
            println!(
                "  first divergence at index {i}: engine {} vs oracle {}",
                out[i], want[i]
            );
            Ok(rate >= args.min_agree)
        }
    }
}

#[derive(Clone)]
struct Args {
    model: PathBuf,
    fixtures: PathBuf,
    layer: usize,
    max_seq: usize,
    tol_norm: f32,
    tol_rel: f32,
    quiet: bool,
    raw: bool,
    prompt: String,
    n_new: usize,
    profile: bool,
    /// Number of sequences for the batch-parity gate.
    n_seq: usize,
    /// Minimum token agreement with the bf16 oracle before the gate passes.
    /// The oracle is bf16-rounded, so bit-exactness is not expected; see
    /// docs/TARGETS.md T7.
    min_agree: f64,
    /// `generate`: run the whole prefill+decode this many times from fresh
    /// state and require every run to produce the same tokens. One run proves
    /// nothing about a race; this is the high-power version of the question
    /// "is the compute path deterministic".
    repeat: usize,
    /// `perplexity`: file of token ids (llama-tokenize `--ids` format).
    tokens: Option<PathBuf>,
    /// `perplexity`: raw text to tokenize with the engine's own tokenizer, used
    /// to cross-check the token stream against the one llama.cpp produced.
    text: Option<PathBuf>,
    /// `perplexity`: window length. llama-perplexity hardcodes 512.
    ctx: usize,
    /// `perplexity`: number of windows to score, -1 for all.
    chunks: i64,
    /// `perplexity`: where to write the JSON result.
    out: Option<PathBuf>,
    /// `choice`: JSONL of multiple-choice questions (id, subject, question,
    /// options, answer).
    jsonl: Option<PathBuf>,
    /// `choice`: score at most this many questions; -1 for all.
    limit: i64,
}

fn parse_args() -> Result<(String, Args)> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // The loop invokes `gb10-verify --oracle ... --model ...` with no
    // subcommand, so a leading flag means "run the default gate".
    let (cmd, rest) = match argv.first() {
        Some(a) if !a.starts_with('-') => (a.clone(), argv[1..].to_vec()),
        _ => ("layer-parity".to_string(), argv.clone()),
    };

    let mut a = Args {
        model: PathBuf::from("models/Qwen3.8-27B-NVFP4"),
        fixtures: PathBuf::from("fixtures/oracle"),
        layer: usize::MAX,
        max_seq: 512,
        tol_norm: 2e-3,
        tol_rel: 5e-2,
        quiet: false,
        raw: false,
        prompt: String::new(),
        n_new: 32,
        n_seq: 4,
        min_agree: 0.85,
        repeat: 1,
        profile: false,
        tokens: None,
        text: None,
        ctx: 512,
        chunks: -1,
        out: None,
        jsonl: None,
        limit: -1,
    };
    let mut i = 0;
    while i < rest.len() {
        let k = rest[i].as_str();
        let mut val = || -> Result<String> {
            rest.get(i + 1)
                .cloned()
                .with_context(|| format!("flag {k} needs a value"))
        };
        match k {
            "--model" => {
                a.model = PathBuf::from(val()?);
                i += 2;
            }
            // `--oracle` is the name the loop script uses.
            "--fixtures" | "--oracle" => {
                a.fixtures = PathBuf::from(val()?);
                i += 2;
            }
            "--layer" => {
                a.layer = val()?.parse()?;
                i += 2;
            }
            "--max-seq" => {
                a.max_seq = val()?.parse()?;
                i += 2;
            }
            "--n-seq" => {
                a.n_seq = val()?.parse()?;
                i += 2;
            }
            "--tol-norm" => {
                a.tol_norm = val()?.parse()?;
                i += 2;
            }
            "--tol-rel" => {
                a.tol_rel = val()?.parse()?;
                i += 2;
            }
            "--quiet" => {
                a.quiet = true;
                i += 1;
            }
            "--raw" => {
                a.raw = true;
                i += 1;
            }
            "--prompt" => {
                a.prompt = val()?;
                i += 2;
            }
            "--n" => {
                a.n_new = val()?.parse()?;
                i += 2;
            }
            "--profile" => {
                a.profile = true;
                i += 1;
            }
            "--min-agree" => {
                a.min_agree = val()?.parse()?;
                i += 2;
            }
            "--repeat" => {
                a.repeat = val()?.parse()?;
                i += 2;
            }
            "--tokens" => {
                a.tokens = Some(PathBuf::from(val()?));
                i += 2;
            }
            "--jsonl" => {
                a.jsonl = Some(PathBuf::from(val()?));
                i += 2;
            }
            "--limit" => {
                a.limit = val()?.parse()?;
                i += 2;
            }
            "--text" => {
                a.text = Some(PathBuf::from(val()?));
                i += 2;
            }
            "--ctx" => {
                a.ctx = val()?.parse()?;
                i += 2;
            }
            "--chunks" => {
                a.chunks = val()?.parse()?;
                i += 2;
            }
            "--out" => {
                a.out = Some(PathBuf::from(val()?));
                i += 2;
            }
            other => bail!("unknown flag {other}"),
        }
    }
    Ok((cmd, a))
}

/// Loads the MTP head and reports what it costs, without running it yet.
///
/// The head is unquantized BF16 while the decoder it drafts for is FP4, so the
/// first thing worth knowing is what fraction of a decoder step the head costs.
/// That ratio, not the acceptance rate, is what bounds the payoff of
/// speculative decoding on this machine.
fn mtp_probe(args: &Args) -> Result<()> {
    let cfg = load_config(&args.model)?;
    let dev = Device::new(0)?;
    // The MTP tensors sit at the top level of the checkpoint, outside the
    // `model.language_model.` prefix the decoder uses.
    let store = Store::open(&args.model, "")?;
    let model = Model::load_from(&dev, cfg.clone(), &args.model)?;
    let mtp = Mtp::load(&store, &dev, &cfg.text_config)?;

    let decoder = model.traffic_bytes();
    let head = mtp.traffic_bytes();
    println!("== MTP head ==\n");
    println!("  mtp.fc              [{} x {}]", mtp.fc.n, mtp.fc.k);
    println!("  self_attn.q_proj    [{} x {}]", mtp.layer.q_proj.n, mtp.layer.q_proj.k);
    println!("  self_attn.k_proj    [{} x {}]", mtp.layer.k_proj.n, mtp.layer.k_proj.k);
    println!("  self_attn.v_proj    [{} x {}]", mtp.layer.v_proj.n, mtp.layer.v_proj.k);
    println!("  self_attn.o_proj    [{} x {}]", mtp.layer.o_proj.n, mtp.layer.o_proj.k);
    println!("  mlp.gate_proj       [{} x {}]", mtp.layer.mlp.gate.n, mtp.layer.mlp.gate.k);
    println!("  mlp.down_proj       [{} x {}]", mtp.layer.mlp.down.n, mtp.layer.mlp.down.k);
    println!();
    println!("  decoder traffic     {decoder:>10} B/token");
    println!(
        "  mtp head traffic    {head:>10} B/token   ({:.1}% of a decoder step)",
        100.0 * head as f64 / decoder as f64
    );
    println!();

    // ---- does the head actually predict the decoder's next token? ----------
    // Two candidate inputs, because the checkpoint is ambiguous about which one
    // `pre_fc_norm_hidden` expects: the post-final-norm hidden the lm_head
    // consumes, or the pre-norm residual stream leaving the last decoder
    // layer (which is what DeepSeek-style MTP heads take). Measuring both
    // settles it without guessing. Each variant needs its own KV cache, since
    // a ruling-out pass must not consume the other's positions.
    let tok = QwenTokenizer::from_model_dir(&args.model)?;
    let text = cfg.text_config.clone();
    let mut st = ModelState::new(&dev, &model, args.max_seq, 1)?;
    let mut sc = Scratch::new(&dev, &text, args.max_seq)?;
    let mut a = MtpState::new(&dev, &text, args.max_seq, model.vocab_size())?;
    let mut b = MtpState::new(&dev, &text, args.max_seq, model.vocab_size())?;
    let mut c = MtpState::new(&dev, &text, args.max_seq, model.vocab_size())?;
    // Control: feed a zero hidden. If acceptance is unchanged, the hidden half
    // of the concatenation is not reaching the prediction at all.
    let zeros: gb10_cuda::CudaSlice<f32> = dev.stream().alloc_zeros::<f32>(text.hidden_size)?;
    let mut idx: gb10_cuda::CudaSlice<i32> = dev.stream().alloc_zeros::<i32>(1)?;

    let probe_prompt = if args.prompt.is_empty() {
        "What is the capital of France?".to_string()
    } else {
        args.prompt.clone()
    };
    println!("  prompt: {probe_prompt:?}");
    let ids = tok.encode(&probe_prompt, true)?;
    let mut next = model.prefill_seq(&dev, &ids, &mut st, &mut sc, 0)?;

    let (mut hit_post, mut hit_pre, mut hit_zero, mut total) = (0usize, 0usize, 0usize, 0usize);
    for _ in 0..args.n_new {
        // `st.normed` is the post-final-norm hidden the lm_head just consumed;
        // `st.a` is the residual stream leaving the last decoder layer.
        mtp.forward(&dev, &text, &st.normed, next, &model.embed, &model.lm_head, &mut a)?;
        dev.ops().argmax(&dev, &a.logits, &mut idx, model.vocab_size())?;
        let d_post = dev.stream().memcpy_dtov(&idx)?[0] as u32;

        mtp.forward(&dev, &text, &st.a, next, &model.embed, &model.lm_head, &mut b)?;
        dev.ops().argmax(&dev, &b.logits, &mut idx, model.vocab_size())?;
        let d_pre = dev.stream().memcpy_dtov(&idx)?[0] as u32;

        mtp.forward(&dev, &text, &zeros, next, &model.embed, &model.lm_head, &mut c)?;
        dev.ops().argmax(&dev, &c.logits, &mut idx, model.vocab_size())?;
        let d_zero = dev.stream().memcpy_dtov(&idx)?[0] as u32;

        let actual = model.step(&dev, next, &mut st, &mut sc)?;
        total += 1;
        if d_post == actual { hit_post += 1; }
        if d_pre == actual { hit_pre += 1; }
        if d_zero == actual { hit_zero += 1; }
        next = actual;
    }

    println!("== MTP draft acceptance over {} greedy steps ==", total);
    println!("  post-final-norm hidden : {:>3}/{total} = {:.1}%", hit_post, 100.0 * hit_post as f64 / total as f64);
    println!("  pre-norm residual      : {:>3}/{total} = {:.1}%", hit_pre, 100.0 * hit_pre as f64 / total as f64);
    println!("  hidden half ZEROED     : {:>3}/{total} = {:.1}%", hit_zero, 100.0 * hit_zero as f64 / total as f64);
    println!();
    println!("mtp-probe: OK");
    Ok(())
}

/// Speculative decoding with the MTP head, gated on reproducing *exactly* the
/// tokens plain greedy decoding produces.
///
/// Greedy accept/reject cannot change the output -- a rejected draft is thrown
/// away and the decoder's own token is emitted instead -- so the sequence is
/// identical by construction. The gate checks that rather than trusting it,
/// because an off-by-one in the draft/verify handshake still yields fluent
/// text and a plausible-looking acceptance rate.
/// Batched MTP speculative decoding, gated on reproducing plain greedy output.
///
/// This is where the three verified prerequisites are composed:
///
///   1. a forward from a non-empty KV cache (`attn_prefill` `start`/`kv_base`),
///   2. per-row scoring (`Model::all_logits`),
///   3. a reversible round (`snapshot_recurrent` / `restore_recurrent`).
///
/// The naive per-token loop measured 0.95x because every emitted token still
/// needed its own 17.6 GB weight read. Batching is what changes that: one
/// forward over `K` drafted tokens reads the weights **once**, so a round costs
/// two weight reads (verify + commit) and emits `accepted + 2` tokens.
/// Sequential decoding would need `accepted + 2` reads for the same tokens.
fn mtp_generate(args: &Args, n_new: usize) -> Result<bool> {
    use gb10_cuda::CudaSlice;

    // Drafts verified per round. Every extra draft adds ~0.85 GB of head work
    // against the 17.6 GB the verify forward costs anyway.
    const K: usize = 4;

    let cfg = load_config(&args.model)?;
    let text = cfg.text_config.clone();
    let dev = Device::new(0)?;
    let model = Model::load_from(&dev, cfg, &args.model)?;
    let tok = QwenTokenizer::from_model_dir(&args.model)?;
    let store = Store::open(&args.model, "")?;
    let mtp = Mtp::load(&store, &dev, &text)?;

    let prompt = if args.prompt.is_empty() {
        "What is the capital of France?".to_string()
    } else {
        args.prompt.clone()
    };
    let ids = tok.encode(&prompt, false)?;

    // --- reference: plain greedy ---
    let mut st = ModelState::new(&dev, &model, args.max_seq, 1)?;
    let mut sc = Scratch::new(&dev, &text, args.max_seq)?;
    let mut next = model.prefill(&dev, &ids, &mut st, &mut sc)?;
    let t0 = std::time::Instant::now();
    let mut want = Vec::new();
    for _ in 0..n_new {
        if tok.is_eos(next) {
            break;
        }
        want.push(next);
        next = model.step(&dev, next, &mut st, &mut sc)?;
    }
    let dt_ref = t0.elapsed().as_secs_f64();

    // --- speculative ---
    let mut st = ModelState::new(&dev, &model, args.max_seq, 1)?;
    let mut sc = Scratch::new(&dev, &text, args.max_seq)?;
    let mut ms = MtpState::new(&dev, &text, args.max_seq, model.vocab_size())?;
    let h = text.hidden_size;
    // Invariant at the top of a round: `chain[..h]` is the hidden at position P
    // and `cur` is the token at P+1, predicted but not yet emitted.
    let mut chain: CudaSlice<f32> = dev.stream().alloc_zeros::<f32>(h)?;
    let mut cur = model.prefill(&dev, &ids, &mut st, &mut sc)?;

    let t1 = std::time::Instant::now();
    let mut got: Vec<u32> = Vec::new();
    let (mut drafted, mut accepted, mut rounds) = (0usize, 0usize, 0usize);
    while got.len() < n_new && rounds < n_new * 4 {
        if tok.is_eos(cur) {
            break;
        }
        // Hidden at position P is the last row the decoder wrote. The row count
        // has to come from the state, not the prompt length: after the first
        // round the context has grown past `ids.len()`, and a stale count reads
        // a row from the middle of the sequence -- which still yields fluent
        // drafts, just wrong ones.
        dev.ops()
            .copy_last_row(&dev, &st.normed, &mut chain, st.n_tokens, h)?;

        // 1. Draft K tokens, chaining each draft's own hidden into the next.
        let mut drafts: Vec<u32> = Vec::with_capacity(K);
        let mut t = cur;
        for _ in 0..K {
            mtp.forward(&dev, &text, &chain, t, &model.embed, &model.lm_head, &mut ms)?;
            let d = dev.stream().clone_dtoh(&ms.logits)?;
            let d = d
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            drafts.push(d);
            dev.stream().memcpy_dtod(&ms.out, &mut chain)?;
            t = d;
        }

        // 2. Verify: feed the known token plus K-1 drafts, score every row.
        st.snapshot_recurrent(&dev)?;
        let mut feed: Vec<u32> = Vec::with_capacity(K);
        feed.push(cur);
        feed.extend_from_slice(&drafts[..K - 1]);
        model.prefill_seq(&dev, &feed, &mut st, &mut sc, 0)?;
        model.all_logits(&dev, &mut st, K)?;
        let pred = dev.stream().clone_dtoh(&st.idx_all)?;

        // 3. Accept the longest matching prefix.
        let mut acc = 0usize;
        while acc < K - 1 && pred[acc] as u32 == drafts[acc] {
            acc += 1;
        }
        let next_true = pred[acc] as u32;
        drafted += K - 1;
        accepted += acc;

        // 4. Rewind, then commit only what survived. The Gated DeltaNet
        //    recurrence cannot be unwound, so the rejected drafts must never
        //    reach the real state -- hence the snapshot and this replay.
        st.restore_recurrent(&dev)?;
        let mut commit: Vec<u32> = Vec::with_capacity(acc + 2);
        commit.push(cur);
        commit.extend_from_slice(&drafts[..acc]);
        commit.push(next_true);
        got.push(cur);
        for d in &drafts[..acc] {
            got.push(*d);
        }
        got.push(next_true);
        cur = model.prefill_seq(&dev, &commit, &mut st, &mut sc, 0)?;
        rounds += 1;
    }
    let dt_spec = t1.elapsed().as_secs_f64();
    // A round emits `accepted + 2` tokens at once, so it can overshoot.
    got.truncate(n_new);

    let rps = want.len() as f64 / dt_ref;
    let sps = got.len() as f64 / dt_spec;
    println!("== MTP speculative decoding (batched verify, K={K}) ==");
    println!("  prompt: {prompt:?}");
    println!("  greedy reference : {:>3} tokens in {:.3}s -> {:.2} tok/s", want.len(), dt_ref, rps);
    println!("  MTP speculative  : {:>3} tokens in {:.3}s -> {:.2} tok/s", got.len(), dt_spec, sps);
    println!(
        "  draft acceptance : {accepted}/{drafted} = {:.1}%  over {rounds} rounds",
        100.0 * accepted as f64 / drafted.max(1) as f64
    );
    println!("  speedup          : {:.2}x", sps / rps);

    let same = got == want;
    println!("  token-exact vs greedy: {}", if same { "YES" } else { "NO" });
    if !same {
        let d = want.iter().zip(&got).position(|(a, b)| a != b);
        println!(
            "  first difference at {:?}: want {:?} got {:?}",
            d,
            d.map(|i| want[i]),
            d.map(|i| got[i])
        );
    }
    println!();
    Ok(same)
}

/// Prefill the same prompt in one shot and in two chunks, and require the
/// results to agree.
///
/// This is the only test that drives `start > 0` in `attn_prefill`: the second
/// chunk attends over a cache that already holds the first chunk. Without it
/// the new parameter is exercised only at `start == 0`, where it is
/// indistinguishable from the old hardcoded behaviour -- so a green
/// `batch-parity` would say nothing about it.
fn chunked_prefill(args: &Args, n_dec: usize) -> Result<bool> {
    let cfg = load_config(&args.model)?;
    let text = cfg.text_config.clone();
    let dev = Device::new(0)?;
    let model = Model::load_from(&dev, cfg, &args.model)?;
    let tok = QwenTokenizer::from_model_dir(&args.model)?;

    let prompt = if args.prompt.is_empty() {
        "The quick brown fox jumps over the lazy dog. Count from one to ten and \
         then explain, in two sentences, why the sky appears blue at midday."
            .to_string()
    } else {
        args.prompt.clone()
    };
    let ids = tok.encode(&prompt, false)?;
    if ids.len() < 4 {
        bail!("prompt too short to split: {} tokens", ids.len());
    }
    let split = ids.len() / 2;

    // --- one shot ---
    let mut st = ModelState::new(&dev, &model, args.max_seq, 1)?;
    let mut sc = Scratch::new(&dev, &text, args.max_seq)?;
    let mut one = vec![model.prefill_seq(&dev, &ids, &mut st, &mut sc, 0)?];
    for _ in 1..n_dec {
        let t = *one.last().unwrap();
        one.push(model.step(&dev, t, &mut st, &mut sc)?);
    }

    // --- two chunks: the second one starts from a non-empty cache ---
    let mut st2 = ModelState::new(&dev, &model, args.max_seq, 1)?;
    let mut sc2 = Scratch::new(&dev, &text, args.max_seq)?;
    let mut two = vec![model.prefill_seq(&dev, &ids[..split], &mut st2, &mut sc2, 0)?];
    two[0] = model.prefill_seq(&dev, &ids[split..], &mut st2, &mut sc2, 0)?;
    for _ in 1..n_dec {
        let t = *two.last().unwrap();
        two.push(model.step(&dev, t, &mut st2, &mut sc2)?);
    }

    // `all_logits` scores every row; the row `prefill_seq` scored on its own
    // must agree with it, or the verify path is reading different hidden states
    // than the decode path.
    let mut st3 = ModelState::new(&dev, &model, args.max_seq, 1)?;
    let mut sc3 = Scratch::new(&dev, &text, args.max_seq)?;
    let last = model.prefill_seq(&dev, &ids, &mut st3, &mut sc3, 0)?;
    model.all_logits(&dev, &mut st3, ids.len())?;
    let rows = dev.stream().clone_dtoh(&st3.idx_all)?;
    let row_last = rows[ids.len() - 1] as u32;
    let rows_ok = row_last == last;

    // Snapshot/restore must make a decoder step exactly reversible -- that is
    // what lets a speculative round throw away rejected drafts, since the
    // Gated DeltaNet recurrence cannot be rewound the way the KV cache can.
    st3.snapshot_recurrent(&dev)?;
    let first = model.step(&dev, last, &mut st3, &mut sc3)?;
    st3.restore_recurrent(&dev)?;
    let again = model.step(&dev, last, &mut st3, &mut sc3)?;
    let rev_ok = first == again;
    println!("  snapshot/restore: step gave {first} then {again} -> {}", if rev_ok { "reversible" } else { "DIVERGED" });
    println!("  all_logits row {} = {}, prefill_seq = {}", ids.len() - 1, row_last, last);

    println!("== chunked prefill (start > 0) ==");
    println!("  prompt {} tokens, split {}+{}", ids.len(), split, ids.len() - split);
    println!("  one shot : {:?}", &one[..one.len().min(8)]);
    println!("  two chunk: {:?}", &two[..two.len().min(8)]);
    let same = one == two && rows_ok && rev_ok;
    println!("  decoded  : {:?}", tok.decode(&one, false)?);
    println!("  agree: {}", if same { "YES" } else { "NO" });
    println!();
    Ok(same)
}

/// The measurement that decides whether batched MTP verification can ever pay.
///
/// Both paths read the same 17.6 GB of weights; the question is how the cost
/// scales with rows. If one `t`-row `prefill_seq` costs about the same as one
/// single-row `step`, then verifying `t` drafts in one forward is a real `t`x
/// saving. If it costs close to `t` steps, the scheme cannot pay off here no
/// matter how cheap the head is.
fn forward_cost(args: &Args) -> Result<()> {
    let cfg = load_config(&args.model)?;
    let text = cfg.text_config.clone();
    let dev = Device::new(0)?;
    let model = Model::load_from(&dev, cfg, &args.model)?;
    let tok = QwenTokenizer::from_model_dir(&args.model)?;
    let ids = tok.encode("The quick brown fox jumps over the lazy dog.", false)?;

    println!("== cost of a t-row forward vs t single-row steps ==");
    println!("     t   t x step(ms)   one t-row fwd(ms)   ratio   per-row");
    for t in [1usize, 2, 4, 8, 16, 32, 64] {
        let mut st = ModelState::new(&dev, &model, args.max_seq, 1)?;
        let mut sc = Scratch::new(&dev, &text, args.max_seq)?;
        let mut next = model.prefill(&dev, &ids, &mut st, &mut sc)?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..t {
            next = model.step(&dev, next, &mut st, &mut sc)?;
        }
        dev.synchronize()?;
        let step_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut st2 = ModelState::new(&dev, &model, args.max_seq, 1)?;
        let mut sc2 = Scratch::new(&dev, &text, args.max_seq)?;
        let _ = model.prefill(&dev, &ids, &mut st2, &mut sc2)?;
        let feed: Vec<u32> = vec![next; t];
        dev.synchronize()?;
        let t1 = std::time::Instant::now();
        let _ = model.prefill_seq(&dev, &feed, &mut st2, &mut sc2, 0)?;
        dev.synchronize()?;
        let pre_ms = t1.elapsed().as_secs_f64() * 1e3;

        println!(
            "  {t:>4}   {step_ms:>12.2}   {pre_ms:>16.2}   {:>5.2}   {:>7.2} ms/row",
            pre_ms / step_ms,
            pre_ms / t as f64
        );
    }
    println!();
    println!("ratio ~1.0 means rows are nearly free and batched verify can pay;");
    println!("ratio ~t means a t-row forward costs t steps and it cannot.");
    println!();
    Ok(())
}

/// Number of window rows scored per `lm_head` call. The GEMM tiles `T` in
/// blocks of `GB10_TILE_T` (64), so this reads `lm_head` exactly once per tile.
/// It does not have to match that tile -- any `t` is legal -- but matching it
/// means every launch is a single tile pass with no partially idle block.
#[allow(dead_code)]
const PPL_TILE: usize = 64;

/// `-log p(target)` for one row of logits, i.e. `logsumexp(row) - row[target]`.
///
/// Accumulated in f64: the sum runs over 248320 terms and a f32 accumulator
/// loses enough of them to move the fourth decimal of the perplexity.
fn row_nll(logits: &[f32], vocab: usize, r: usize, target: u32) -> f64 {
    let row = &logits[r * vocab..(r + 1) * vocab];
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
    (m + s.ln()) - row[target as usize] as f64
}

/// Sum of `row_nll` over `rows` rows, spread across cores.
///
/// This is 255 x 248320 `exp` calls per window, which single-threaded would
/// cost more than the GPU work it is measuring.
fn tile_nll(logits: &[f32], vocab: usize, rows: usize, targets: &[u32]) -> f64 {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let nthreads = cores.min(rows).max(1);
    if nthreads <= 1 {
        return (0..rows).map(|r| row_nll(logits, vocab, r, targets[r])).sum();
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..nthreads)
            .map(|tid| {
                s.spawn(move || {
                    let mut acc = 0.0f64;
                    let mut r = tid;
                    while r < rows {
                        acc += row_nll(logits, vocab, r, targets[r]);
                        r += nthreads;
                    }
                    acc
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

/// Reads `llama-tokenize --ids` output: a python list, optionally followed by a
/// `Total number of tokens: N` line.
fn load_token_ids(path: &Path) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let body = match (s.find('['), s.find(']')) {
        (Some(a), Some(b)) if b > a => &s[a + 1..b],
        _ => s.as_str(),
    };
    let mut ids = Vec::new();
    for tok in body.split(|c: char| c == ',' || c.is_whitespace()) {
        if tok.is_empty() {
            continue;
        }
        ids.push(
            tok.parse::<u32>()
                .with_context(|| format!("token id {tok:?} in {}", path.display()))?,
        );
    }
    Ok(ids)
}

/// Perplexity of a token stream over non-overlapping 512-token windows.
///
/// Replicates `tools/llama.cpp/tools/perplexity` with its default
/// `--ppl-stride 0` (`llama_perplexity` hardcodes `n_ctx = 512`):
///
/// * whole, non-overlapping windows of `n_ctx` tokens, state cleared per window
/// * `first = n_ctx/2 = 256`
/// * each window contributes `n_ctx - first - 1 = 255` predictions, from rows
///   `first .. first+254` (positions 256..510) against the token that follows
///   each row (`window[257 .. 511]`)
/// * `ppl = exp(mean(nll))` over all of them
///
/// `add_bos` is false for this checkpoint's vocab (verified: `llama-tokenize`
/// emits no 248044 for the same file), so windows are used verbatim.
///
/// Matching the protocol exactly is the point: the same token ids go to
/// llama.cpp and to the bf16 reference, so the weights are the only difference
/// between the three numbers.
fn perplexity(args: &Args) -> Result<()> {
    let tokens_path = args
        .tokens
        .as_ref()
        .context("perplexity needs --tokens FILE (llama-tokenize --ids output)")?;
    let ids = load_token_ids(tokens_path)?;

    // Cross-check the engine's own tokenizer against the llama.cpp stream. If
    // these disagree, a perplexity difference would be a tokenizer artefact and
    // not a weight difference, so it is reported rather than assumed.
    if let Some(tf) = &args.text {
        let raw = std::fs::read_to_string(tf)
            .with_context(|| format!("reading {}", tf.display()))?;
        let tok = QwenTokenizer::from_model_dir(&args.model)?;
        let mine = tok.encode(&raw, false)?;
        let n = mine.len().min(ids.len());
        let first_bad = (0..n).find(|&i| mine[i] != ids[i]);
        let bad = (0..n).filter(|&i| mine[i] != ids[i]).count();
        println!("token stream cross-check");
        println!("  engine tokenizer   {} tokens", mine.len());
        println!("  llama-tokenize     {} tokens", ids.len());
        println!("  first divergence   {first_bad:?}");
        println!("  mismatches         {bad} / {n}");
        println!();
    }

    let n_ctx = args.ctx;
    anyhow::ensure!(n_ctx >= 4, "--ctx must be at least 4");
    let first = n_ctx / 2;
    let per_chunk = n_ctx - 1 - first;
    let n_chunk_max = ids.len() / n_ctx;
    anyhow::ensure!(n_chunk_max > 0, "not enough tokens for one {n_ctx}-token window");
    let n_chunk = if args.chunks < 0 {
        n_chunk_max
    } else {
        (args.chunks as usize).min(n_chunk_max)
    };

    println!("perplexity: {} tokens", ids.len());
    println!(
        "  n_ctx={n_ctx} first={first} preds/window={per_chunk} windows={n_chunk} (max {n_chunk_max})"
    );

    let cfg = load_config(&args.model)?;
    let text = cfg.text_config.clone();
    let dev = Device::new(0)?;
    let t0 = std::time::Instant::now();
    let model = Model::load_from(&dev, cfg.clone(), &args.model)?;
    println!("  model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let vocab = model.vocab_size();
    let hidden = text.hidden_size;
    let mut state = ModelState::new(&dev, &model, n_ctx, 1)?;
    let mut sc = Scratch::new(&dev, &text, n_ctx)?;
    let mut tile_x = dev.stream().alloc_zeros::<f32>(hidden * PPL_TILE)?;
    let mut tile_logits = dev.stream().alloc_zeros::<f32>(vocab * PPL_TILE)?;

    let mut nll = 0.0f64;
    let mut count = 0usize;
    let mut trace: Vec<f64> = Vec::with_capacity(n_chunk);
    let mut targets = vec![0u32; PPL_TILE];
    let started = std::time::Instant::now();

    for i in 0..n_chunk {
        let start = i * n_ctx;
        let window = &ids[start..start + n_ctx];

        // llama.cpp clears its memory before every window, which for this
        // hybrid model means the conv history and the Gated DeltaNet
        // recurrence as well as the KV cache. Both are order-dependent, so a
        // window that inherited them would not be the window llama.cpp scored.
        state.reset(&dev)?;
        model.forward_normed(&dev, window, &mut state, &mut sc, 0)?;

        let mut window_nll = 0.0f64;
        let mut off = first;
        while off < first + per_chunk {
            let rows = (first + per_chunk - off).min(PPL_TILE);
            for r in 0..rows {
                targets[r] = window[off + r + 1];
            }
            dev.ops()
                .copy_rows(&dev, &state.normed, &mut tile_x, off, rows, hidden)?;
            model
                .lm_head
                .forward_prefill(&dev, &tile_x, &mut tile_logits, rows)?;
            let host = dev.stream().memcpy_dtov(&tile_logits)?;
            dev.check_err()?;
            window_nll += tile_nll(&host, vocab, rows, &targets[..rows]);
            off += rows;
        }

        nll += window_nll;
        count += per_chunk;
        trace.push(window_nll / per_chunk as f64);

        if i % 10 == 0 || i == n_chunk - 1 {
            let el = started.elapsed().as_secs_f64();
            let ppl = (nll / count as f64).exp();
            println!(
                "  [{i}] ppl={ppl:.4}  {:.2}s/window  eta {:.1} min",
                el / (i + 1) as f64,
                el / (i + 1) as f64 * (n_chunk - i - 1) as f64 / 60.0
            );
        }
    }

    let mean_nll = nll / count as f64;
    let ppl = mean_nll.exp();
    let var = if trace.len() > 1 {
        let m = trace.iter().sum::<f64>() / trace.len() as f64;
        trace.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (trace.len() - 1) as f64
    } else {
        0.0
    };
    println!();
    println!("Final estimate: PPL = {ppl:.4}");
    println!("  predictions   {count}");
    println!("  mean nll      {mean_nll:.6}");
    println!("  wall          {:.1}s", started.elapsed().as_secs_f64());

    let result = serde_json::json!({
        "kind": "gb10-engine-ppl",
        "model": args.model.display().to_string(),
        "quant": "nvfp4-fp8-mixed (modelopt)",
        "tokens_file": tokens_path.display().to_string(),
        "n_ctx": n_ctx,
        "first": first,
        "preds_per_window": per_chunk,
        "windows": n_chunk,
        "predictions": count,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "ppl_stderr_window": (var / trace.len() as f64).sqrt(),
        "wall_s": started.elapsed().as_secs_f64(),
    });
    if let Some(out) = &args.out {
        std::fs::write(out, serde_json::to_string_pretty(&result)?)
            .with_context(|| format!("writing {}", out.display()))?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

/// One MMLU-style multiple-choice question.
#[derive(serde::Deserialize, Debug, Clone)]
struct Question {
    id: String,
    subject: String,
    question: String,
    options: Vec<String>,
    answer: usize,
}

/// Render a question in the canonical MMLU format.
///
/// This string is a contract between the engine and the Python reference
/// scorer: if they disagree on a single character the comparison measures the
/// prompt rather than the weights, so both build it from the same rule and the
/// engine records its token count per question for the reference to check.
fn render_choice_prompt(q: &Question) -> String {
    let subject = q.subject.replace('_', " ");
    let mut s = format!(
        "The following are multiple choice questions (with answers) about {subject}.\n\n{}\n",
        q.question.trim()
    );
    for (i, opt) in q.options.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", (b'A' + i as u8) as char, opt.trim()));
    }
    s.push_str("Answer:");
    s
}

/// Multiple-choice accuracy by answer-letter log-probability.
///
/// Standard MMLU scoring: one forward pass per question, then argmax over the
/// logits of the four answer-letter tokens at the position after `Answer:`.
/// " A".." D" are each a single token in this vocabulary, which is asserted
/// rather than assumed.
fn choice(args: &Args) -> Result<()> {
    let jsonl = args
        .jsonl
        .as_ref()
        .context("choice needs --jsonl FILE")?;
    let raw = std::fs::read_to_string(jsonl)
        .with_context(|| format!("reading {}", jsonl.display()))?;
    let parsed: Vec<Question> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("parsing {}", jsonl.display()))?;
    anyhow::ensure!(!parsed.is_empty(), "no questions in {}", jsonl.display());
    let n = if args.limit < 0 {
        parsed.len()
    } else {
        (args.limit as usize).min(parsed.len())
    };
    let questions = &parsed[..n];

    let cfg = load_config(&args.model)?;
    let text = cfg.text_config.clone();
    let dev = Device::new(0)?;
    let tok = QwenTokenizer::from_model_dir(&args.model)?;

    let mut letters = [0u32; 4];
    for (i, l) in [" A", " B", " C", " D"].iter().enumerate() {
        let ids = tok.encode(l, false)?;
        anyhow::ensure!(
            ids.len() == 1,
            "answer letter {l:?} tokenises to {} tokens; letter scoring assumes 1",
            ids.len()
        );
        letters[i] = ids[0];
    }
    println!("answer letters \" A\"..\" D\" -> token ids {letters:?}");

    println!("MMLU: {} questions", questions.len());
    let t0 = std::time::Instant::now();
    let model = Model::load_from(&dev, cfg.clone(), &args.model)?;
    println!("  model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let vocab = model.vocab_size();
    let hidden = text.hidden_size;
    let max_seq = args.max_seq;
    let mut state = ModelState::new(&dev, &model, max_seq, 1)?;
    let mut sc = Scratch::new(&dev, &text, max_seq)?;
    let mut x = dev.stream().alloc_zeros::<f32>(hidden)?;
    let mut lg = dev.stream().alloc_zeros::<f32>(vocab)?;

    let mut correct = 0usize;
    let mut per_subject: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut rows = Vec::with_capacity(questions.len());
    let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
    let started = std::time::Instant::now();

    for (qi, q) in questions.iter().enumerate() {
        anyhow::ensure!(q.options.len() == 4, "{}: {} options", q.id, q.options.len());
        anyhow::ensure!(q.answer < 4, "{}: answer out of range", q.id);
        let prompt = render_choice_prompt(q);
        let ids = tok.encode(&prompt, false)?;
        anyhow::ensure!(
            ids.len() < max_seq,
            "{}: prompt {} tokens exceeds max_seq {}",
            q.id,
            ids.len(),
            max_seq
        );

        state.reset(&dev)?;
        model.forward_normed(&dev, &ids, &mut state, &mut sc, 0)?;
        dev.ops()
            .copy_rows(&dev, &state.normed, &mut x, ids.len() - 1, 1, hidden)?;
        model.lm_head.forward_prefill(&dev, &x, &mut lg, 1)?;
        let host = dev.stream().memcpy_dtov(&lg)?;
        dev.check_err()?;

        let mut scores = [f32::NEG_INFINITY; 4];
        let mut pick = 0usize;
        for (i, &t) in letters.iter().enumerate() {
            scores[i] = host[t as usize];
            if scores[i] > scores[pick] {
                pick = i;
            }
        }
        let ok = pick == q.answer;
        if ok {
            correct += 1;
        }
        let e = counts.entry(q.subject.clone()).or_insert((0, 0));
        e.1 += 1;
        if ok {
            e.0 += 1;
        }
        rows.push(serde_json::json!({
            "id": q.id, "subject": q.subject, "gold": q.answer, "pred": pick,
            "scores": scores, "prompt_tokens": ids.len(),
        }));

        if qi % 100 == 0 || qi + 1 == questions.len() {
            let el = started.elapsed().as_secs_f64();
            println!(
                "  [{qi}/{}] acc={:.4}  {:.2}s/q  eta {:.1} min",
                questions.len(),
                correct as f64 / (qi + 1) as f64,
                el / (qi + 1) as f64,
                el / (qi + 1) as f64 * (questions.len() - qi - 1) as f64 / 60.0
            );
        }
    }

    let acc = correct as f64 / questions.len() as f64;
    // Wilson score interval, so a subset result carries its own uncertainty.
    let nf = questions.len() as f64;
    let z = 1.96f64;
    let denom = 1.0 + z * z / nf;
    let centre = (acc + z * z / (2.0 * nf)) / denom;
    let half = z * ((acc * (1.0 - acc) / nf + z * z / (4.0 * nf * nf)).sqrt()) / denom;

    println!();
    println!("MMLU accuracy: {correct}/{} = {:.4}", questions.len(), acc);
    println!("  95% CI: [{:.4}, {:.4}]", centre - half, centre + half);
    println!();
    let mut subs: Vec<_> = counts.iter().collect();
    subs.sort_by(|a, b| a.0.cmp(b.0));
    for (s, (c, t)) in &subs {
        println!("  {s:<40} {c:>4}/{t:<4} {:.3}", *c as f64 / *t as f64);
        per_subject.insert(
            (*s).clone(),
            serde_json::json!({"correct": c, "total": t, "accuracy": *c as f64 / *t as f64}),
        );
    }
    println!("  wall {:.1}s", started.elapsed().as_secs_f64());

    if let Some(out) = &args.out {
        let result = serde_json::json!({
            "kind": "mmlu-choice",
            "model": args.model.display().to_string(),
            "quant": "nvfp4-fp8-mixed (modelopt)",
            "jsonl": jsonl.display().to_string(),
            "questions": questions.len(),
            "correct": correct,
            "accuracy": acc,
            "ci95": [centre - half, centre + half],
            "letter_token_ids": letters.to_vec(),
            "prompt_tokens": rows.iter().filter_map(|r| r["prompt_tokens"].as_u64()).sum::<u64>(),
            "per_subject": serde_json::Value::Object(per_subject),
            "rows": rows,
            "wall_s": started.elapsed().as_secs_f64(),
        });
        std::fs::write(out, serde_json::to_string_pretty(&result)?)
            .with_context(|| format!("writing {}", out.display()))?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let (cmd, args) = parse_args()?;
    let layers: Vec<usize> = match cmd.as_str() {
        // Layer 0 is a Gated DeltaNet block, layer 3 a full-attention block.
        // Both are part of the gate.
        "generate" => {
            let prompt = if args.prompt.is_empty() {
                "What is the capital of France?".to_string()
            } else {
                args.prompt.clone()
            };
            let ok = generate(&args, &prompt, args.n_new)?;
            if !ok {
                bail!("generate gate FAILED");
            }
            println!("\ngenerate: OK");
            return Ok(());
        }
        "forward-cost" => {
            forward_cost(&args)?;
            return Ok(());
        }
        "chunked-prefill" => {
            let ok = chunked_prefill(&args, args.n_new.max(1))?;
            if !ok {
                bail!("chunked-prefill gate FAILED");
            }
            println!("\nchunked-prefill: OK");
            return Ok(());
        }
        "mtp-generate" => {
            let ok = mtp_generate(&args, args.n_new)?;
            if !ok {
                bail!("mtp-generate gate FAILED");
            }
            println!("\nmtp-generate: OK");
            return Ok(());
        }
        "mtp-probe" => {
            mtp_probe(&args)?;
            return Ok(());
        }
        "perplexity" => {
            perplexity(&args)?;
            return Ok(());
        }
        "choice" => {
            choice(&args)?;
            return Ok(());
        }
        "batch-parity" => {
            let ok = batch_parity(&args, args.n_seq, args.n_new)?;
            if !ok {
                bail!("batch-parity gate FAILED");
            }
            println!("\nbatch-parity: OK");
            return Ok(());
        }
        "layer-parity" | "all" => {
            if args.layer == usize::MAX {
                vec![0, 3]
            } else {
                vec![args.layer]
            }
        }
        other => bail!(
            "unknown subcommand {other:?} (expected layer-parity|all|generate|batch-parity|mtp-probe|mtp-generate|chunked-prefill|forward-cost|perplexity|choice)"
        ),
    };

    let mut all_ok = true;
    for l in layers {
        let a = Args {
            layer: l,
            model: args.model.clone(),
            fixtures: args.fixtures.clone(),
            max_seq: args.max_seq,
            tol_norm: args.tol_norm,
            tol_rel: args.tol_rel,
            quiet: args.quiet,
            raw: args.raw,
            prompt: args.prompt.clone(),
            n_new: args.n_new,
            n_seq: args.n_seq,
            min_agree: args.min_agree,
            repeat: args.repeat,
            profile: args.profile,
            tokens: args.tokens.clone(),
            text: args.text.clone(),
            ctx: args.ctx,
            chunks: args.chunks,
            out: args.out.clone(),
            jsonl: args.jsonl.clone(),
            limit: args.limit,
        };
        all_ok &= layer_parity(&a)?;
    }
    if !all_ok {
        bail!("correctness gate FAILED");
    }
    println!("\nall gates: OK");
    Ok(())
}
