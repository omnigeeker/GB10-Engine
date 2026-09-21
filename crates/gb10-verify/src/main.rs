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
        profile: false,
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
fn mtp_generate(args: &Args, n_new: usize) -> Result<bool> {
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

    // --- speculative: draft with the head, verify with a decoder step ---
    // Invariant at the top of each round: `st.normed` is h_t and `cur` is
    // x_{t+1}. The head guesses x_{t+2} from exactly those two; the decoder
    // step on x_{t+1} then produces h_{t+1} and the true x_{t+2}.
    let mut st = ModelState::new(&dev, &model, args.max_seq, 1)?;
    let mut sc = Scratch::new(&dev, &text, args.max_seq)?;
    let mut ms = MtpState::new(&dev, &text, args.max_seq, model.vocab_size())?;
    let mut idx: gb10_cuda::CudaSlice<i32> = dev.stream().alloc_zeros::<i32>(1)?;
    let mut cur = model.prefill(&dev, &ids, &mut st, &mut sc)?;
    let t1 = std::time::Instant::now();
    let mut got: Vec<u32> = Vec::new();
    let (mut hits, mut steps) = (0usize, 0usize);
    // On acceptance `cur` becomes a token that was *just* emitted, so the naive
    // "emit cur, then emit actual" loop pushes it a second time on the next
    // round. Tracking whether `cur` is already in `got` is what keeps the
    // emitted sequence equal to greedy. It also makes the accounting legible:
    // each round runs exactly one decoder step and emits `1 + accepted` tokens,
    // so the long-run rate is `1 + acceptance` tokens per step.
    let mut emitted = false;
    while got.len() < n_new && steps < n_new * 4 {
        if !emitted {
            if tok.is_eos(cur) {
                break;
            }
            got.push(cur);
        }
        mtp.forward(&dev, &text, &st.normed, cur, &model.embed, &model.lm_head, &mut ms)?;
        dev.ops().argmax(&dev, &ms.logits, &mut idx, model.vocab_size())?;
        let draft = dev.stream().memcpy_dtov(&idx)?[0] as u32;

        let actual = model.step(&dev, cur, &mut st, &mut sc)?;
        steps += 1;
        if draft == actual {
            hits += 1;
            if got.len() < n_new {
                got.push(actual);
                emitted = true;
            } else {
                emitted = true;
            }
        } else {
            emitted = false;
        }
        cur = actual;
    }
    let dt_spec = t1.elapsed().as_secs_f64();

    let rps = want.len() as f64 / dt_ref;
    let sps = got.len() as f64 / dt_spec;
    println!("== MTP speculative decoding ==");
    println!("  prompt: {prompt:?}");
    println!("  greedy reference : {:>3} tokens in {:.3}s -> {:.2} tok/s", want.len(), dt_ref, rps);
    println!("  MTP speculative  : {:>3} tokens in {:.3}s -> {:.2} tok/s", got.len(), dt_spec, sps);
    println!("  draft acceptance : {hits}/{steps} = {:.1}%", 100.0 * hits as f64 / steps as f64);
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
            "unknown subcommand {other:?} (expected layer-parity|all|generate|batch-parity|mtp-probe|mtp-generate)"
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
            profile: args.profile,
        };
        all_ok &= layer_parity(&a)?;
    }
    if !all_ok {
        bail!("correctness gate FAILED");
    }
    println!("\nall gates: OK");
    Ok(())
}
