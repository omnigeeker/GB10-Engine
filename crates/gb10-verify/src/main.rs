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
use gb10_core::config::ModelConfig;
use gb10_cuda::Device;
use gb10_model::{LayerState, Scratch, Store};
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
    let mut state = LayerState::new(&dev, &text, &layer, args.max_seq)?;
    let mut sc = Scratch::new(&dev, &text)?;

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
        layer.forward(&dev, &text, &xbuf, &mut obuf, &mut state, &mut sc)?;
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

#[derive(Clone)]
struct Args {
    model: PathBuf,
    fixtures: PathBuf,
    layer: usize,
    max_seq: usize,
    tol_norm: f32,
    tol_rel: f32,
    quiet: bool,
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
            other => bail!("unknown flag {other}"),
        }
    }
    Ok((cmd, a))
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
        "layer-parity" | "all" => {
            if args.layer == usize::MAX {
                vec![0, 3]
            } else {
                vec![args.layer]
            }
        }
        other => bail!("unknown subcommand {other:?} (expected layer-parity|all)"),
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
        };
        all_ok &= layer_parity(&a)?;
    }
    if !all_ok {
        bail!("correctness gate FAILED");
    }
    println!("\nall gates: OK");
    Ok(())
}
