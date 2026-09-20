//! GB10-Engine benchmark and verification harness.
//!
//! Subcommands:
//!   hw           print device facts
//!   gemv-parity  GPU GEMV vs an independent CPU reference, on real weights
//!   stream       push the whole text-model weight set through the GEMV
//!                kernels and report achieved bandwidth and projected tok/s
//!
//! `stream` is the M1 gate: it measures whether the decode path actually
//! reaches the ~200 GB/s memory roofline documented in `docs/PHYSICS.md`.

mod reference;

use anyhow::{Context, Result};
use gb10_core::{LayerType, ModelConfig, ShardedSafeTensors};
use gb10_cuda::{CudaSlice, Device};
use std::time::Instant;

const DEFAULT_MODEL: &str = "models/Qwen3.8-27B-NVFP4";
const ROOFLINE_GBPS: f64 = 228.0;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    let opt = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model = opt("--model").unwrap_or_else(|| DEFAULT_MODEL.to_string());

    match cmd {
        "hw" => hw(),
        "gemv-parity" => gemv_parity(&model),
        "stream" => stream(&model, opt("--out")),
        "launch-overhead" => launch_overhead(),
        _ => {
            eprintln!(
                "usage: gb10-bench <hw|gemv-parity|stream|launch-overhead> \
                 [--model DIR] [--out FILE]"
            );
            std::process::exit(2);
        }
    }
}

fn hw() -> Result<()> {
    let dev = Device::new(0)?;
    println!("device            : {}", dev.name()?);
    println!("compute capability: {:?}", dev.compute_capability()?);
    println!(
        "total memory      : {:.1} GiB",
        dev.total_memory()? as f64 / (1u64 << 30) as f64
    );
    println!("ptx modules       : {}", dev.module_count());
    println!("cuda arch         : {}", gb10_cuda::CUDA_ARCH);
    Ok(())
}

// ---------------------------------------------------------------------------
// gemv-parity
// ---------------------------------------------------------------------------

/// Cost of an empty kernel launch, and of the whole host-side launch path.
///
/// Decode issues ~1200 kernel launches per token (497 GEMVs plus ~700
/// elementwise/norm/RoPE/recurrence kernels across 64 layers). If the
/// host-side launch path costs tens of microseconds, that alone accounts for a
/// large fraction of the gap between the measured decode rate and the
/// bandwidth roofline, and the fix is to batch or fuse rather than to tune the
/// GEMV inner loop.
fn launch_overhead() -> Result<()> {
    let dev = Device::new(0)?;
    let n = 1 << 20;
    let mut c = dev.stream().alloc_zeros::<f32>(n)?;
    let a = dev.stream().alloc_zeros::<f32>(n)?;
    let b = dev.stream().alloc_zeros::<f32>(n)?;

    let reps = 2000;
    // Warm up so first-touch allocation and module load are not counted.
    for _ in 0..50 {
        dev.ops().add(&dev, &a, &b, &mut c, n)?;
    }
    dev.synchronize()?;

    let t = std::time::Instant::now();
    for _ in 0..reps {
        dev.ops().add(&dev, &a, &b, &mut c, n)?;
    }
    let queued = t.elapsed();
    dev.synchronize()?;
    let total = t.elapsed();

    let per = total.as_secs_f64() / reps as f64;
    println!("add_kernel on {n} elements, {reps} reps");
    println!(
        "  host-side queue time : {:.1} us/launch",
        queued.as_secs_f64() / reps as f64 * 1e6
    );
    println!("  end-to-end            : {:.1} us/launch", per * 1e6);
    println!(
        "\nprojected at 1200 launches/token: {:.1} ms/token  ({:.1} tok/s ceiling)",
        per * 1200.0 * 1e3,
        1.0 / (per * 1200.0)
    );
    Ok(())
}

fn gemv_parity(model: &str) -> Result<()> {
    let dev = Device::new(0)?;
    let st = ShardedSafeTensors::open(model).context("open checkpoint")?;
    let k = 5120usize;

    println!("== GEMV parity vs independent CPU reference (real weights) ==\n");

    // Deterministic pseudo-random activation vector in [-0.75, 0.75].
    let x: Vec<f32> = (0..k)
        .map(|i| ((i as f32 * 0.61803398875).fract() * 2.0 - 1.0) * 0.75)
        .collect();
    let x_dev = dev.stream().clone_htod(&x)?;

    let cases = [
        ("NVFP4", "model.language_model.layers.0.mlp.gate_proj", 256usize),
        ("FP8", "model.language_model.layers.0.linear_attn.in_proj_qkv", 256),
        ("bf16", "model.language_model.layers.0.linear_attn.in_proj_b", 48),
    ];

    let mut all_ok = true;
    for (label, name, rows) in cases {
        let wname = format!("{name}.weight");
        let info = st.info(&wname).with_context(|| wname.clone())?.clone();
        let full_n = info.shape[0];
        let n = full_n.min(rows);
        let row_bytes = info.nbytes / full_n;
        let raw = st.tensor_bytes(&wname)?;
        let slice = &raw[..row_bytes * n];

        let (got, want) = match label {
            "NVFP4" => {
                let sname = format!("{name}.weight_scale");
                let gname = format!("{name}.weight_scale_2");
                let sinfo = st.info(&sname).context(sname.clone())?.clone();
                let sraw = st.tensor_bytes(&sname)?;
                let srow = sinfo.nbytes / full_n;
                let sslice = &sraw[..srow * n];
                let scale2 =
                    f32::from_le_bytes(st.tensor_bytes(&gname)?[..4].try_into().unwrap());

                let w = dev.stream().clone_htod(slice)?;
                let s = dev.stream().clone_htod(sslice)?;
                let s2 = dev.stream().clone_htod(&[scale2])?;
                let mut y: CudaSlice<f32> = dev.stream().alloc_zeros(n)?;
                dev.kernels()
                    .nvfp4_gemv(&dev, &x_dev, &w, &s, &s2, &mut y, n, k, 1)?;
                dev.synchronize()?;
                (
                    dev.stream().clone_dtoh(&y)?,
                    reference::nvfp4_gemv_ref(&x, slice, sslice, scale2, n, k),
                )
            }
            "FP8" => {
                let sname = format!("{name}.weight_scale");
                let scale =
                    f32::from_le_bytes(st.tensor_bytes(&sname)?[..4].try_into().unwrap());
                let w = dev.stream().clone_htod(slice)?;
                let s = dev.stream().clone_htod(&[scale])?;
                let mut y: CudaSlice<f32> = dev.stream().alloc_zeros(n)?;
                dev.kernels()
                    .fp8_gemv(&dev, &x_dev, &w, &s, &mut y, n, k, 1)?;
                dev.synchronize()?;
                (
                    dev.stream().clone_dtoh(&y)?,
                    reference::fp8_gemv_ref(&x, slice, scale, n, k),
                )
            }
            _ => {
                let words: Vec<u16> = slice
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let w = dev.stream().clone_htod(&words)?;
                let mut y: CudaSlice<f32> = dev.stream().alloc_zeros(n)?;
                dev.kernels()
                    .bf16_gemv(&dev, &x_dev, &w, &mut y, n, k, 1)?;
                dev.synchronize()?;
                (
                    dev.stream().clone_dtoh(&y)?,
                    reference::bf16_gemv_ref(&x, &words, n, k),
                )
            }
        };

        // Error metrics. A per-element relative error is meaningless here:
        // dot products cancel, so individual outputs land arbitrarily close to
        // zero while carrying ~1e-6 of fp32 rounding. The honest measures are
        // (a) absolute error relative to the vector's own scale, and
        // (b) per-element relative error restricted to outputs that are not
        //     swamped by cancellation.
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
        let mut max_abs = 0.0f64;
        let mut max_rel_large = 0.0f64;
        for i in 0..n {
            let a = got[i] as f64;
            let b = want[i] as f64;
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            if b.abs() > 0.1 * scale {
                max_rel_large = max_rel_large.max(d / b.abs());
            }
        }
        let norm_err = if scale > 0.0 { max_abs / scale } else { max_abs };
        // The GPU reduces as a tree while the reference accumulates
        // sequentially, so fp32 reassociation noise is expected; anything
        // above these tolerances is a real bug.
        let (tol_norm, tol_rel) = (1e-5, 1e-4);
        let ok = norm_err < tol_norm && max_rel_large < tol_rel;
        all_ok &= ok;
        println!(
            "{label:6} {name}\n       rows={n} k={k}  max|y|={scale:.4e}  err/scale={norm_err:.3e}  max_rel(|y|>0.1max)={max_rel_large:.3e}  {}",
            if ok { "OK" } else { "FAIL" }
        );
    }

    println!();
    anyhow::ensure!(all_ok, "gemv parity FAILED");
    println!("gemv parity: OK");
    Ok(())
}

// ---------------------------------------------------------------------------
// stream
// ---------------------------------------------------------------------------

enum Kind {
    NvFp4 {
        w: CudaSlice<u8>,
        s: CudaSlice<u8>,
        s2: CudaSlice<f32>,
        n: usize,
        k: usize,
    },
    Fp8 {
        w: CudaSlice<u8>,
        s: CudaSlice<f32>,
        n: usize,
        k: usize,
    },
    Bf16 {
        w: CudaSlice<u16>,
        n: usize,
        k: usize,
    },
}

struct DevWeight {
    name: String,
    bytes: usize,
    kind: Kind,
}

impl DevWeight {
    fn describe(&self) -> String {
        let (n, k) = self.kind.shape();
        format!("{} [{n}x{k}] {} bytes", self.name, self.bytes)
    }
}

impl Kind {
    fn shape(&self) -> (usize, usize) {
        match self {
            Kind::NvFp4 { n, k, .. } | Kind::Fp8 { n, k, .. } | Kind::Bf16 { n, k, .. } => {
                (*n, *k)
            }
        }
    }
}

/// Every matrix the text decoder reads once per token.
fn text_weight_names(cfg: &ModelConfig, st: &ShardedSafeTensors) -> Vec<(String, &'static str)> {
    let t = &cfg.text_config;
    let mut out = Vec::new();
    for i in 0..t.num_hidden_layers {
        let p = format!("model.language_model.layers.{i}");
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            out.push((format!("{p}.mlp.{proj}.weight"), "nvfp4"));
        }
        match t.layer_types[i] {
            LayerType::FullAttention => {
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    out.push((format!("{p}.self_attn.{proj}.weight"), "fp8"));
                }
            }
            LayerType::LinearAttention => {
                for proj in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                    out.push((format!("{p}.linear_attn.{proj}.weight"), "fp8"));
                }
            }
        }
    }
    out.push(("lm_head.weight".to_string(), "nvfp4"));
    out.retain(|(n, _)| st.contains(n));
    out
}

fn stream(model: &str, out: Option<String>) -> Result<()> {
    let dev = Device::new(0)?;
    let st = ShardedSafeTensors::open(model).context("open checkpoint")?;
    let cfg = ModelConfig::from_file(format!("{model}/config.json"))?;
    let names = text_weight_names(&cfg, &st);

    println!("== decode-path weight streaming ==\n");
    println!("device: {}", dev.name()?);

    let mut weights = Vec::new();
    let mut total_bytes = 0usize;
    let t0 = Instant::now();
    for (name, kind) in &names {
        let info = st.info(name).unwrap().clone();
        let n = info.shape[0];
        let k = info.shape[1];
        let raw = st.tensor_bytes(name)?;

        let (dw, bytes) = match *kind {
            "nvfp4" => {
                let base = name.trim_end_matches(".weight");
                let sname = format!("{base}.weight_scale");
                let gname = format!("{base}.weight_scale_2");
                let sraw = st.tensor_bytes(&sname).with_context(|| sname.clone())?;
                let scale2 =
                    f32::from_le_bytes(st.tensor_bytes(&gname)?[..4].try_into().unwrap());
                let w = dev.stream().clone_htod(raw)?;
                let s = dev.stream().clone_htod(sraw)?;
                let s2 = dev.stream().clone_htod(&[scale2])?;
                (
                    Kind::NvFp4 { w, s, s2, n, k },
                    raw.len() + sraw.len() + 4,
                )
            }
            "fp8" => {
                let base = name.trim_end_matches(".weight");
                let sname = format!("{base}.weight_scale");
                let scale =
                    f32::from_le_bytes(st.tensor_bytes(&sname)?[..4].try_into().unwrap());
                let w = dev.stream().clone_htod(raw)?;
                let s = dev.stream().clone_htod(&[scale])?;
                (Kind::Fp8 { w, s, n, k }, raw.len() + 4)
            }
            _ => {
                let words: Vec<u16> = raw
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let w = dev.stream().clone_htod(&words)?;
                (Kind::Bf16 { w, n, k }, raw.len())
            }
        };
        total_bytes += bytes;
        weights.push(DevWeight {
            name: name.clone(),
            bytes,
            kind: dw,
        });
    }
    dev.synchronize()?;
    println!(
        "uploaded {} matrices, {:.2} GB in {:.1}s",
        weights.len(),
        total_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    let (max_n, max_k) = weights
        .iter()
        .map(|w| w.kind.shape())
        .fold((0usize, 0usize), |a, b| (a.0.max(b.0), a.1.max(b.1)));

    let x: Vec<f32> = (0..max_k).map(|i| ((i % 17) as f32) * 0.031 - 0.25).collect();
    let x_dev = dev.stream().clone_htod(&x)?;
    let mut y: CudaSlice<f32> = dev.stream().alloc_zeros(max_n)?;

    let run_once = |weights: &[DevWeight], y: &mut CudaSlice<f32>| -> Result<()> {
        for w in weights {
            match &w.kind {
                Kind::NvFp4 { w, s, s2, n, k } => {
                    dev.kernels().nvfp4_gemv(&dev, &x_dev, w, s, s2, y, *n, *k, 1)?
                }
                Kind::Fp8 { w, s, n, k } => {
                    dev.kernels().fp8_gemv(&dev, &x_dev, w, s, y, *n, *k, 1)?
                }
                Kind::Bf16 { w, n, k } => {
                    dev.kernels().bf16_gemv(&dev, &x_dev, w, y, *n, *k, 1)?
                }
            }
        }
        Ok(())
    };

    for _ in 0..2 {
        run_once(&weights, &mut y)?;
        dev.synchronize()?;
    }

    for w in &weights {
        anyhow::ensure!(w.bytes > 0, "{}", w.describe());
    }
    tracing::debug!("largest matrix: {}", weights.iter().map(|w| w.describe()).max().unwrap());

    // Report every iteration separately: if the achieved bandwidth decays over
    // a sustained run, the short-burst roofline is optimistic and the real
    // ceiling is lower.
    let iters = 30;
    let mut per_iter = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        run_once(&weights, &mut y)?;
        dev.synchronize()?;
        per_iter.push(t.elapsed().as_secs_f64());
    }
    for (i, s) in per_iter.iter().enumerate() {
        if i % 5 == 0 || i == iters - 1 {
            println!(
                "  iter {i:2}: {:.2} ms  {:.1} GB/s",
                s * 1e3,
                total_bytes as f64 / s / 1e9
            );
        }
    }
    let elapsed: f64 = per_iter.iter().sum::<f64>() / iters as f64;

    let gbps = total_bytes as f64 / elapsed / 1e9;
    println!();
    println!("per-token weight traffic : {:.3} GB", total_bytes as f64 / 1e9);
    println!("time per token           : {:.2} ms", elapsed * 1e3);
    println!("achieved bandwidth       : {:.1} GB/s", gbps);
    println!("projected decode         : {:.2} tok/s (single stream)", 1.0 / elapsed);
    println!(
        "roofline at {ROOFLINE_GBPS:.0} GB/s    : {:.2} tok/s",
        ROOFLINE_GBPS * 1e9 / total_bytes as f64
    );
    println!(
        "bandwidth utilisation    : {:.1}% of measured {ROOFLINE_GBPS:.0} GB/s",
        gbps / ROOFLINE_GBPS * 100.0
    );

    if let Some(out) = out {
        let j = serde_json::json!({
            "kind": "stream",
            "device": dev.name()?,
            "matrices": weights.len(),
            "per_token_bytes": total_bytes,
            "seconds_per_token": elapsed,
            "bandwidth_gbps": gbps,
            "projected_tok_s": 1.0 / elapsed,
            "roofline_gbps": ROOFLINE_GBPS,
            "bandwidth_utilisation_pct": gbps / ROOFLINE_GBPS * 100.0,
        });
        if let Some(dir) = std::path::Path::new(&out).parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&out, serde_json::to_string_pretty(&j)?)?;
        println!("wrote {out}");
    }
    Ok(())
}
