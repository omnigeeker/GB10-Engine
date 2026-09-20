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
        "store-stream" => store_stream(&model),
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

    // ---- batched path: does row b of the batch equal the single-row result? --
    // The batched kernels index x by sequence, so a wrong per-sequence stride
    // produces a wrong answer that the single-row cases above cannot see.
    {
        let name = "model.language_model.layers.0.mlp.gate_proj";
        let wname = format!("{name}.weight");
        let info = st.info(&wname).with_context(|| wname.clone())?.clone();
        let full_n = info.shape[0];
        let n = full_n.min(256);
        let row_bytes = info.nbytes / full_n;
        let raw = st.tensor_bytes(&wname)?;
        let slice = &raw[..row_bytes * n];

        let sname = format!("{name}.weight_scale");
        let sinfo = st.info(&sname).with_context(|| sname.clone())?.clone();
        let sraw = st.tensor_bytes(&sname)?;
        let srow = sinfo.nbytes / full_n;
        let sslice = &sraw[..srow * n];
        let scale2 = f32::from_le_bytes(st.tensor_bytes(&format!("{name}.weight_scale_2"))?[..4].try_into().unwrap());

        let w = dev.stream().clone_htod(slice)?;
        let sc = dev.stream().clone_htod(sslice)?;
        let s2 = dev.stream().clone_htod(&[scale2])?;

        let b = 16usize;
        let want = reference::nvfp4_gemv_ref(&x, slice, sslice, scale2, n, k);

        for (label, batch) in [("batch=1 ", 1usize), ("batch=16", b)] {
            // Row 0 is the same vector in every slot; slots 1.. differ so a
            // wrong stride cannot read equivalent data and pass.
            let mut xb = Vec::with_capacity(batch * k);
            for j in 0..batch {
                for i in 0..k {
                    let v = ((i as f32 * 0.61803398875).fract() * 2.0 - 1.0) * 0.75;
                    xb.push(if j == 0 { v } else { v * (1.0 + j as f32 * 0.01) });
                }
            }
            let xb_dev = dev.stream().clone_htod(&xb)?;
            let mut y: CudaSlice<f32> = dev.stream().alloc_zeros(n * batch)?;
            dev.kernels().nvfp4_gemv(&dev, &xb_dev, &w, &sc, &s2, &mut y, n, k, batch)?;
            dev.synchronize()?;
            let got = dev.stream().clone_dtoh(&y)?;

            let mut worst = 0.0f64;
            let mut worst_row = 0usize;
            for j in 0..batch {
                for i in 0..n {
                    let want_v = if j == 0 { want[i] } else { want[i] * (1.0 + j as f32 * 0.01) };
                    let d = (got[j * n + i] as f64 - want_v as f64).abs();
                    if d > worst {
                        worst = d;
                        worst_row = j;
                    }
                }
            }
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
            let norm = if scale > 0.0 { worst / scale } else { worst };
            let ok = norm < 1e-4;
            all_ok &= ok;
            println!(
                "batched {label}  n={n} k={k} b={batch}  max|y|={scale:.4e}  err/scale={norm:.3e} (worst seq {worst_row})  {}",
                if ok { "OK" } else { "FAIL" }
            );
        }
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

/// The same measurement as `stream`, but every matrix is loaded through
/// `gb10_model::Store` -- the exact path the engine uses -- instead of the
/// benchmark's own uploader.
///
/// This exists to settle one question: the NVFP4 GEMV runs exactly 2x slower
/// inside the model than inside `stream` while the FP8 GEMV is identical in
/// both. If the cause is the weight buffers or how they are loaded, this
/// command will show the slow number with nothing else running; if it shows
/// the fast number, the buffers are innocent and the cause is contextual.
fn store_stream(model: &str) -> Result<()> {
    let dev = Device::new(0)?;
    let dir = std::path::Path::new(model);
    let store = gb10_model::Store::open(dir, "model.language_model.")?;

    // The tiny bf16 in_proj_a/b run at ~11 GB/s (gridX=2). Excluding them
    // tests whether 96 starved kernels cost more than their 47 MB of traffic
    // suggests, by draining the memory pipeline around them.
    let skip_small = std::env::args().any(|a| a == "--skip-small");
    let mut names: Vec<String> = Vec::new();
    for i in 0..64 {
        let p = format!("layers.{i}");
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            names.push(format!("{p}.mlp.{proj}"));
        }
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            names.push(format!("{p}.self_attn.{proj}"));
        }
        let mut lin = vec!["in_proj_qkv", "in_proj_z", "out_proj"];
        if !skip_small {
            lin.push("in_proj_a");
            lin.push("in_proj_b");
        }
        for proj in lin {
            names.push(format!("{p}.linear_attn.{proj}"));
        }
    }
    names.push("lm_head".to_string());

    let t0 = Instant::now();
    let mut weights = Vec::new();
    let mut total_bytes = 0usize;
    for name in &names {
        let wname = if name == "lm_head" {
            name.clone()
        } else {
            name.clone()
        };
        if !store.has(&format!("{wname}.weight")) {
            continue;
        }
        // lm_head lives outside the `model.language_model.` prefix.
        let lin = if name == "lm_head" {
            gb10_model::Store::open(dir, "")?.linear(&dev, "lm_head")?
        } else {
            store.linear(&dev, name)?
        };
        total_bytes += lin.traffic_bytes();
        weights.push(lin);
    }
    dev.synchronize()?;
    println!(
        "loaded {} matrices via Store, {:.3} GB in {:.1}s",
        weights.len(),
        total_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    let mut max_k = 0usize;
    let mut max_n = 0usize;
    for w in &weights {
        max_k = max_k.max(w.k);
        max_n = max_n.max(w.n);
    }
    let x: Vec<f32> = (0..max_k).map(|i| ((i % 17) as f32) * 0.031 - 0.25).collect();
    let x_dev = dev.stream().clone_htod(&x)?;
    let mut y: CudaSlice<f32> = dev.stream().alloc_zeros(max_n)?;

    for _ in 0..2 {
        for w in &weights {
            w.forward(&dev, &x_dev, &mut y, 1)?;
        }
        dev.synchronize()?;
    }
    let iters = 10;
    let t = Instant::now();
    for _ in 0..iters {
        for w in &weights {
            w.forward(&dev, &x_dev, &mut y, 1)?;
        }
    }
    dev.synchronize()?;
    let elapsed = t.elapsed().as_secs_f64() / iters as f64;

    // Per-shape attribution: which matrices actually cost the time, and what
    // bandwidth does each achieve? One synchronise per matrix, so this is a
    // diagnostic pass rather than a fast path.
    {
        let mut groups: std::collections::BTreeMap<(String, usize, usize), (usize, f64, usize)> =
            std::collections::BTreeMap::new();
        for w in &weights {
            let kind = match &w.data {
                gb10_model::LinearData::NvFp4 { .. } => "nvfp4",
                gb10_model::LinearData::Fp8 { .. } => "fp8",
                gb10_model::LinearData::Bf16 { .. } => "bf16",
            }
            .to_string();
            dev.synchronize()?;
            let t = Instant::now();
            for _ in 0..3 {
                w.forward(&dev, &x_dev, &mut y, 1)?;
            }
            dev.synchronize()?;
            let ms = t.elapsed().as_secs_f64() * 1e3 / 3.0;
            let e = groups.entry((kind, w.n, w.k)).or_insert((0, 0.0, 0));
            e.0 += 1;
            e.1 += ms;
            e.2 = w.traffic_bytes();
        }
        let mut rows: Vec<_> = groups.into_iter().collect();
        rows.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        println!();
        println!(
            "{:6} {:>7} {:>7} {:>5} {:>10} {:>10} {:>8}",
            "kind", "n", "k", "calls", "ms/step", "MB", "GB/s"
        );
        for ((kind, n, k), (calls, ms, bytes)) in rows.iter().take(14) {
            let gb = (*calls as f64) * (*bytes as f64) / 1e9;
            println!(
                "{kind:6} {n:7} {k:7} {calls:5} {ms:10.3} {:10.1} {:8.1}",
                gb * 1e3,
                gb / (ms / 1e3)
            );
        }
    }
    println!("per-token weight traffic : {:.3} GB", total_bytes as f64 / 1e9);
    println!("time per token           : {:.2} ms", elapsed * 1e3);
    println!("achieved bandwidth       : {:.1} GB/s", total_bytes as f64 / elapsed / 1e9);
    Ok(())
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
                // `weight` is stored PACKED as [N, K/2], so `info.shape[1]` is
                // half the true K. Passing it straight through made the kernel
                // read only half of every row while this function still
                // credited the full byte count -- which inflated the reported
                // bandwidth by exactly 2x for NVFP4 (and only NVFP4, since FP8
                // and bf16 are stored unpacked). Derive K from the group-scale
                // shape, exactly as `Store::linear` does.
                let group = st.info(&sname).with_context(|| sname.clone())?.shape[1];
                let k = group * 16;
                // Guard against the exact bug this replaced: a packed [N, K/2]
                // tensor silently halved K, which halved the bytes actually
                // read while still crediting the full count, inflating the
                // reported bandwidth by 2x.
                anyhow::ensure!(
                    raw.len() == n * k / 2 && sraw.len() == n * group,
                    "{name}: packed weight {} bytes / scale {} bytes inconsistent with n={n} k={k}",
                    raw.len(),
                    sraw.len()
                );
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

    // `--interleave` inserts a tiny dependency-breaking kernel after every
    // GEMV, mimicking the model's structure, to test whether separation --
    // rather than any property of the GEMV kernels -- is what costs bandwidth.
    let interleave = std::env::args().any(|a| a == "--interleave");
    let small_a: CudaSlice<f32> = dev.stream().alloc_zeros(32)?;
    let mut small_c: CudaSlice<f32> = dev.stream().alloc_zeros(32)?;

    let mut run_once = |weights: &[DevWeight], y: &mut CudaSlice<f32>| -> Result<()> {
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
            if interleave {
                dev.ops().add(&dev, &small_a, &small_a, &mut small_c, 32)?;
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
    println!("interleave: {interleave}");
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
