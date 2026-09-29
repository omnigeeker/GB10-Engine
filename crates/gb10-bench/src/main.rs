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
        "cublas-gemm" => cublas_gemm(),
        "dequant-parity" => dequant_parity(),
        "cublas-parity" => cublas_parity(),
        "tc-parity" => tc_parity(&model),
        "tc-phase" => tc_phase(),
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

/// cuBLAS bf16 GEMM at the shapes the prefill actually runs, to answer one
/// question with a measurement rather than an assumption: can this part reach
/// the ~43 TFLOPS that llama.cpp gets on bf16 tensor cores, when the current
/// fp32 CUDA-core GEMM is stuck at ~7?
///
/// Layout: y[t, n] = x[t, k] * W[n, k]^T with y row-major [t, n_out]. cuBLAS is
/// column-major and computes C(m, n) = op(A) * op(B), so with m = n_out and
/// n = t the output C viewed column-major is exactly our y row-major. W is
/// row-major [n_out, k] which, read column-major with lda = k, is W^T -- hence
/// transa = T. x is row-major [t, k] which, read column-major with ldb = k, is
/// already the (k, t) matrix we want -- hence transb = N.
fn cublas_gemm() -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use half::bf16;

    let dev = Device::new(0)?;
    let blas = CudaBlas::new(dev.stream().clone())?;

    println!("{:<22} {:>8} {:>8} {:>9} {:>10} {:>9}", "shape (n x k x t)", "ms", "TFLOP/s", "GB", "GB/s", "vs fp32");
    let fp32_tflops = 7.0f64;
    for (label, n, k, t) in [
        ("mlp gate/up", 17408usize, 5120usize, 2048usize),
        ("mlp down", 5120, 17408, 2048),
        ("lm_head", 248320, 5120, 2048),
        ("attn q_proj", 6144, 5120, 2048),
        // The model's small projections. Round 51 concluded that G's 3.2x gap is
        // the difference between this benchmark's shapes and the model's real mix,
        // so the small ones have to be measured rather than assumed.
        ("attn o_proj", 5120, 6144, 2048),
        ("attn k_proj", 1024, 5120, 2048),
        ("attn v_proj", 1024, 5120, 2048),
        ("small n, short t", 1024, 5120, 256),
        ("mlp gate/up t=256", 17408, 5120, 256),
    ] {
        let w = dev.stream().alloc_zeros::<bf16>(n * k)?;
        let x = dev.stream().alloc_zeros::<bf16>(t * k)?;
        let mut y = dev.stream().alloc_zeros::<bf16>(n * t)?;
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: t as i32,
            k: k as i32,
            alpha: bf16::from_f32(1.0),
            lda: k as i32,
            ldb: k as i32,
            beta: bf16::from_f32(0.0),
            ldc: n as i32,
        };
        for _ in 0..3 {
            unsafe { blas.gemm(cfg, &w, &x, &mut y) }?;
        }
        dev.synchronize()?;
        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            unsafe { blas.gemm(cfg, &w, &x, &mut y) }?;
        }
        dev.synchronize()?;
        let secs = t0.elapsed().as_secs_f64() / reps as f64;
        let flop = 2.0 * n as f64 * k as f64 * t as f64;
        let tflops = flop / secs / 1e12;
        let gbytes = (n * k * 2) as f64 / 1e9;
        println!(
            "{:<22} {:>8.2} {:>8.1} {:>9.3} {:>10.0} {:>8.1}x",
            label, secs * 1e3, tflops, gbytes, gbytes / secs, tflops / fp32_tflops
        );

        // How much of the model's in-situ cost is the allocation rather than the
        // GEMM? The prefill path allocates its three scratch buffers per matrix
        // with `alloc_zeros`, which zeroes ~67 GB per 2048-token chunk across the
        // whole model. That is the leading suspect for the gap between the
        // 11x this benchmark predicts and the 2.74x `prefill-shape` measured.
        // Re-allocating the same three buffers per rep, exactly as the model
        // does, isolates it.
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let _w = dev.stream().alloc_zeros::<bf16>(n * k)?;
            let _x = dev.stream().alloc_zeros::<bf16>(t * k)?;
            let _y = dev.stream().alloc_zeros::<bf16>(n * t)?;
        }
        dev.synchronize()?;
        let alloc_s = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{:<22} {:>8.2} {:>8} {:>9.3} {:>10.0} {:>8}   <- alloc_zeros only",
            "  (same buffers)", alloc_s * 1e3, "-", "-", "-", "-"
        );
    }
    println!("\nreference: the current fp32 CUDA-core GEMM measures ~7 TFLOP/s, {:.1}x below this", 43.0/7.0);
    Ok(())
}

/// Gate the NVFP4 -> bf16 dequantise kernel against the host reference
/// `dequant_nvfp4_row` on a synthetic matrix.
///
/// This is the one piece of the cuBLAS prefill path whose correctness depends on
/// a bit-level layout (nibble order, group-16 scale row, E4M3 decode) rather
/// than on plumbing, so it gets its own gate before anything is wired up.
fn dequant_parity() -> Result<()> {
    use crate::reference::dequant_nvfp4_row;
    use half::bf16;

    let dev = Device::new(0)?;
    let (n, k) = (256usize, 512usize);

    // Deterministic synthetic matrix. Scales are kept in E4M3's small positive
    // range (exponent 6, mantissa 0..7) so no byte decodes to NaN/Inf, and the
    // packed nibbles cover all 16 E2M1 codes.
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut packed = vec![0u8; n * (k / 2)];
    for b in packed.iter_mut() {
        *b = (next() & 0xFF) as u8;
    }
    let mut scales = vec![0u8; n * (k / 16)];
    for b in scales.iter_mut() {
        *b = 0x30 | ((next() & 0x07) as u8);
    }

    let cpu: Vec<f32> = (0..n)
        .flat_map(|row| {
            dequant_nvfp4_row(
                &packed[row * (k / 2)..(row + 1) * (k / 2)],
                &scales[row * (k / 16)..(row + 1) * (k / 16)],
                1.0,
                k,
            )
        })
        .collect();

    let w_dev = dev.stream().memcpy_stod(&packed)?;
    let sc_dev = dev.stream().memcpy_stod(&scales)?;
    let mut out_dev = dev.stream().alloc_zeros::<bf16>(n * k)?;
    dev.ops().dequant_nvfp4_to_bf16(&dev, &w_dev, &sc_dev, &mut out_dev, n, k)?;
    let gpu_raw = dev.stream().memcpy_dtov(&out_dev)?;

    let mut bad = 0usize;
    let mut worst = 0.0f32;
    let mut first_bad = None;
    for i in 0..n * k {
        let g = bf16::to_f32(gpu_raw[i]);
        let c = cpu[i];
        if g.to_bits() != c.to_bits() {
            bad += 1;
            let rel = if c == 0.0 { 0.0 } else { ((g - c) / c).abs() };
            if rel > worst {
                worst = rel;
            }
            if first_bad.is_none() {
                first_bad = Some((i, i / k, i % k, c, g));
            }
        }
    }
    // ---- fp8 (E4M3) path -------------------------------------------------
    // Per-tensor scale lives at a device address, as in the model.
    // Mask off the two E4M3 NaN encodings (S.1111.111 = 0x7F / 0xFF). E4M3 has
    // no Inf, and these are the only bytes where the CUDA and host E4M3 decoders
    // disagree; no real weight is ever NaN, so they are a harness artifact rather
    // than a kernel fault. They accounted for exactly 1039/131072 = 1/128 of the
    // elements before masking.
    let fp8_w: Vec<u8> = (0..n * k)
        .map(|_| {
            let b = (next() & 0xFF) as u8;
            if b & 0x7F == 0x7F { b & 0x80 } else { b }
        })
        .collect();
    let wscale: f32 = 0.0173;
    let cpu8: Vec<f32> = fp8_w
        .iter()
        .map(|&b| crate::reference::e4m3_to_f32(b) * wscale)
        .collect();
    let fw_dev = dev.stream().memcpy_stod(&fp8_w)?;
    let fs_dev = dev.stream().memcpy_stod(&[wscale])?;
    let mut fout_dev = dev.stream().alloc_zeros::<bf16>(n * k)?;
    dev.ops().dequant_fp8_to_bf16(&dev, &fw_dev, &fs_dev, &mut fout_dev, n, k)?;
    let fgpu = dev.stream().memcpy_dtov(&fout_dev)?;
    let mut fbad = 0usize;
    for i in 0..n * k {
        if fgpu[i].to_bits() != bf16::from_f32(cpu8[i]).to_bits() {
            fbad += 1;
        }
    }
    println!("dequant fp8   -> bf16   {n} x {k}  ({} elements)", n * k);
    println!(
        "  {} {}",
        if fbad == 0 {
            "bit-exact against e4m3_to_f32 * scale:"
        } else {
            "MISMATCH vs e4m3_to_f32 * scale:"
        },
        if fbad == 0 {
            format!("{} / {}", n * k, n * k)
        } else {
            format!("{fbad} bad")
        }
    );

    // ---- f32 -> bf16 activation cast ------------------------------------
    let src: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let sd = dev.stream().memcpy_stod(&src)?;
    let mut bo = dev.stream().alloc_zeros::<bf16>(1024)?;
    dev.ops().f32_to_bf16(&dev, &sd, &mut bo, 1024)?;
    let bg = dev.stream().memcpy_dtov(&bo)?;
    let act_ok = (0..1024).all(|i| bg[i].to_bits() == bf16::from_f32(src[i]).to_bits());
    println!("f32 -> bf16 cast: {}", if act_ok { "OK" } else { "MISMATCH" });

    println!("dequant nvfp4 -> bf16   {n} x {k}  ({n} rows, {} elements)", n * k);
    if bad == 0 && fbad == 0 && act_ok {
        println!("  bit-exact against dequant_nvfp4_row: {} / {}", n * k, n * k);
        println!("dequant-parity: OK");
        return Ok(());
    }
    if bad > 0 {
        let (i, row, col, c, g) = first_bad.unwrap();
        println!("  nvfp4 mismatches {bad} / {}   worst relative {worst:.3e}", n * k);
        println!("  first at idx {i} (row {row}, col {col}): cpu {c:e} gpu {g:e}");
    }
    anyhow::bail!("dequant-parity: FAILED (nvfp4 {bad} bad, fp8 {fbad} bad)");
}

/// Gate the cuBLAS bf16 GEMM wrapper's *layout* against a host matmul.
///
/// `cublas-gemm` already measures throughput; this checks the thing that is
/// actually easy to get wrong. The call transposes one operand and not the other,
/// and if `m`/`n` are swapped the result comes out silently transposed rather
/// than failing. Integer-valued inputs keep every product exact in bf16 (and the
/// sums exact in fp32), so the comparison is exact rather than tolerance-based.
/// Numerical parity of the two real prefill paths, on real model weights.
///
/// The tensor-core prefill GEMM (`GB10_TC_GEMM=1`) emits EOS as the first token
/// for prompts over ~970 tokens, where the fp32 CUDA-core path answers. Five
/// operand precisions were tried (bf16, fp16, bf16 two-way split, three-way
/// split, plus an fp8 scale fix) and all of them failed, including one at ~24
/// mantissa bits -- so "not enough precision" cannot be the explanation, and
/// llama.cpp runs tensor-core MMA over this same NVFP4 model successfully.
///
/// That leaves a systematic defect in this path, and the tool to find it is a
/// direct numerical comparison on real weights rather than the existing
/// 48x32x17 integer fixture, which only proves the operand layout is not
/// transposed.
///
/// This runs BOTH paths over the same activation for a spread of real matrices
/// -- covering all three stored dtypes (NVFP4, FP8, bf16), both layer kinds
/// (DeltaNet and full attention) and shapes from huge to tiny -- and reports the
/// error of the tensor-core path relative to the fp32 reference. If the error is
/// at fp32-rounding scale the GEMM is fine and the defect is elsewhere; if it is
/// orders of magnitude larger, the defect is here and the dtype/name that shows
/// it is the lead.
fn tc_parity(model: &str) -> Result<()> {
    let dev = Device::new(0)?;
    let store = gb10_model::Store::open(model, "model.language_model.")?;

    // `t` is deliberately in the failing regime: the bug is invisible at t=59
    // (the oracle fixture) and appears past ~970 tokens.
    // Overridable so the same matrices can be compared in the short-prompt
    // regime (where the token-exact oracle gate passes) and the long-prompt one
    // (where the engine emits EOS). A defect that only appears at large t is a
    // t-dependent bug; one that appears at both would have broken the oracle.
    let t: usize = std::env::var("TC_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);

    let names = [
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.down_proj",
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.linear_attn.in_proj_z",
        "layers.0.linear_attn.in_proj_a",
        "layers.3.self_attn.q_proj",
        "layers.3.self_attn.k_proj",
        "layers.3.self_attn.o_proj",
    ];

    // Deterministic, roughly unit-variance activations. Real prefill
    // activations are post-layernorm, so unit scale is the right regime.
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let unit = |r: u64| ((r >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;

    println!("tc-parity: tensor-core prefill GEMM vs the fp32 CUDA-core reference");
    println!("  t = {t} activations per matrix, unit-scale pseudo-random input");
    println!(
        "  {:<38} {:>6} {:>6} {:>11} {:>11} {:>11}",
        "matrix", "n", "k", "ref_rms", "tc_rms_err", "rel_rms"
    );

    let mut worst_rel = 0f64;
    let mut worst_name = String::new();
    for name in names {
        let lin = match store.linear(&dev, name) {
            Ok(l) => l,
            Err(e) => {
                println!("  {name:<38}  skipped: {e}");
                continue;
            }
        };
        let (n, k) = (lin.n, lin.k);
        let xs: Vec<f32> = (0..t * k).map(|_| unit(next()) as f32).collect();
        let xd = dev.stream().memcpy_stod(&xs)?;

        let run = |tc: bool| -> Result<Vec<f32>> {
            std::env::set_var("GB10_TC_GEMM", if tc { "1" } else { "0" });
            let mut y = dev.stream().alloc_zeros::<f32>(t * n)?;
            lin.forward_prefill(&dev, &xd, &mut y, t)?;
            Ok(dev.stream().memcpy_dtov(&y)?)
        };
        let y_tc = run(true)?;
        let y_ref = run(false)?;

        let n_el = t * n;
        let mut ref_sq = 0f64;
        let mut err_sq = 0f64;
        let mut max_abs = 0f64;
        for i in 0..n_el {
            let r = y_ref[i] as f64;
            let d = (y_tc[i] as f64) - r;
            ref_sq += r * r;
            err_sq += d * d;
            max_abs = max_abs.max(d.abs());
        }
        let ref_rms = (ref_sq / n_el as f64).sqrt();
        let err_rms = (err_sq / n_el as f64).sqrt();
        let rel = if ref_rms > 0.0 { err_rms / ref_rms } else { 0.0 };
        let dtype = match &lin.data {
            gb10_model::LinearData::NvFp4 { .. } => "nvfp4",
            gb10_model::LinearData::Fp8 { .. } => "fp8",
            gb10_model::LinearData::Bf16 { .. } => "bf16",
        };
        println!(
            "  {:<38} {:>6} {:>6} {:>11.4e} {:>11.4e} {:>11.3e}  {dtype}",
            name, n, k, ref_rms, err_rms, rel
        );
        if rel > worst_rel {
            worst_rel = rel;
            worst_name = format!("{name} ({dtype})");
        }
    }

    println!();
    println!("  worst: {worst_name} at rel_rms {worst_rel:.3e}");
    std::env::remove_var("GB10_TC_GEMM");

    // Threshold. The legitimate differences here are ~1e-6 for the nvfp4 path
    // (exact to fp32 rounding) and ~2.5e-3 for the fp8 path, where the fp32
    // reference rounds `e4m3(w) * wscale` into a bf16 staged weight and the
    // tensor-core path is the more accurate of the two. The defect this gate
    // exists for measured 1.5e3. So 1e-2 separates cleanly with two orders of
    // margin on the passing side and five on the failing side.
    const LIMIT: f64 = 1e-2;
    if worst_rel > LIMIT {
        anyhow::bail!(
            "tc-parity: FAILED -- {worst_name} disagrees with the fp32 reference by \
             rel_rms {worst_rel:.3e} (limit {LIMIT:.0e}). The tensor-core prefill \
             GEMM is not reproducing the fp32 path; do NOT ship it enabled."
        );
    }
    println!("tc-parity: OK (worst rel_rms {worst_rel:.3e} <= {LIMIT:.0e})");
    Ok(())
}

fn cublas_parity() -> Result<()> {
    use half::bf16;

    let dev = Device::new(0)?;
    let (n, k, t) = (48usize, 32usize, 17usize);

    // x[t][k], w[n][k] in small integers so the bf16 products are exact.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let xs: Vec<f32> = (0..t * k).map(|_| ((next() % 7) as f32) - 3.0).collect();
    let ws: Vec<f32> = (0..n * k).map(|_| ((next() % 7) as f32) - 3.0).collect();

    // Host reference: y[t][n] = sum_k x[t][k] * w[n][k]
    let mut want = vec![0f32; t * n];
    for i in 0..t {
        for j in 0..n {
            let mut a = 0f32;
            for kk in 0..k {
                a += xs[i * k + kk] * ws[j * k + kk];
            }
            want[i * n + j] = a;
        }
    }

    let xb: Vec<bf16> = xs.iter().map(|&v| bf16::from_f32(v)).collect();
    let wb: Vec<bf16> = ws.iter().map(|&v| bf16::from_f32(v)).collect();
    let xd = dev.stream().memcpy_stod(&xb)?;
    let wd = dev.stream().memcpy_stod(&wb)?;
    let mut yd = dev.stream().alloc_zeros::<bf16>(t * n)?;
    dev.ops().cublas_gemm_bf16(&dev, &wd, &xd, &mut yd, n, k, t)?;
    let got = dev.stream().memcpy_dtov(&yd)?;

    let mut bad = 0usize;
    let mut first = None;
    for i in 0..t * n {
        if bf16::to_f32(got[i]) != want[i] {
            bad += 1;
            if first.is_none() {
                first = Some((i, i / n, i % n, want[i], bf16::to_f32(got[i])));
            }
        }
    }
    // ---- full pipeline: GEMM (bf16) -> epilogue (bf16 -> fp32, * s2) --------
    let s2v: f32 = 0.375;
    let s2d = dev.stream().memcpy_stod(&[s2v])?;
    let mut yf = dev.stream().alloc_zeros::<f32>(t * n)?;
    dev.ops()
        .bf16_to_f32_scaled(&dev, &yd, &mut yf, &s2d, true, t * n)?;
    let yfv = dev.stream().memcpy_dtov(&yf)?;
    // The epilogue must reproduce bf16-rounded values scaled by s2 exactly:
    // x * s2 with s2 = 0.375 = 3/8 is a power-of-two times 3, so it stays exact
    // in fp32 for the small integer magnitudes these products have.
    let mut ep_bad = 0usize;
    for i in 0..t * n {
        let want_scaled = bf16::to_f32(got[i]) * s2v;
        if yfv[i] != want_scaled {
            ep_bad += 1;
        }
    }
    println!(
        "epilogue bf16->f32 * s2: {}",
        if ep_bad == 0 {
            format!("exact: {} / {}", t * n, t * n)
        } else {
            format!("MISMATCH: {ep_bad} bad")
        }
    );

    // ---- u16-held bf16 weights -> bf16 operand ----------------------------
    let u16s: Vec<u16> = wb.iter().map(|v| v.to_bits()).collect();
    let ud = dev.stream().memcpy_stod(&u16s)?;
    let mut ub = dev.stream().alloc_zeros::<bf16>(n * k)?;
    dev.ops().u16_to_bf16(&dev, &ud, &mut ub, n * k)?;
    let ug = dev.stream().memcpy_dtov(&ub)?;
    let u_ok = (0..n * k).all(|i| ug[i].to_bits() == wb[i].to_bits());
    println!("weights u16 -> bf16 operand: {}", if u_ok { "exact" } else { "MISMATCH" });

    println!("cublas bf16 layout: y[{t},{n}] = x[{t},{k}] * W[{n},{k}]^T");
    if bad == 0 && ep_bad == 0 && u_ok {
        println!("  exact against host matmul: {} / {}", t * n, t * n);
        println!("cublas-parity: OK");
        return Ok(());
    }
    let (i, row, col, w, g) = first.unwrap();
    println!("  mismatches {bad} / {}", t * n);
    println!("  first at idx {i} (y[{row},{col}]): want {w} got {g}");
    // A wholesale transpose is the failure this test is really for.
    let mut transposed = true;
    for i in 0..t {
        for j in 0..n {
            if bf16::to_f32(got[i * n + j]) != want[j * n + i] {
                transposed = false;
            }
        }
    }
    if transposed {
        println!("  NOTE: the output is exactly the transpose of the reference -- m/n or the transposes are swapped");
    }
    anyhow::bail!("cublas-parity: FAILED");
}

/// Per-phase timing of `forward_prefill_tensor_core` for the dominant shape.
///
/// Four independent end-to-end fits now agree with each other and disagree with
/// the phase model, so the way to locate the residual ~2 s per 2048-token chunk
/// is to time the phases directly instead of fitting the total again. Sizes are
/// the MLP gate/up matrix (17408 x 5120) at t = 2048, which is the largest one
/// on the prefill path.
fn tc_phase() -> Result<()> {
    use half::bf16;

    let dev = Device::new(0)?;
    let (n, k, t) = (17408usize, 5120usize, 2048usize);
    let reps = 5;

    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // Synthetic NVFP4 weights: packed E2M1 plus E4M3 group-16 scales.
    let wh: Vec<u8> = (0..n * k / 2).map(|_| (next() & 0xFF) as u8).collect();
    let sh: Vec<u8> = (0..n * k / 16)
        .map(|_| (0x30 | (next() & 0x07)) as u8)
        .collect();
    let xh: Vec<f32> = (0..t * k)
        .map(|_| ((next() % 1024) as f32) / 512.0 - 1.0)
        .collect();

    let wd = dev.stream().memcpy_stod(&wh)?;
    let sd = dev.stream().memcpy_stod(&sh)?;
    let xd = dev.stream().memcpy_stod(&xh)?;
    let s2 = dev.stream().memcpy_stod(&[1.0f32])?;

    let mut wb = dev.stream().alloc_zeros::<bf16>(n * k)?;
    let mut xb = dev.stream().alloc_zeros::<bf16>(t * k)?;
    let mut yb = dev.stream().alloc_zeros::<bf16>(t * n)?;
    let mut yf = dev.stream().alloc_zeros::<f32>(t * n)?;
    let kern = dev.ops();

    // Each phase timed over `reps`, after one warmup pass so the first-touch
    // page mapping is not billed to the phase.
    let mut time = |label: &str, f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        f()?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f()?;
        }
        dev.synchronize()?;
        let secs = t0.elapsed().as_secs_f64() / reps as f64;
        println!("  {:<34} {:>9.3} ms", label, secs * 1e3);
        Ok(secs)
    };

    println!("tc-phase: mlp gate/up  n={n} k={k} t={t}  (prefill, bf16 tensor core)");
    let a = time("alloc_zeros (w,x,y)", &mut || {
        let _a = dev.stream().alloc_zeros::<bf16>(n * k)?;
        let _b = dev.stream().alloc_zeros::<bf16>(t * k)?;
        let _c = dev.stream().alloc_zeros::<bf16>(t * n)?;
        Ok(())
    })?;
    let d = time("dequant_nvfp4_to_bf16 (w)", &mut || {
        kern.dequant_nvfp4_to_bf16(&dev, &wd, &sd, &mut wb, n, k)?;
        Ok(())
    })?;
    let c = time("f32_to_bf16 (x)", &mut || {
        kern.f32_to_bf16(&dev, &xd, &mut xb, t * k)?;
        Ok(())
    })?;
    let g = time("cublas_gemm_bf16", &mut || {
        kern.cublas_gemm_bf16(&dev, &wb, &xb, &mut yb, n, k, t)?;
        Ok(())
    })?;
    let e = time("bf16_to_f32_scaled (epilogue)", &mut || {
        kern.bf16_to_f32_scaled(&dev, &yb, &mut yf, &s2, true, t * n)?;
        Ok(())
    })?;

    let total = a + d + c + g + e;
    println!("  {:-<34} {:>9.3} ms", "sum", total * 1e3);
    println!(
        "  GEMM is {:.1}% of the pipeline; the non-GEMM work is {:.2}x the GEMM",
        g / total * 100.0,
        (total - g) / g
    );
    println!();
    println!("  For reference, the whole prefill measured 4.44 s per 2048-token chunk.");
    Ok(())
}
