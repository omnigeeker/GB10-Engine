//! Checkpoint access: quantized weights are uploaded to the device verbatim
//! and dequantized on the fly by the GEMV kernels.
//!
//! Nothing in this module converts a weight to fp32. Doing so would multiply
//! the per-token byte traffic by 2.5x for NVFP4 and 4x for FP8 and destroy the
//! roofline documented in `docs/PHYSICS.md`.

use anyhow::{bail, Context, Result};
use gb10_core::safetensors::{DType, ShardedSafeTensors, TensorInfo};
use gb10_cuda::{CudaEvent, CUevent_flags, CudaSlice, Device, TcScratch};
use std::path::Path;
use std::sync::Mutex;

/// DIAGNOSTIC (round 58): GPU time of the cuBLAS calls in the prefill path.
/// Rounds 53-57 named this measurement and never performed it: the model moves
/// 44.6 GFLOP/token over shapes that measure 74.8-89.2 TFLOP/s in isolation, yet
/// takes 2.4-3.5x longer than that implies, and every other candidate has been
/// eliminated. Events are recorded asynchronously and read once at the end --
/// `elapsed_ms` synchronizes, so it must never be called inside the op.
pub static GEMM_EVENTS: Mutex<Vec<Vec<CudaEvent>>> = Mutex::new(Vec::new());

/// Phase names, in the order `gemm_event_snapshot` returns their totals.
/// The epilogue is not bracketed here; it is the remainder to the op total.
pub const PHASES: [&str; 4] = ["weight stage", "activ cast", "cublas gemm", "epilogue"];

/// Drain the recorded events and return (per-phase GPU ms, call count). Four
/// events per op bracket three phases so no two phases share a boundary.
pub fn gemm_event_snapshot() -> ([f64; 4], usize) {
    let mut v = GEMM_EVENTS.lock().unwrap();
    let n = v.len();
    let mut acc = [0.0f64; 4];
    for ev in v.drain(..) {
        for i in 0..4 {
            if let Ok(ms) = ev[i].elapsed_ms(&ev[i + 1]) {
                acc[i] += ms as f64;
            }
        }
    }
    (acc, n)
}

/// A weight matrix plus the scales its format needs.
pub enum LinearData {
    /// `w` packed E2M1 `[N, K/2]`, `wscale` E4M3 group scales `[N, K/16]`,
    /// `scale2` the per-tensor global scale.
    NvFp4 {
        w: CudaSlice<u8>,
        wscale: CudaSlice<u8>,
        scale2: CudaSlice<f32>,
        /// The same constant on the host, so it can be folded into the GEMM's
        /// `alpha` instead of being applied by a separate `f32_scale` pass.
        scale2_host: f32,
    },
    /// `w` E4M3 `[N, K]` with a per-tensor fp32 scale.
    Fp8 {
        w: CudaSlice<u8>,
        scale: CudaSlice<f32>,
        scale_host: f32,
    },
    /// Unquantized bf16 `[N, K]`.
    Bf16 { w: CudaSlice<u16> },
}

pub struct Linear {
    pub n: usize,
    pub k: usize,
    pub data: LinearData,
}

impl Linear {
    /// `y[b, :] = W @ x[b, :]` for `b` in `0..batch`.
    pub fn forward(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<()> {
        let kern = dev.kernels();
        match &self.data {
            LinearData::NvFp4 { w, wscale, scale2, .. } => {
                kern.nvfp4_gemv(dev, x, w, wscale, scale2, y, self.n, self.k, batch)?
            }
            LinearData::Fp8 { w, scale, .. } => {
                kern.fp8_gemv(dev, x, w, scale, y, self.n, self.k, batch)?
            }
            LinearData::Bf16 { w } => kern.bf16_gemv(dev, x, w, y, self.n, self.k, batch)?,
        }
        Ok(())
    }

    /// Batched prefill: `y[t, :] = W @ x[t, :]` for all `t` in one launch.
    ///
    /// Unlike `forward(batch = t)`, which runs `t` independent GEMVs and
    /// re-reads W every time, this reads each weight once and reuses it across
    /// all `t` activations.
    /// The prefill GEMM on bf16 tensor cores, via cuBLAS.
    ///
    /// The fp32 CUDA-core GEMM this replaces runs at ~7 TFLOP/s, which is 76% of
    /// what fp32 can reach on this part at all (128 FMA/cycle/SM * 48 SM *
    /// 1.5 GHz = 9.2 TFLOP/s), so there is no fp32 tuning left in it. cuBLAS bf16
    /// measures ~80 TFLOP/s at these exact shapes -- an 11-12x step -- and the
    /// prefill is 75-93% GEMM, so this is the whole remaining gap.
    ///
    /// Three kernels per matrix: dequantise W to bf16, cast the activations to
    /// bf16, then one cuBLAS GEMM that accumulates in fp32 and **writes fp32**
    /// into the caller's output. The NVFP4 path owes one more tiny kernel to
    /// apply its per-tensor `s2` (which must *not* be folded into the weights --
    /// the reference applies it to the fp32 accumulator).
    ///
    /// Rounding the GEMM *output* to bf16 was a real accuracy bug, not a
    /// rounding detail: it discarded precision cuBLAS had already computed, in
    /// roughly 450 GEMMs per prefill, feeding a 48-layer Gated-DeltaNet
    /// recurrence whose state compounds the error with sequence length. It went
    /// unnoticed because perplexity at a 512-token window moves only 0.023%
    /// (mean NLL 1.875052 -> 1.875490) while the long-context greedy argmax
    /// flips: at 970+ prompt tokens the model emitted EOS as its first token and
    /// answered nothing, where the fp32 path answers correctly. Measured across
    /// a fixed battery of 5 long prompts as 5/5 -> 0/5. See `bench/longctx`.
    ///
    /// The buffers are per-call rather than a shared persistent scratch, which
    /// relies on cudarc's caching allocator to make the repeat allocations
    /// cheap. It keeps this change local instead of threading a scratch through
    /// 17 call sites. If the measured gain is real, the next step is to hoist
    /// them into one 321 MB shared scratch (largest W 17408x5120 = 178 MB, and
    /// x/y 2048x17408 = 71 MB each; `lm_head` is not on this path).
    ///
    /// `alloc_zeros` is the only allocator cudarc exposes, so every buffer is
    /// zeroed before being fully overwritten. That is pure waste and is the
    /// first thing to remove if the profile says so.
    #[allow(clippy::too_many_arguments)]
    fn forward_prefill_tensor_core(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        t: usize,
    ) -> Result<()> {
        use half::bf16;
        let kern = dev.ops();
        let (n, k) = (self.n, self.k);
        let stream = dev.stream();

        // Grow the shared scratch to fit, then split it so the three buffers can
        // be borrowed independently. Each is only ever allocated once and then
        // grown, which is what removes the per-call allocator cost.
        let mut sc = dev.tc_scratch();
        if sc.w.as_ref().map_or(true, |b| b.len() < n * k) {
            sc.w = Some(stream.alloc_zeros::<bf16>(n * k)?);
        }
        if sc.x.as_ref().map_or(true, |b| b.len() < t * k) {
            sc.x = Some(stream.alloc_zeros::<bf16>(t * k)?);
        }
        if sc.x2.as_ref().map_or(true, |b| b.len() < t * k) {
            sc.x2 = Some(stream.alloc_zeros::<bf16>(t * k)?);
        }
        if sc.x3.as_ref().map_or(true, |b| b.len() < t * k) {
            sc.x3 = Some(stream.alloc_zeros::<bf16>(t * k)?);
        }
        // A literal 1.0, passed to the fp8 dequant so it does not fold the
        // per-tensor scale into the bf16 weights (see below).
        if sc.one.is_none() {
            let mut one = stream.alloc_zeros::<f32>(1)?;
            dev.stream().memcpy_htod(&[1.0f32], &mut one)?;
            sc.one = Some(one);
        }
        // Named with an `s` prefix because `x` and `y` are already the fp32
        // input and output of this function.
        //
        // There is no `sc.y`: the GEMM writes fp32 directly into the caller's
        // `y`, so the accumulator is never rounded on the way out.
        let TcScratch { w: sw, x: sx, x2: sx2, x3: sx3, one, .. } = &mut *sc;
        let (wb, xhi, xmid, xlo) = (
            sw.as_mut().unwrap(),
            sx.as_mut().unwrap(),
            sx2.as_mut().unwrap(),
            sx3.as_mut().unwrap(),
        );
        let one = one.as_ref().unwrap();

        // Gated: the server shares this path and only `gemm_event_snapshot`
        // drains the Vec, so unconditional recording would retain events for
        // every request. Four events bracket the three per-op phases.
        let ev_ctx = dev.stream().context().clone();
        let want_ev = std::env::var("GB10_GEMM_EVENTS").is_ok();
        let mut evs: Option<Vec<CudaEvent>> = None;
        if want_ev {
            let mut t = Vec::with_capacity(5);
            let mut ok = true;
            for _ in 0..5 {
                match ev_ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)) {
                    Ok(e) => t.push(e),
                    Err(_) => { ok = false; break; }
                }
            }
            if ok { evs = Some(t); }
        }
        macro_rules! mark {
            ($i:expr) => {
                if let Some(t) = &evs {
                    let _ = t[$i].record(dev.stream());
                }
            };
        }
        mark!(0);
        match &self.data {
            LinearData::NvFp4 { w: qw, wscale, .. } => {
                // The 2D kernel removes a per-element 32-bit integer division
                // from the dequantise, which is what makes it ALU-bound rather
                // than bandwidth-bound. It indexes rows with `blockIdx.y`, so it
                // needs `N <= 65535`; `lm_head` is [248320, 5120] and must stay
                // on the grid-stride kernel. This is the one call site that
                // serves both, so the branch belongs here rather than in a
                // widened guard -- a loud error on `lm_head` would surface as
                // garbage output on long contexts while a short `generate`
                // still looked perfect.
                // `GB10_DEQ_2D=0` forces the original grid-stride kernel, so the
                // two forms can be compared in one session with one binary.
                // (This box has drifted 23% between sessions; only same-session
                // pairs are comparable.)
                let use_2d = std::env::var("GB10_DEQ_2D").map(|v| v != "0").unwrap_or(true);
                if use_2d && n <= 65535 {
                    kern.dequant_nvfp4_to_bf16_2d(dev, qw, wscale, wb, n, k)?;
                } else {
                    kern.dequant_nvfp4_to_bf16(dev, qw, wscale, wb, n, k)?;
                }
            }
            LinearData::Fp8 { w: qw, .. } => {
                // Deliberately dequantise with a 1.0 scale and apply the real
                // per-tensor scale to the fp32 accumulator below. fp8's scale is
                // a single fp32 constant, so folding it into the bf16 weight
                // rounds every weight to 8 mantissa bits for no reason -- the
                // reference (`stage_wtile_fp8`) applies it in fp32 during
                // staging. These are the attention projections.
                kern.dequant_fp8_to_bf16(dev, qw, one, wb, n, k)?;
            }
            LinearData::Bf16 { w: qw } => {
                kern.u16_to_bf16(dev, qw, wb, n * k)?;
            }
        }
        mark!(1);
        // How many bf16 parts the activation is split into.
        //
        // This was 3 for a while, on the theory that the long-context failure
        // was a precision floor (8 bits failed, 10 failed, 16 failed, 24
        // worked in the sense of not being *worse*). That theory was wrong: the
        // real defect was a truncated grid in the element-wise kernels, and
        // once it is fixed ONE bf16 operand is enough. Kept selectable so the
        // cost/precision trade-off stays measurable rather than asserted.
        //   1: one GEMM   (fastest; bf16 operand, ~8 mantissa bits)
        //   2: two GEMMs  (~16 bits)
        //   3: three GEMMs (~24 bits)
        let split: u32 = std::env::var("GB10_TC_SPLIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        match split {
            1 => {
                kern.f32_to_bf16(dev, x, xhi, t * k)?;
            }
            2 => {
                kern.f32_split_bf16(dev, x, xhi, xlo, t * k)?;
            }
            _ => {
                kern.f32_split3_bf16(dev, x, xhi, xmid, xlo, t * k)?;
            }
        }
        mark!(2);
        // Three GEMMs accumulating in fp32 straight into the caller's `y`: the
        // first starts from zero and the other two add their residual terms, so
        // the result is W*(hi + mid + lo) with no intermediate rounding to 16
        // bits. This is the price of fp32-grade precision on tensor cores --
        // three bf16 GEMMs against one fp32 CUDA-core GEMM, still ~3.7x cheaper.
        // The per-tensor scale, folded into the GEMM's `alpha` rather than
        // applied to `y` afterwards. The epilogue pass this replaces measured
        // 1301 ms at 8192 tokens -- 10.5% of the whole prefill, and the second
        // largest GEMM phase in the model.
        let alpha_scale: f32 = match &self.data {
            LinearData::NvFp4 { scale2_host, .. } => *scale2_host,
            LinearData::Fp8 { scale_host, .. } => *scale_host,
            LinearData::Bf16 { .. } => 1.0,
        };
        kern.cublas_gemm_bf16_f32(dev, wb, xhi, y, n, k, t, 0.0, alpha_scale)?;
        if split >= 2 {
            kern.cublas_gemm_bf16_f32(dev, wb, xlo, y, n, k, t, 1.0, alpha_scale)?;
        }
        if split >= 3 {
            kern.cublas_gemm_bf16_f32(dev, wb, xmid, y, n, k, t, 1.0, alpha_scale)?;
        }
        mark!(3);

        // The per-tensor scales are applied to the fp32 accumulator, never
        // folded into a 16-bit weight: NVFP4's `s2` and fp8's `scale` both are
        // single constants, and the reference applies both in fp32. `bf16` rows
        // have no scale and are already done.
        // The scales are applied inside the GEMM now (see `alpha_scale`
        // above); this pass used to be `kern.f32_scale(dev, y, scale, t * n)`.
        let _ = &self.data;
        mark!(4);
        if let Some(t) = evs.take() {
            GEMM_EVENTS.lock().unwrap().push(t);
        }
        Ok(())
    }

    pub fn forward_prefill(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        t: usize,
    ) -> Result<()> {
        // The GEMM tiles N in blocks of 64. Below that the whole grid collapses
        // to a single block on a single SM: `in_proj_a/b` are [48, 5120], which
        // measured 73.5 ms that way against 16.5 ms for the batched GEMV. For a
        // matrix this small, re-reading it per token is far cheaper than
        // starving 47 of the 48 SMs.
        //
        // The same trade-off turns out to hold for short prompts, and much more
        // strongly than expected. `nsys` puts the GEMM's weight loads at ~58 GB/s
        // against ~148 GB/s for the GEMV path on the same weights -- a 2.5x gap
        // -- because the GEMM's staging scatters each load instruction across 8
        // rows to fill a shared tile, while the GEMV has no tile and streams 256
        // contiguous bytes of one row per instruction.
        //
        // So the batched GEMV wins whenever the prompt is short enough that
        // re-reading W is cheaper than the GEMM's slower loads. Measured on
        // `forward-cost` (t: GEMM -> batched GEMV):
        //
        //     1: 271.5 -> 116.7 ms   2: 269.4 -> 145.8   4: 280.3 -> 157.2
        //     8: 277.5 -> 193.2     16: 286.9 -> 347.5
        //
        // The crossover is between 8 and 16; 16 is worse because the batched
        // GEMV then re-reads every weight once per token, which outweighs its
        // better load pattern. The 8..16 range is unmeasured, so 8 is the
        // conservative cut.
        // Small-n projections must take the batched GEMV only for SHORT
        // prompts, and the tensor-core GEMM otherwise.
        //
        // This clause used to read `self.n < 256 || t <= 16`, with no `t`
        // dependence on the `n` arm. The comment above it -- "re-reading it per
        // token is far cheaper than starving 47 of the 48 SMs" -- is true at
        // t = 16 and catastrophically false at t = 8192, where re-reading
        // `in_proj_a`'s 0.49 MB weight once per token is 4.0 GB of traffic for
        // 4.8 ms of arithmetic. Because the arm had no `t`, a conclusion that
        // holds at t = 16 was applied at every length.
        //
        // Measured crossover for the two n = 48 projections, from
        // `prefill-shape` with the PROJ instrument (GEMV -> GEMM, in_proj_a):
        //
        //     t =   32:  2 ms -> 11 ms   (GEMV wins; the GEMM number is
        //     t =  128:  6 ms -> 10 ms    inflated by one-time cuBLAS init)
        //     t =  512: 23 ms ->  5 ms   GEMM 4.6x
        //     t = 2048: 96 ms -> 19 ms   GEMM 5.1x
        //     t = 8192: 368 ms -> 82 ms  GEMM 4.5x
        //
        // The crossover is between 128 and 512. 256 is the conservative cut,
        // matching the shape of the `t <= 16` clause rather than guessing an
        // exact point between two measured samples.
        let small_n_cut: usize = 256;
        if t <= 16 || (self.n < 256 && t < small_n_cut) {
            return self.forward(dev, x, y, t);
        }
        // Tensor-core prefill GEMM, on by default. Set `GB10_TC_GEMM=0` to fall
        // back to the fp32 CUDA-core path (kept as the reference the gates
        // compare against).
        //
        // This was opt-in for a while because it broke long context: for any
        // prompt over ~900 tokens the engine emitted EOS as its first sampled
        // token and answered nothing at all. Five operand-precision rewrites
        // (bf16, fp16, two-way and three-way bf16 splits, plus an fp8 scale
        // fix) all failed to change that, which read as a precision floor above
        // 10 bits. It was not a precision problem at all.
        //
        // The element-wise kernels that stage the operands and apply the
        // per-tensor scale used `if (i < n)` with a host grid capped at 65535
        // blocks x 256 threads = 16,776,960 elements. Every element past that
        // was silently skipped, and for `mlp.gate_proj` (n = 17408) the
        // activation covers t*17408 elements, which crosses the cap at t = 964.
        // That is exactly where the answers stopped: 950-token prompts worked,
        // 970-token prompts did not. The skipped tail was also the part the
        // `s2` scale had not been applied to, which is why it looked like a
        // precision problem instead of a missing-work problem.
        //
        // Fixed by making all eight staging/scale kernels grid-stride, so their
        // correctness no longer depends on the grid size. `gb10-bench tc-parity`
        // is the gate: on real weights it shows the tensor-core path within
        // 1.3e-6 of the fp32 reference at t=1024, where it used to be 1.5e3.
        if std::env::var("GB10_TC_GEMM").map(|v| v != "0").unwrap_or(true) {
            return self.forward_prefill_tensor_core(dev, x, y, t);
        }
        let kern = dev.ops();
        match &self.data {
            LinearData::NvFp4 { w, wscale, scale2, .. } => {
                kern.nvfp4_gemm(dev, w, wscale, scale2, x, y, self.n, self.k, t)?
            }
            LinearData::Fp8 { w, scale, .. } => {
                kern.fp8_gemm(dev, w, scale, x, y, self.n, self.k, t)?
            }
            LinearData::Bf16 { w } => kern.bf16_gemm(dev, w, x, y, self.n, self.k, t)?,
        }
        Ok(())
    }

    /// Bytes this matrix contributes to the per-token weight stream.
    pub fn traffic_bytes(&self) -> usize {
        match &self.data {
            LinearData::NvFp4 { w, wscale, .. } => w.len() + wscale.len() + 4,
            LinearData::Fp8 { w, .. } => w.len() + 4,
            LinearData::Bf16 { w } => w.len() * 2,
        }
    }
}

/// Read-only view over the checkpoint's shards.
pub struct Store {
    st: ShardedSafeTensors,
    prefix: String,
}

impl Store {
    pub fn open(dir: impl AsRef<Path>, prefix: &str) -> Result<Self> {
        let st = ShardedSafeTensors::open(dir.as_ref())
            .with_context(|| format!("opening checkpoint at {}", dir.as_ref().display()))?;
        Ok(Self {
            st,
            prefix: prefix.to_string(),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.st.len()
    }

    fn full(&self, name: &str) -> String {
        format!("{}{}", self.prefix, name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.contains(&self.full(name))
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        let full = self.full(name);
        self.st
            .info(&full)
            .with_context(|| format!("tensor {full} not in checkpoint"))
    }

    fn bytes(&self, name: &str) -> Result<&[u8]> {
        let full = self.full(name);
        self.st
            .tensor_bytes(&full)
            .with_context(|| format!("reading {full}"))
    }

    /// Upload a quantized linear layer, choosing the format from the stored
    /// dtype. `name` is the module path without the `.weight` suffix, e.g.
    /// `layers.0.mlp.gate_proj`.
    pub fn linear(&self, dev: &Device, name: &str) -> Result<Linear> {
        let wname = format!("{name}.weight");
        let info = self.info(&wname)?.clone();
        let shape = &info.shape;
        if shape.len() != 2 {
            bail!("{}: expected a 2-D weight, got {:?}", name, shape);
        }
        let n = shape[0];

        let data = match info.dtype {
            DType::U8 => {
                // NVFP4: K is only recoverable from the group-scale shape.
                let sinfo = self.info(&format!("{name}.weight_scale"))?.clone();
                if sinfo.shape.len() != 2 {
                    bail!("{}: weight_scale shape {:?}", name, sinfo.shape);
                }
                let group = sinfo.shape[1];
                let k = group * 16;
                if shape[1] != k / 2 {
                    bail!(
                        "{}: packed weight {:?} inconsistent with K={k} from scale {:?}",
                        name,
                        shape,
                        sinfo.shape
                    );
                }
                let w = dev.stream().memcpy_stod(self.bytes(&wname)?)?;
                let wscale = dev
                    .stream()
                    .memcpy_stod(self.bytes(&format!("{name}.weight_scale"))?)?;
                let s2host = self.f32(&format!("{name}.weight_scale_2"))?;
                let scale2 = dev.stream().memcpy_stod(&s2host)?;
                LinearData::NvFp4 { w, wscale, scale2, scale2_host: s2host[0] }
            }
            DType::F8_E4M3 => {
                let w = dev.stream().memcpy_stod(self.bytes(&wname)?)?;
                let shost = self.f32(&format!("{name}.weight_scale"))?;
                let scale = dev.stream().memcpy_stod(&shost)?;
                LinearData::Fp8 { w, scale, scale_host: shost[0] }
            }
            DType::BF16 => {
                let w = self.bf16_u16(dev, &wname)?;
                LinearData::Bf16 { w }
            }
            other => bail!("{wname}: unsupported weight dtype {other:?}"),
        };

        let k = match &data {
            LinearData::NvFp4 { .. } => self.info(&format!("{name}.weight_scale"))?.shape[1] * 16,
            LinearData::Fp8 { w, .. } => w.len() / n,
            LinearData::Bf16 { w } => w.len() / n,
        };
        Ok(Linear { n, k, data })
    }

    fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        if info.dtype != DType::F32 {
            bail!("{}: expected F32, got {:?}", name, info.dtype);
        }
        let b = self.bytes(name)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// bf16 tensor uploaded as raw u16 (for the bf16 GEMV kernel).
    pub fn bf16_u16(&self, dev: &Device, name: &str) -> Result<CudaSlice<u16>> {
        let info = self.info(name)?;
        if info.dtype != DType::BF16 {
            bail!("{}: expected BF16, got {:?}", name, info.dtype);
        }
        let b = self.bytes(name)?;
        let v: Vec<u16> = b
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(dev.stream().memcpy_stod(&v)?)
    }

    /// bf16 tensor widened to fp32 on the host. Only used for small tensors
    /// (norms, conv weights, `A_log`, `dt_bias`) where the traffic is noise.
    pub fn bf16_f32(&self, dev: &Device, name: &str) -> Result<CudaSlice<f32>> {
        let info = self.info(name)?;
        if info.dtype != DType::BF16 {
            bail!("{}: expected BF16, got {:?}", name, info.dtype);
        }
        let b = self.bytes(name)?;
        let v: Vec<f32> = b
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]) as u32;
                f32::from_bits(bits << 16)
            })
            .collect();
        Ok(dev.stream().memcpy_stod(&v)?)
    }

    /// fp32 tensor uploaded directly (used for reference weights in tests).
    pub fn f32_dev(&self, dev: &Device, name: &str) -> Result<CudaSlice<f32>> {
        let v = self.f32(name)?;
        Ok(dev.stream().memcpy_stod(&v)?)
    }
}
