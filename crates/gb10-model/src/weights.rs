//! Checkpoint access: quantized weights are uploaded to the device verbatim
//! and dequantized on the fly by the GEMV kernels.
//!
//! Nothing in this module converts a weight to fp32. Doing so would multiply
//! the per-token byte traffic by 2.5x for NVFP4 and 4x for FP8 and destroy the
//! roofline documented in `docs/PHYSICS.md`.

use anyhow::{bail, Context, Result};
use gb10_core::safetensors::{DType, ShardedSafeTensors, TensorInfo};
use gb10_cuda::{CudaSlice, Device};
use std::path::Path;

/// A weight matrix plus the scales its format needs.
pub enum LinearData {
    /// `w` packed E2M1 `[N, K/2]`, `wscale` E4M3 group scales `[N, K/16]`,
    /// `scale2` the per-tensor global scale.
    NvFp4 {
        w: CudaSlice<u8>,
        wscale: CudaSlice<u8>,
        scale2: CudaSlice<f32>,
    },
    /// `w` E4M3 `[N, K]` with a per-tensor fp32 scale.
    Fp8 {
        w: CudaSlice<u8>,
        scale: CudaSlice<f32>,
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
            LinearData::NvFp4 { w, wscale, scale2 } => {
                kern.nvfp4_gemv(dev, x, w, wscale, scale2, y, self.n, self.k, batch)?
            }
            LinearData::Fp8 { w, scale } => {
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
        if self.n < 256 {
            return self.forward(dev, x, y, t);
        }
        let kern = dev.ops();
        match &self.data {
            LinearData::NvFp4 { w, wscale, scale2 } => {
                kern.nvfp4_gemm(dev, w, wscale, scale2, x, y, self.n, self.k, t)?
            }
            LinearData::Fp8 { w, scale } => {
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
                LinearData::NvFp4 { w, wscale, scale2 }
            }
            DType::F8_E4M3 => {
                let w = dev.stream().memcpy_stod(self.bytes(&wname)?)?;
                let shost = self.f32(&format!("{name}.weight_scale"))?;
                let scale = dev.stream().memcpy_stod(&shost)?;
                LinearData::Fp8 { w, scale }
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
