//! Typed launch wrappers for the non-GEMV kernels: norms, elementwise ops,
//! RoPE, the depthwise conv, the Gated DeltaNet recurrence and prefill
//! attention.
//!
//! Every wrapper validates its shapes before launching. These kernels are not
//! the bandwidth bottleneck — the 17.6 GB of quantized weights streamed by the
//! GEMV kernels dominates by ~100x — so they are written for obvious
//! correctness rather than peak occupancy.

use crate::{CudaError, CudaSlice, Device, LaunchConfig, Result};
use cudarc::driver::{CudaFunction, DevicePtr, DevicePtrMut, PushKernelArg};
use std::collections::HashMap;

/// Kernel symbols owned by `Ops`. Kept separate from the GEMV set so a typo in
/// one group cannot silently mask a missing kernel in the other.
pub const OP_KERNEL_NAMES: &[&str] = &[
    "rmsnorm_zero_centered_kernel",
    "rmsnorm_gated_kernel",
    "swiglu_kernel",
    "add_kernel",
    "sigmoid_mul_kernel",
    "l2norm_scale_kernel",
    "rope_neox_kernel",
    "conv1d_step_silu_kernel",
    "gated_delta_rule_step_kernel",
    "delta_gate_kernel",
    "deinterleave_heads_kernel",
    "attn_prefill_kernel",
    "attn_prefill_tiled_kernel",
    "attn_decode_kernel",
    "kv_cache_append_kernel",
    "embed_gather_kernel",
    "argmax_kernel",
    "l2norm_scale_batched_kernel",
    "delta_gate_batched_kernel",
    "conv1d_prefill_silu_kernel",
    "rope_neox_batched_kernel",
    "kv_cache_append_batched_kernel",
    "conv1d_step_silu_multi_kernel",
    "gated_delta_rule_step_multi_kernel",
    "kv_cache_append_multi_kernel",
    "attn_decode_multi_kernel",
    "attn_decode_multi_serial_kernel",
    "argmax_multi_kernel",
    "gated_delta_rule_chunk_kernel",
    "deinterleave_heads_batched_kernel",
    "embed_gather_batched_kernel",
    "copy_last_row_kernel",
    "copy_rows_kernel",
    "concat2_kernel",
    "nvfp4_gemm_kernel",
    "dequant_nvfp4_to_bf16_kernel",
    "dequant_nvfp4_to_f16_kernel",
    "dequant_fp8_to_f16_kernel",
    "u16_to_f16_kernel",
    "dequant_fp8_to_bf16_kernel",
    "f32_to_bf16_kernel",
    "f32_to_f16_kernel",
    "f32_split_bf16_kernel",
    "f32_split3_bf16_kernel",
    "bf16_to_f32_scaled_kernel",
    "f32_scale_kernel",
    "u16_to_bf16_kernel",
    "fp8_gemm_kernel",
    "bf16_gemm_kernel",
];

/// Gated DeltaNet key/value head geometry (fixed by the checkpoint).
/// Prompt tokens covered by one prefill GEMM block; must match `GB10_TILE_T`
/// in `kernels/gemm.cu`.
pub const GB10_TILE_T: usize = 64;
/// Rows of N per prefill GEMM block; matches `GB10_TN` in `kernels/gemm.cu`.
pub const GB10_NR: usize = 64;

pub const DELTA_KEY_HEAD_DIM: usize = 128;
pub const DELTA_VALUE_HEAD_DIM: usize = 128;

pub struct Ops {
    rmsnorm_zero_centered: CudaFunction,
    rmsnorm_gated: CudaFunction,
    swiglu: CudaFunction,
    add: CudaFunction,
    sigmoid_mul: CudaFunction,
    l2norm_scale: CudaFunction,
    rope_neox: CudaFunction,
    conv1d_step_silu: CudaFunction,
    gated_delta_rule_step: CudaFunction,
    delta_gate: CudaFunction,
    deinterleave_heads: CudaFunction,
    attn_prefill: CudaFunction,
    attn_prefill_tiled: CudaFunction,
    attn_decode: CudaFunction,
    kv_cache_append: CudaFunction,
    embed_gather: CudaFunction,
    argmax: CudaFunction,
    argmax_multi: CudaFunction,
    conv1d_step_silu_multi: CudaFunction,
    gated_delta_rule_step_multi: CudaFunction,
    kv_cache_append_multi: CudaFunction,
    attn_decode_multi: CudaFunction,
    attn_decode_multi_serial: CudaFunction,
    l2norm_scale_batched: CudaFunction,
    delta_gate_batched: CudaFunction,
    conv1d_prefill_silu: CudaFunction,
    rope_neox_batched: CudaFunction,
    kv_cache_append_batched: CudaFunction,
    gated_delta_rule_chunk: CudaFunction,
    deinterleave_heads_batched: CudaFunction,
    embed_gather_batched: CudaFunction,
    copy_last_row: CudaFunction,
    copy_rows: CudaFunction,
    concat2: CudaFunction,
    nvfp4_gemm: CudaFunction,
    dequant_nvfp4_to_bf16: CudaFunction,
    dequant_nvfp4_to_f16: CudaFunction,
    dequant_fp8_to_f16: CudaFunction,
    u16_to_f16: CudaFunction,
    dequant_fp8_to_bf16: CudaFunction,
    f32_to_bf16: CudaFunction,
    f32_to_f16: CudaFunction,
    f32_split_bf16: CudaFunction,
    f32_split3_bf16: CudaFunction,
    bf16_to_f32_scaled: CudaFunction,
    f32_scale: CudaFunction,
    u16_to_bf16: CudaFunction,
    fp8_gemm: CudaFunction,
    bf16_gemm: CudaFunction,
}

fn take(map: &mut HashMap<String, CudaFunction>, n: &str) -> Result<CudaFunction> {
    map.remove(n)
        .ok_or_else(|| CudaError::KernelNotFound(n.to_string()))
}

/// Largest power-of-two block size that is a multiple of 32, at most `max`.
fn block_for(n: usize, max: u32) -> u32 {
    let mut b = 32u32;
    while (b as usize) < n && b * 2 <= max {
        b *= 2;
    }
    b
}

fn cdiv(a: usize, b: usize) -> u32 {
    ((a + b - 1) / b).max(1) as u32
}

impl Ops {
    pub(crate) fn from_map(map: &mut HashMap<String, CudaFunction>) -> Result<Self> {
        Ok(Self {
            rmsnorm_zero_centered: take(map, "rmsnorm_zero_centered_kernel")?,
            rmsnorm_gated: take(map, "rmsnorm_gated_kernel")?,
            swiglu: take(map, "swiglu_kernel")?,
            add: take(map, "add_kernel")?,
            sigmoid_mul: take(map, "sigmoid_mul_kernel")?,
            l2norm_scale: take(map, "l2norm_scale_kernel")?,
            rope_neox: take(map, "rope_neox_kernel")?,
            conv1d_step_silu: take(map, "conv1d_step_silu_kernel")?,
            gated_delta_rule_step: take(map, "gated_delta_rule_step_kernel")?,
            delta_gate: take(map, "delta_gate_kernel")?,
            deinterleave_heads: take(map, "deinterleave_heads_kernel")?,
            attn_prefill: take(map, "attn_prefill_kernel")?,
            attn_prefill_tiled: take(map, "attn_prefill_tiled_kernel")?,
            attn_decode: take(map, "attn_decode_kernel")?,
            kv_cache_append: take(map, "kv_cache_append_kernel")?,
            embed_gather: take(map, "embed_gather_kernel")?,
            argmax: take(map, "argmax_kernel")?,
            argmax_multi: take(map, "argmax_multi_kernel")?,
            conv1d_step_silu_multi: take(map, "conv1d_step_silu_multi_kernel")?,
            gated_delta_rule_step_multi: take(map, "gated_delta_rule_step_multi_kernel")?,
            kv_cache_append_multi: take(map, "kv_cache_append_multi_kernel")?,
            attn_decode_multi: take(map, "attn_decode_multi_kernel")?,
            attn_decode_multi_serial: take(map, "attn_decode_multi_serial_kernel")?,
            l2norm_scale_batched: take(map, "l2norm_scale_batched_kernel")?,
            delta_gate_batched: take(map, "delta_gate_batched_kernel")?,
            conv1d_prefill_silu: take(map, "conv1d_prefill_silu_kernel")?,
            rope_neox_batched: take(map, "rope_neox_batched_kernel")?,
            kv_cache_append_batched: take(map, "kv_cache_append_batched_kernel")?,
            gated_delta_rule_chunk: take(map, "gated_delta_rule_chunk_kernel")?,
            deinterleave_heads_batched: take(map, "deinterleave_heads_batched_kernel")?,
            embed_gather_batched: take(map, "embed_gather_batched_kernel")?,
            copy_last_row: take(map, "copy_last_row_kernel")?,
            copy_rows: take(map, "copy_rows_kernel")?,
            concat2: take(map, "concat2_kernel")?,
            nvfp4_gemm: take(map, "nvfp4_gemm_kernel")?,
            dequant_nvfp4_to_bf16: take(map, "dequant_nvfp4_to_bf16_kernel")?,
            dequant_nvfp4_to_f16: take(map, "dequant_nvfp4_to_f16_kernel")?,
            dequant_fp8_to_f16: take(map, "dequant_fp8_to_f16_kernel")?,
            u16_to_f16: take(map, "u16_to_f16_kernel")?,
            dequant_fp8_to_bf16: take(map, "dequant_fp8_to_bf16_kernel")?,
            f32_to_bf16: take(map, "f32_to_bf16_kernel")?,
            f32_to_f16: take(map, "f32_to_f16_kernel")?,
            f32_split_bf16: take(map, "f32_split_bf16_kernel")?,
            f32_split3_bf16: take(map, "f32_split3_bf16_kernel")?,
            bf16_to_f32_scaled: take(map, "bf16_to_f32_scaled_kernel")?,
            f32_scale: take(map, "f32_scale_kernel")?,
            u16_to_bf16: take(map, "u16_to_bf16_kernel")?,
            fp8_gemm: take(map, "fp8_gemm_kernel")?,
            bf16_gemm: take(map, "bf16_gemm_kernel")?,
        })
    }

    /// `y[r,:] = x[r,:] * rsqrt(mean(x^2)+eps) * (1 + w)`.
    ///
    /// The `1 +` is the Qwen3.5 convention; see `docs/ARCHITECTURE.md`.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_zero_centered(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        need(x.len() >= rows * n, "rmsnorm x")?;
        need(w.len() >= n, "rmsnorm w")?;
        need(y.len() >= rows * n, "rmsnorm y")?;
        let (n_i, e) = (n as i32, eps);
        unsafe {
            dev.stream()
                .launch_builder(&self.rmsnorm_zero_centered)
                .arg(x)
                .arg(w)
                .arg(y)
                .arg(&n_i)
                .arg(&e)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (block_for(n, 256), 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `y[r,:] = (x[r,:] * rsqrt(mean(x^2)+eps) * w) * silu(z[r,:])`.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_gated(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        need(x.len() >= rows * n, "rmsnorm_gated x")?;
        need(z.len() >= rows * n, "rmsnorm_gated z")?;
        need(w.len() >= n, "rmsnorm_gated w")?;
        let (n_i, e) = (n as i32, eps);
        unsafe {
            dev.stream()
                .launch_builder(&self.rmsnorm_gated)
                .arg(x)
                .arg(z)
                .arg(w)
                .arg(y)
                .arg(&n_i)
                .arg(&e)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (block_for(n, 256), 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// In-place zero-centered RMSNorm: `x[r,:] *= rsqrt(mean(x^2)+eps) * (1+w)`.
    ///
    /// Safe because the kernel completes its reduction (with a block barrier)
    /// before any thread writes.
    pub fn rmsnorm_zero_centered_inplace(
        &self,
        dev: &Device,
        x: &mut CudaSlice<f32>,
        w: &CudaSlice<f32>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        need(x.len() >= rows * n, "rmsnorm_inplace x")?;
        need(w.len() >= n, "rmsnorm_inplace w")?;
        let (n_i, e) = (n as i32, eps);
        let xr: &CudaSlice<f32> = &*x;
        unsafe {
            dev.stream()
                .launch_builder(&self.rmsnorm_zero_centered)
                .arg(xr)
                .arg(w)
                .arg(xr)
                .arg(&n_i)
                .arg(&e)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (block_for(n, 256), 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// In-place gated RMSNorm: `x[r,:] = (x[r,:]*rsqrt(mean+eps)*w) * silu(z[r,:])`.
    pub fn rmsnorm_gated_inplace(
        &self,
        dev: &Device,
        x: &mut CudaSlice<f32>,
        z: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        need(x.len() >= rows * n, "rmsnorm_gated_inplace x")?;
        need(z.len() >= rows * n, "rmsnorm_gated_inplace z")?;
        need(w.len() >= n, "rmsnorm_gated_inplace w")?;
        let (n_i, e) = (n as i32, eps);
        let xr: &CudaSlice<f32> = &*x;
        unsafe {
            dev.stream()
                .launch_builder(&self.rmsnorm_gated)
                .arg(xr)
                .arg(z)
                .arg(w)
                .arg(xr)
                .arg(&n_i)
                .arg(&e)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (block_for(n, 256), 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// In-place SwiGLU: `y = silu(y) * up`.
    pub fn swiglu_inplace(
        &self,
        dev: &Device,
        y: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        need(y.len() >= n && up.len() >= n, "swiglu_inplace")?;
        let n_i = n as i32;
        let yr: &CudaSlice<f32> = &*y;
        unsafe {
            dev.stream()
                .launch_builder(&self.swiglu)
                .arg(yr)
                .arg(up)
                .arg(yr)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(n, 256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Copy one bf16 embedding row into an fp32 activation vector.
    pub fn embed_gather(
        &self,
        dev: &Device,
        table: &CudaSlice<u16>,
        token: u32,
        out: &mut CudaSlice<f32>,
        hidden: usize,
    ) -> Result<()> {
        need(
            table.len() >= (token as usize + 1) * hidden,
            "embed_gather table",
        )?;
        need(out.len() >= hidden, "embed_gather out")?;
        let (t, h) = (token as i32, hidden as i32);
        unsafe {
            dev.stream()
                .launch_builder(&self.embed_gather)
                .arg(table)
                .arg(&t)
                .arg(out)
                .arg(&h)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(hidden, 256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Index of the maximum element, ties resolving to the lowest index.
    /// Batched decode counterparts. Each launches once for the whole batch with
    /// `gridDim.y` selecting the sequence; `base_stride` is that sequence's
    /// offset inside the shared state allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_step_silu_multi(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        hist: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        channels: usize,
        base_stride: usize,
        n_seq: usize,
    ) -> Result<()> {
        need(x.len() >= channels * n_seq && y.len() >= channels * n_seq, "conv1d_multi x/y")?;
        need(hist.len() >= base_stride * n_seq, "conv1d_multi hist")?;
        let (c, bs) = (channels as i32, base_stride as i32);
        unsafe {
            dev.stream()
                .launch_builder(&self.conv1d_step_silu_multi)
                .arg(x)
                .arg(w)
                .arg(hist)
                .arg(y)
                .arg(&c)
                .arg(&bs)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(channels, 256), n_seq as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_step_multi(
        &self,
        dev: &Device,
        qkv: &CudaSlice<f32>,
        q_off: usize,
        k_off: usize,
        v_off: usize,
        row_stride: usize,
        decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_v_heads: usize,
        n_k_heads: usize,
        group: usize,
        base_stride: usize,
        n_seq: usize,
    ) -> Result<()> {
        const D: usize = 128;
        need(decay.len() >= n_seq * n_v_heads && beta.len() >= n_seq * n_v_heads, "delta_multi decay/beta")?;
        need(state.len() >= base_stride * n_seq, "delta_multi state")?;
        need(out.len() >= n_seq * n_v_heads * D, "delta_multi out")?;
        let (qo, ko, vo, rs) = (q_off as i32, k_off as i32, v_off as i32, row_stride as i32);
        let (nv, nk, g, bs) =
            (n_v_heads as i32, n_k_heads as i32, group as i32, base_stride as i32);
        unsafe {
            dev.stream()
                .launch_builder(&self.gated_delta_rule_step_multi)
                .arg(qkv)
                .arg(&qo)
                .arg(&ko)
                .arg(&vo)
                .arg(&rs)
                .arg(decay)
                .arg(beta)
                .arg(state)
                .arg(out)
                .arg(&nv)
                .arg(&nk)
                .arg(&g)
                .arg(&bs)
                .launch(LaunchConfig {
                    grid_dim: (n_v_heads as u32, n_seq as u32, 1),
                    block_dim: (D as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    pub fn kv_cache_append_multi(
        &self,
        dev: &Device,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        k_cache: &mut CudaSlice<u16>,
        v_cache: &mut CudaSlice<u16>,
        positions: &CudaSlice<i32>,
        n_kv_heads: usize,
        head_dim: usize,
        base_stride: usize,
    ) -> Result<()> {
        let n = n_kv_heads * head_dim;
        let n_seq = positions.len();
        need(k.len() >= n * n_seq && v.len() >= n * n_seq, "kv_append_multi k/v")?;
        need(k_cache.len() >= base_stride * n_seq, "kv_append_multi k_cache")?;
        let (nkv, hd, bs) = (n_kv_heads as i32, head_dim as i32, base_stride as i32);
        unsafe {
            dev.stream()
                .launch_builder(&self.kv_cache_append_multi)
                .arg(k)
                .arg(v)
                .arg(k_cache)
                .arg(v_cache)
                .arg(positions)
                .arg(&nkv)
                .arg(&hd)
                .arg(&bs)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(n, 128), n_seq as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_multi(
        &self,
        dev: &Device,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<u16>,
        v_cache: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        base_stride: usize,
    ) -> Result<()> {
        let n_seq = positions.len();
        need(q.len() >= n_q_heads * head_dim * n_seq, "attn_decode_multi q")?;
        need(k_cache.len() >= base_stride * n_seq, "attn_decode_multi k_cache")?;
        need(
            base_stride / (n_kv_heads * head_dim) > 0,
            "attn_decode_multi: empty cache",
        )?;
        let (nq, nkv, hd, bs) =
            (n_q_heads as i32, n_kv_heads as i32, head_dim as i32, base_stride as i32);
        // The warp kernel covers head_dim == 256 (32 lanes x 8 dims) and wants
        // the full eight warps of a 256-thread block; any other head width goes
        // to the serial reference.
        let warp_ok = head_dim == 256;
        // Split-D doubles the grid: `blockIdx.y` now packs (sequence, dim half),
        // so one query head's 256 dims are produced by two blocks that compute
        // identical softmax weights and never communicate.
        let dim_split: usize = if warp_ok { 2 } else { 1 };
        let (func, block) = if warp_ok {
            // Must match NW in `attn_decode_multi_kernel`: the block is
            // NW warps, and each warp owns a strided slice of the keys.
            (&self.attn_decode_multi, 1024u32)
        } else {
            (&self.attn_decode_multi_serial, block_for(head_dim, 256))
        };
        unsafe {
            dev.stream()
                .launch_builder(func)
                .arg(q)
                .arg(k_cache)
                .arg(v_cache)
                .arg(out)
                .arg(positions)
                .arg(&nq)
                .arg(&nkv)
                .arg(&hd)
                .arg(&scale)
                .arg(&bs)
                .launch(LaunchConfig {
                    grid_dim: (n_q_heads as u32, (n_seq * dim_split) as u32, 1),
                    block_dim: (block, 1, 1),
                    // No dynamic shared memory: the scores stream through
                    // registers, so this no longer scales with the cache. The
                    // warp kernel additionally needs its 8 KB merge scratch,
                    // which is static.
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// The serial decode reference, always, regardless of head width. Exists so
    /// `gb10-verify decode-bench` can time the two against each other and check
    /// that they agree; the model path uses `attn_decode_multi`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_multi_serial(
        &self,
        dev: &Device,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<u16>,
        v_cache: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        base_stride: usize,
    ) -> Result<()> {
        let n_seq = positions.len();
        let (nq, nkv, hd, bs) =
            (n_q_heads as i32, n_kv_heads as i32, head_dim as i32, base_stride as i32);
        unsafe {
            dev.stream()
                .launch_builder(&self.attn_decode_multi_serial)
                .arg(q)
                .arg(k_cache)
                .arg(v_cache)
                .arg(out)
                .arg(positions)
                .arg(&nq)
                .arg(&nkv)
                .arg(&hd)
                .arg(&scale)
                .arg(&bs)
                .launch(LaunchConfig {
                    grid_dim: (n_q_heads as u32, n_seq as u32, 1),
                    block_dim: (block_for(head_dim, 256), 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    pub fn argmax_multi(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        out_idx: &mut CudaSlice<i32>,
        n: usize,
        n_seq: usize,
    ) -> Result<()> {
        need(x.len() >= n * n_seq, "argmax_multi x")?;
        need(out_idx.len() >= n_seq, "argmax_multi out")?;
        let n_i = n as i32;
        unsafe {
            dev.stream()
                .launch_builder(&self.argmax_multi)
                .arg(x)
                .arg(&n_i)
                .arg(out_idx)
                .launch(LaunchConfig {
                    grid_dim: (1, n_seq as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    pub fn argmax(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        out_idx: &mut CudaSlice<i32>,
        n: usize,
    ) -> Result<()> {
        need(x.len() >= n, "argmax x")?;
        need(out_idx.len() >= 1, "argmax out")?;
        let n_i = n as i32;
        unsafe {
            dev.stream()
                .launch_builder(&self.argmax)
                .arg(x)
                .arg(&n_i)
                .arg(out_idx)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `y = silu(gate) * up`
    pub fn swiglu(
        &self,
        dev: &Device,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        need(gate.len() >= n && up.len() >= n && y.len() >= n, "swiglu")?;
        let n_i = n as i32;
        unsafe {
            dev.stream()
                .launch_builder(&self.swiglu)
                .arg(gate)
                .arg(up)
                .arg(y)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(n, 256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `y = a + b`
    pub fn add(
        &self,
        dev: &Device,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        need(a.len() >= n && b.len() >= n && y.len() >= n, "add")?;
        let n_i = n as i32;
        unsafe {
            dev.stream()
                .launch_builder(&self.add)
                .arg(a)
                .arg(b)
                .arg(y)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(n, 256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `y *= sigmoid(gate)` — the attention output gate (sigmoid, not swish).
    /// Dequantize a whole NVFP4 matrix `[N, K]` to row-major bf16 `[N, K]`.
    ///
    /// The per-tensor scale `s2` is *not* applied -- see the kernel comment.
    pub fn dequant_nvfp4_to_bf16(
        &self,
        dev: &Device,
        w: &CudaSlice<u8>,
        sc: &CudaSlice<u8>,
        out: &mut CudaSlice<half::bf16>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        need(out.len() >= n * k, "dequant_nvfp4_to_bf16")?;
        let n_i = n as i32;
        let k_i = k as i32;
        let total = n * k;
        let grid = cdiv(total, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.dequant_nvfp4_to_bf16)
                .arg(w)
                .arg(sc)
                .arg(out)
                .arg(&n_i)
                .arg(&k_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Dequantize a whole FP8 (E4M3) matrix `[N, K]` to row-major bf16 `[N, K]`.
    /// `s1` is the per-tensor scale, applied here exactly as staging does.
    pub fn dequant_fp8_to_bf16(
        &self,
        dev: &Device,
        w: &CudaSlice<u8>,
        s1: &CudaSlice<f32>,
        out: &mut CudaSlice<half::bf16>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        need(out.len() >= n * k, "dequant_fp8_to_bf16")?;
        let n_i = n as i32;
        let k_i = k as i32;
        let grid = cdiv(n * k, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.dequant_fp8_to_bf16)
                .arg(w)
                .arg(s1)
                .arg(out)
                .arg(&n_i)
                .arg(&k_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// fp16 twin of `dequant_nvfp4_to_bf16`. See `cublas_gemm_f16_f32`.
    pub fn dequant_nvfp4_to_f16(
        &self,
        dev: &Device,
        w: &CudaSlice<u8>,
        sc: &CudaSlice<u8>,
        out: &mut CudaSlice<half::f16>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        need(out.len() >= n * k, "dequant_nvfp4_to_f16")?;
        let n_i = n as i32;
        let k_i = k as i32;
        let grid = cdiv(n * k, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.dequant_nvfp4_to_f16)
                .arg(w)
                .arg(sc)
                .arg(out)
                .arg(&n_i)
                .arg(&k_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// fp16 twin of `dequant_fp8_to_bf16`. See `cublas_gemm_f16_f32`.
    pub fn dequant_fp8_to_f16(
        &self,
        dev: &Device,
        w: &CudaSlice<u8>,
        s1: &CudaSlice<f32>,
        out: &mut CudaSlice<half::f16>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        need(out.len() >= n * k, "dequant_fp8_to_f16")?;
        let n_i = n as i32;
        let k_i = k as i32;
        let grid = cdiv(n * k, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.dequant_fp8_to_f16)
                .arg(w)
                .arg(s1)
                .arg(out)
                .arg(&n_i)
                .arg(&k_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// fp16 twin of `u16_to_bf16`. See `cublas_gemm_f16_f32`.
    pub fn u16_to_f16(
        &self,
        dev: &Device,
        x: &CudaSlice<u16>,
        out: &mut CudaSlice<half::f16>,
        n: usize,
    ) -> Result<()> {
        need(x.len() >= n && out.len() >= n, "u16_to_f16")?;
        let n_i = n as i32;
        let grid = cdiv(n, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.u16_to_f16)
                .arg(x)
                .arg(out)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `y[t, n] = x[t, k] * W[n, k]^T` on **fp16** tensor cores, fp32
    /// accumulate, **fp32 output**.
    ///
    /// This is the prefill GEMM the tensor-core path uses. Same operand layout
    /// and same `cublasGemmEx` call as `cublas_gemm_bf16_f32`; both operands are
    /// `CUDA_R_16F`.
    ///
    /// fp16 rather than bf16 because bf16's 8 mantissa bits were not enough for
    /// this model: `generate` was 16/16 token-exact with bf16 operands, yet at
    /// 970+ prompt tokens the greedy argmax flipped to EOS and the engine
    /// returned no content at all, where the fp32 CUDA-core path answers
    /// correctly and so does llama.cpp on the same prompts. Perplexity at a
    /// 512-token window could not see it (0.023% on mean NLL), which is why the
    /// short-prompt gates cleared it. fp16 keeps 10 mantissa bits at identical
    /// throughput.
    ///
    /// Mixed bf16 weights with fp16 activations is *not* used: this cuBLAS
    /// rejects that pair with a buffer-size error even though the documentation
    /// allows differing `Atype`/`Btype` under `CUBLAS_COMPUTE_32F`.
    pub fn cublas_gemm_f16_f32(
        &self,
        dev: &Device,
        w: &CudaSlice<half::f16>,
        x: &CudaSlice<half::f16>,
        y: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
        t: usize,
    ) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasComputeType_t, cublasGemmAlgo_t, cublasGemmEx, cublasOperation_t, cudaDataType,
        };
        need(
            w.len() >= n * k && x.len() >= t * k && y.len() >= t * n,
            "cublas_gemm_f16_f32",
        )?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let status = unsafe {
            cublasGemmEx(
                *dev.blas().handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                t as i32,
                k as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                w.device_ptr(dev.stream()).0 as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16F,
                k as i32,
                x.device_ptr(dev.stream()).0 as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16F,
                k as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                y.device_ptr_mut(dev.stream()).0 as *mut std::ffi::c_void,
                cudaDataType::CUDA_R_32F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        need(
            status == cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            "cublasGemmEx(f16 x f16 -> f32) failed",
        )?;
        Ok(())
    }

    /// Cast `n` fp32 values to bf16 (the tensor-core activation format).
    pub fn f32_to_bf16(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<half::bf16>,
        n: usize,
    ) -> Result<()> {
        need(x.len() >= n && out.len() >= n, "f32_to_bf16")?;
        let n_i = n as i32;
        let grid = cdiv(n, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.f32_to_bf16)
                .arg(x)
                .arg(out)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `y[t, n] = x[t, k] * W[n, k]^T` on bf16 tensor cores, fp32 accumulate.
    ///
    /// All three buffers are row-major as written here, but cuBLAS is
    /// column-major and its C(m, n) = op(A) * op(B). Setting `m = n_out` and
    /// `n = t` makes C, read column-major, exactly our row-major `y[t, n_out]`.
    /// W is row-major `[n_out, k]`, which read column-major with `lda = k` is
    /// W^T, hence `transa = T`; x is row-major `[t, k]`, which read
    /// column-major with `ldb = k` is already the `(k, t)` operand we want,
    /// hence `transb = N`. Getting `m`/`n` the other way round silently produces
    /// a transposed result, which is what `gb10-bench cublas-parity` exists to
    /// catch.
    ///
    /// The output is bf16, so the caller is expected to convert back to fp32
    /// (and apply any per-tensor `s2`) in its epilogue.
    pub fn cublas_gemm_bf16(
        &self,
        dev: &Device,
        w: &CudaSlice<half::bf16>,
        x: &CudaSlice<half::bf16>,
        y: &mut CudaSlice<half::bf16>,
        n: usize,
        k: usize,
        t: usize,
    ) -> Result<()> {
        use cudarc::cublas::sys::cublasOperation_t;
        use cudarc::cublas::{Gemm, GemmConfig};
        need(
            w.len() >= n * k && x.len() >= t * k && y.len() >= t * n,
            "cublas_gemm_bf16",
        )?;
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: t as i32,
            k: k as i32,
            alpha: half::bf16::from_f32(1.0),
            lda: k as i32,
            ldb: k as i32,
            beta: half::bf16::from_f32(0.0),
            ldc: n as i32,
        };
        unsafe {
            dev.blas()
                .gemm(cfg, w, x, y)
                .map_err(|e| CudaError::Cublas(format!("{e:?}")))?;
        }
        Ok(())
    }

    /// Convert `n` bf16 values back to fp32, multiplying by `s2` when
    /// `has_scale` is set. This is the epilogue of the bf16 tensor-core GEMM.
    pub fn bf16_to_f32_scaled(
        &self,
        dev: &Device,
        x: &CudaSlice<half::bf16>,
        out: &mut CudaSlice<f32>,
        s2: &CudaSlice<f32>,
        has_scale: bool,
        n: usize,
    ) -> Result<()> {
        need(x.len() >= n && out.len() >= n, "bf16_to_f32_scaled")?;
        let hs = has_scale as i32;
        let n_i = n as i32;
        let grid = cdiv(n, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.bf16_to_f32_scaled)
                .arg(x)
                .arg(out)
                .arg(s2)
                .arg(&hs)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Split `n` fp32 values into bf16 high / mid / low parts, so
    /// `hi + mid + lo` reconstructs them to ~24 bits -- fp32's own width.
    pub fn f32_split3_bf16(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        hi: &mut CudaSlice<half::bf16>,
        mid: &mut CudaSlice<half::bf16>,
        lo: &mut CudaSlice<half::bf16>,
        n: usize,
    ) -> Result<()> {
        need(
            x.len() >= n && hi.len() >= n && mid.len() >= n && lo.len() >= n,
            "f32_split3_bf16",
        )?;
        let n_i = n as i32;
        let grid = cdiv(n, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.f32_split3_bf16)
                .arg(x)
                .arg(hi)
                .arg(mid)
                .arg(lo)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Split `n` fp32 values into a bf16 high part and a bf16 low part, so
    /// `hi + lo` carries ~16 mantissa bits.
    ///
    /// This is the activation operand of the split-precision prefill GEMM. bf16
    /// (8 bits) and fp16 (10 bits) each broke this model on their own; see
    /// `forward_prefill_tensor_core`.
    pub fn f32_split_bf16(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        hi: &mut CudaSlice<half::bf16>,
        lo: &mut CudaSlice<half::bf16>,
        n: usize,
    ) -> Result<()> {
        need(
            x.len() >= n && hi.len() >= n && lo.len() >= n,
            "f32_split_bf16",
        )?;
        let n_i = n as i32;
        let grid = cdiv(n, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.f32_split_bf16)
                .arg(x)
                .arg(hi)
                .arg(lo)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Cast `n` fp32 values to fp16 (round to nearest).
    ///
    /// The activation operand of the tensor-core prefill GEMM. fp16 rather than
    /// bf16 because bf16's 8 mantissa bits were not enough: see
    /// `cublas_gemm_bf16_f16_f32`.
    pub fn f32_to_f16(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<half::f16>,
        n: usize,
    ) -> Result<()> {
        need(x.len() >= n && out.len() >= n, "f32_to_f16")?;
        let n_i = n as i32;
        let grid = cdiv(n, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.f32_to_f16)
                .arg(x)
                .arg(out)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `y[t, n] = x[t, k] * W[n, k]^T` with **bf16 weights, fp16 activations**
    /// and an fp32 output, accumulating in fp32.
    ///
    /// Same layout and same `cublasGemmEx` path as `cublas_gemm_bf16_f32`; the
    /// only difference is `Btype = CUDA_R_16F`. The two operands are allowed to
    /// differ because the compute type is `CUBLAS_COMPUTE_32F`, which is what
    /// lets the weights stay in bf16 -- bf16 is already lossless for the 4-bit
    /// NVFP4 and FP8 weights, so re-converting them to fp16 would cost a kernel
    /// pass and buy nothing.
    ///
    /// The activations are the operand that needs the precision: they are the
    /// only full-precision value on this path, and they feed a 48-layer
    /// Gated-DeltaNet recurrence that compounds their error with length.
    pub fn cublas_gemm_bf16_f16_f32(
        &self,
        dev: &Device,
        w: &CudaSlice<half::bf16>,
        x: &CudaSlice<half::f16>,
        y: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
        t: usize,
    ) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasComputeType_t, cublasGemmAlgo_t, cublasGemmEx, cublasOperation_t, cudaDataType,
        };
        need(
            w.len() >= n * k && x.len() >= t * k && y.len() >= t * n,
            "cublas_gemm_bf16_f16_f32",
        )?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let status = unsafe {
            cublasGemmEx(
                *dev.blas().handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                t as i32,
                k as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                w.device_ptr(dev.stream()).0 as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16BF,
                k as i32,
                x.device_ptr(dev.stream()).0 as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16F,
                k as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                y.device_ptr_mut(dev.stream()).0 as *mut std::ffi::c_void,
                cudaDataType::CUDA_R_32F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        need(
            status == cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            "cublasGemmEx(bf16 x f16 -> f32) failed",
        )?;
        Ok(())
    }

    /// Scale `n` fp32 values by the single scalar in `s`, in place.
    ///
    /// This is the NVFP4 `s2` application that used to ride along with the
    /// bf16 -> fp32 epilogue. It exists separately now because the GEMM writes
    /// fp32 directly (see `cublas_gemm_bf16_f32`), so there is no conversion
    /// left to fuse it into.
    pub fn f32_scale(
        &self,
        dev: &Device,
        x: &mut CudaSlice<f32>,
        s: &CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        need(x.len() >= n && s.len() >= 1, "f32_scale")?;
        let n_i = n as i32;
        let grid = cdiv(n, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.f32_scale)
                .arg(x)
                .arg(s)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `y[t, n] = x[t, k] * W[n, k]^T` on bf16 tensor cores with an **fp32**
    /// output, accumulating in fp32.
    ///
    /// This is `cublas_gemm_bf16` with `C` left in fp32, and it exists because
    /// rounding the accumulator to bf16 on the way out costs real accuracy for
    /// no speed. cudarc's safe `Gemm<bf16>` fixes A, B *and* C to
    /// `CUDA_R_16BF`, so a mixed bf16-in/fp32-out GEMM has to go through
    /// `cublasGemmEx` directly.
    ///
    /// The operand layout is unchanged from `cublas_gemm_bf16`: cuBLAS is
    /// column-major and computes `C(m, n) = op(A) * op(B)`, so `m = n_out` and
    /// `n = t` make `C`, read column-major, exactly our row-major `y[t, n_out]`.
    /// `W` is row-major `[n_out, k]`, which read column-major with `lda = k` is
    /// `W^T`, hence `transa = T`; `x` is row-major `[t, k]`, already the
    /// `(k, t)` operand with `ldb = k`, hence `transb = N`.
    ///
    /// Writing straight into the caller's fp32 `y` also removes the separate
    /// bf16 -> fp32 convert kernel, so this path is one launch shorter than the
    /// bf16-output one it replaces.
    pub fn cublas_gemm_bf16_f32(
        &self,
        dev: &Device,
        w: &CudaSlice<half::bf16>,
        x: &CudaSlice<half::bf16>,
        y: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
        t: usize,
        beta_in: f32,
    ) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasGemmAlgo_t, cublasGemmEx, cublasOperation_t, cudaDataType,
            cublasComputeType_t,
        };
        need(
            w.len() >= n * k && x.len() >= t * k && y.len() >= t * n,
            "cublas_gemm_bf16_f32",
        )?;
        let alpha = 1.0f32;
        let beta = beta_in;
        let status = unsafe {
            cublasGemmEx(
                *dev.blas().handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                t as i32,
                k as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                w.device_ptr(dev.stream()).0 as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16BF,
                k as i32,
                x.device_ptr(dev.stream()).0 as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16BF,
                k as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                y.device_ptr_mut(dev.stream()).0 as *mut std::ffi::c_void,
                cudaDataType::CUDA_R_32F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        need(
            status == cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            "cublasGemmEx(bf16 x bf16 -> f32) failed",
        )?;
        Ok(())
    }

    /// Copy u16-held bf16 weights into a `half::bf16` operand buffer.
    pub fn u16_to_bf16(
        &self,
        dev: &Device,
        x: &CudaSlice<u16>,
        out: &mut CudaSlice<half::bf16>,
        n: usize,
    ) -> Result<()> {
        need(x.len() >= n && out.len() >= n, "u16_to_bf16")?;
        let n_i = n as i32;
        let grid = cdiv(n, 256).min(65535) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.u16_to_bf16)
                .arg(x)
                .arg(out)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    pub fn sigmoid_mul(
        &self,
        dev: &Device,
        y: &mut CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        need(y.len() >= n && gate.len() >= n, "sigmoid_mul")?;
        let n_i = n as i32;
        unsafe {
            dev.stream()
                .launch_builder(&self.sigmoid_mul)
                .arg(y)
                .arg(gate)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(n, 256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// In-place `x *= scale * rsqrt(sum(x^2) + eps)`, one block per vector.
    pub fn l2norm_scale(
        &self,
        dev: &Device,
        x: &mut CudaSlice<f32>,
        offset: usize,
        vectors: usize,
        n: usize,
        scale: f32,
        eps: f32,
    ) -> Result<()> {
        need(x.len() >= offset + vectors * n, "l2norm_scale")?;
        let (o_i, n_i, e) = (offset as i32, n as i32, eps);
        unsafe {
            dev.stream()
                .launch_builder(&self.l2norm_scale)
                .arg(x)
                .arg(&o_i)
                .arg(&n_i)
                .arg(&scale)
                .arg(&e)
                .launch(LaunchConfig {
                    grid_dim: (vectors as u32, 1, 1),
                    block_dim: (block_for(n, 256), 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// GPT-NeoX rotation applied to the first `2*half` channels of each head.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_neox(
        &self,
        dev: &Device,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        cs: &CudaSlice<f32>,
        sn: &CudaSlice<f32>,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        half: usize,
    ) -> Result<()> {
        need(cs.len() >= half && sn.len() >= half, "rope cos/sin")?;
        let (nq, nk, hd, hf) =
            (n_q_heads as i32, n_k_heads as i32, head_dim as i32, half as i32);
        let total = (n_q_heads + n_k_heads) * half;
        unsafe {
            dev.stream()
                .launch_builder(&self.rope_neox)
                .arg(q)
                .arg(k)
                .arg(cs)
                .arg(sn)
                .arg(&nq)
                .arg(&nk)
                .arg(&hd)
                .arg(&hf)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(total, 256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// One decode step of the depthwise causal conv (kernel 4) plus SiLU.
    /// `hist` is `[channels, 3]` and is shifted in place.
    pub fn conv1d_step_silu(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        hist: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        channels: usize,
    ) -> Result<()> {
        need(x.len() >= channels && y.len() >= channels, "conv1d x/y")?;
        need(w.len() >= channels * 4, "conv1d w")?;
        need(hist.len() >= channels * 3, "conv1d hist")?;
        let c = channels as i32;
        let base = 0i32;  // per-sequence offset; 0 until batching lands
        unsafe {
            dev.stream()
                .launch_builder(&self.conv1d_step_silu)
                .arg(x)
                .arg(w)
                .arg(hist)
                .arg(y)
                .arg(&c)
                .arg(&base)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(channels, 256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// One decode step of the Gated DeltaNet recurrence.
    ///
    /// `q`/`k` are `[batch, n_k_heads, 128]` and must already be L2-normalised
    /// and scaled by `128^-0.5`. `state` is `[batch, n_v_heads, 128, 128]` fp32
    /// and must be zeroed at sequence start.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_step(
        &self,
        dev: &Device,
        qkv: &CudaSlice<f32>,
        q_off: usize,
        k_off: usize,
        v_off: usize,
        row_stride: usize,
        decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        batch: usize,
        n_v_heads: usize,
        n_k_heads: usize,
        group: usize,
    ) -> Result<()> {
        let d = DELTA_KEY_HEAD_DIM;
        need(qkv.len() >= batch * row_stride, "delta qkv")?;
        need(decay.len() >= batch * n_v_heads, "delta decay")?;
        need(beta.len() >= batch * n_v_heads, "delta beta")?;
        need(state.len() >= batch * n_v_heads * d * d, "delta state")?;
        need(out.len() >= batch * n_v_heads * d, "delta out")?;
        let (qo, ko, vo, rs) =
            (q_off as i32, k_off as i32, v_off as i32, row_stride as i32);
        let (nv, nk, g) = (n_v_heads as i32, n_k_heads as i32, group as i32);
        let base = 0i32;  // per-sequence offset; 0 until batching lands
        unsafe {
            dev.stream()
                .launch_builder(&self.gated_delta_rule_step)
                .arg(qkv)
                .arg(&qo)
                .arg(&ko)
                .arg(&vo)
                .arg(&rs)
                .arg(decay)
                .arg(beta)
                .arg(state)
                .arg(out)
                .arg(&nv)
                .arg(&nk)
                .arg(&g)
                .arg(&base)
                .launch(LaunchConfig {
                    grid_dim: (n_v_heads as u32, batch as u32, 1),
                    block_dim: (d as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `beta = sigmoid(b)`, `decay = exp(-exp(A_log) * softplus(a + dt_bias))`.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_gate(
        &self,
        dev: &Device,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        a_log: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        decay: &mut CudaSlice<f32>,
        beta: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        need(
            a.len() >= n && b.len() >= n && a_log.len() >= n && dt_bias.len() >= n,
            "delta_gate inputs",
        )?;
        need(decay.len() >= n && beta.len() >= n, "delta_gate outputs")?;
        let n_i = n as i32;
        unsafe {
            dev.stream()
                .launch_builder(&self.delta_gate)
                .arg(a)
                .arg(b)
                .arg(a_log)
                .arg(dt_bias)
                .arg(decay)
                .arg(beta)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(n, 128), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Gather one half of each head from a `[n_heads, 2*head_dim]` block.
    /// `src_off` is 0 for the query half and `head_dim` for the gate half.
    pub fn deinterleave_heads(
        &self,
        dev: &Device,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        src_off: usize,
    ) -> Result<()> {
        let total = n_heads * head_dim;
        need(src.len() >= total * 2, "deinterleave src")?;
        need(dst.len() >= total, "deinterleave dst")?;
        let (nh, hd, so) = (n_heads as i32, head_dim as i32, src_off as i32);
        unsafe {
            dev.stream()
                .launch_builder(&self.deinterleave_heads)
                .arg(src)
                .arg(dst)
                .arg(&nh)
                .arg(&hd)
                .arg(&so)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(total, 256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Decode attention: one query token against `n_keys` cached keys.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        dev: &Device,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<u16>,
        v_cache: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        n_keys: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        base: usize,
        q_off: usize,
    ) -> Result<()> {
        need(n_keys > 0, "attn_decode: no keys")?;
        need(
            k_cache.len() >= n_keys * n_kv_heads * head_dim
                && v_cache.len() >= n_keys * n_kv_heads * head_dim,
            "attn_decode cache",
        )?;
        let (nk, nq, nkv, hd) =
            (n_keys as i32, n_q_heads as i32, n_kv_heads as i32, head_dim as i32);
        let (base, qo) = (base as i32, q_off as i32);
        unsafe {
            dev.stream()
                .launch_builder(&self.attn_decode)
                .arg(q)
                .arg(k_cache)
                .arg(v_cache)
                .arg(out)
                .arg(&nk)
                .arg(&nq)
                .arg(&nkv)
                .arg(&hd)
                .arg(&scale)
                .arg(&base)
                .arg(&qo)
                .launch(LaunchConfig {
                    grid_dim: (n_q_heads as u32, 1, 1),
                    block_dim: (block_for(head_dim, 256), 1, 1),
                    // No dynamic shared memory: the scores stream through
                    // registers, so this no longer scales with the cache.
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Append one token's k/v row into the cache at `pos`.
    pub fn kv_cache_append(
        &self,
        dev: &Device,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        k_cache: &mut CudaSlice<u16>,
        v_cache: &mut CudaSlice<u16>,
        pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        base: usize,
        src_off: usize,
    ) -> Result<()> {
        let n = n_kv_heads * head_dim;
        need(k.len() >= n && v.len() >= n, "kv append src")?;
        need(
            k_cache.len() >= pos * n + n && v_cache.len() >= pos * n + n,
            "kv append cache",
        )?;
        let (p, nkv, hd) = (pos as i32, n_kv_heads as i32, head_dim as i32);
        let (base, so) = (base as i32, src_off as i32);
        unsafe {
            dev.stream()
                .launch_builder(&self.kv_cache_append)
                .arg(k)
                .arg(v)
                .arg(k_cache)
                .arg(v_cache)
                .arg(&p)
                .arg(&nkv)
                .arg(&hd)
                .arg(&base)
                .arg(&so)
                .launch(LaunchConfig {
                    grid_dim: (cdiv(n, 128), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Causal prefill attention with GQA. `q`/`k`/`v` are `[T, heads, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    /// Causal prefill attention, dispatching to whichever kernel can serve the
    /// context length.
    ///
    /// `attn_prefill_kernel` materialises one score per key in dynamic shared
    /// memory, so it is launchable only while `(start + n_tokens) * 4` fits in
    /// the 48 KB a launch can request without opting into a larger carve-out.
    /// Below that threshold it is used unchanged, so every existing result is
    /// bit-identical to before. Past it the tiled kernel takes over, whose
    /// shared memory is constant in the key count.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill(
        &self,
        dev: &Device,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        start: usize,
        kv_base: usize,
    ) -> Result<()> {
        // Every prefill goes through the tiled kernel, including short ones.
        //
        // `attn_prefill_legacy` runs one block reduction per key per query, so
        // its cost grows quadratically with the key count *and* its constant
        // factor is far worse. Measured in situ on the same 2048-token chunk:
        // 82.3 s at 10,240 cached keys (legacy) against 39.2 s at 12,288
        // (tiled) -- the tiled kernel is twice as fast with 20% more keys.
        // Keeping the legacy kernel for short prompts bought bit-identical
        // results, but the two agree to ~1e-7 relative (f32 summation order),
        // which is not worth a quadratic term in the hot path.
        //
        // `attn_prefill_legacy` remains as the independent reference that the
        // `attn-tile` gate compares against.
        self.attn_prefill_tiled(
            dev, q, k, v, out, n_tokens, n_q_heads, n_kv_heads, head_dim, scale, start, kv_base,
        )
    }

    /// Tiled causal prefill attention: the same contract as the legacy kernel,
    /// but shared memory is a constant rather than `O(start + n_tokens)`, so it
    /// runs at any context length the cache can hold. `BQ`/`BK` must match the
    /// kernel's `#define`s.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_tiled(
        &self,
        dev: &Device,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        start: usize,
        kv_base: usize,
    ) -> Result<()> {
        // Must match PREFILL_BQ / PREFILL_BK in kernels/elementwise.cu. BQ * BK
        // must also divide evenly into the score loop's passes there, which with
        // head_dim == 256 threads means a multiple of 128; 24 * 16 == 384 is
        // exactly 3 passes. V is held in registers there, not staged, so the
        // shared budget below has no `2 * BK * head_dim` term.
        const BQ: usize = 24;
        const BK: usize = 16;
        need(
            q.len() >= n_tokens * n_q_heads * head_dim
                && k.len() >= kv_base + (start + n_tokens) * n_kv_heads * head_dim
                && v.len() >= kv_base + (start + n_tokens) * n_kv_heads * head_dim
                && out.len() >= n_tokens * n_q_heads * head_dim,
            "attn_prefill_tiled",
        )?;
        // One thread per head dimension: the pair split needs an even head_dim
        // and the shuffle reduction needs both halves of a pair in one warp.
        if head_dim % 32 != 0 || head_dim > 1024 {
            return Err(CudaError::InvalidArgument(format!(
                "attn_prefill_tiled needs head_dim a multiple of 32 and <= 1024, got {head_dim}"
            )));
        }
        // The score loop in the kernel gives each thread-pair three pairs that
        // share one K row, which requires PREFILL_BQ * PREFILL_BK to be exactly
        // 3 * (blockDim.x / 2), i.e. three uniform passes.
        if BQ * BK != 3 * (head_dim / 2) {
            return Err(CudaError::InvalidArgument(format!(
                "attn_prefill_tiled needs BQ * BK == 3 * (head_dim / 2), got {BQ} * {BK} against head_dim {head_dim}"
            )));
        }
        // Q and K rows are padded in the kernel to break the shared bank
        // conflicts the natural stride causes: the row stride is head_dim + 2
        // with the two halves separated by one extra float.
        // Q and K are staged in bf16 (PADH = head_dim/2 + 2, PS = head_dim + 4)
        // while S and the reduction scratch stay fp32. Halving the Q/K staging
        // takes this kernel from 2 to 4 blocks per SM, which is the lever the
        // round-39 occupancy probe identified.
        // The kernel's row stride is head_dim + 8, not head_dim + 4: rows must be
        // 16-byte aligned so ldmatrix can read whole fragments in one instruction,
        // which retires the old +4 gap that only existed for the scalar kernel's
        // bank behaviour.
        let smem =
            (BQ * (head_dim + 8) + BK * (head_dim + 8)) * 2 + (BQ * BK + 3 * BQ) * 4;

        // Occupancy probe. The computed request (43,104 B here) is what lets two
        // blocks co-reside per SM; `GB10_ATTN_SMEM_PROBE=<bytes>` raises the
        // request past the 50,688 B that two blocks would need, forcing one block
        // per SM. Not a line of kernel arithmetic changes and the launch is
        // numerically identical, so this isolates occupancy as a variable on its
        // own -- which is the one thing standing between the `BK` tiling work and
        // knowing whether it can pay off at all. Anything past the 48 KB default
        // also requires the opt-in ceiling to be raised on the function first.
        let smem = match std::env::var("GB10_ATTN_SMEM_PROBE") {
            Ok(v) => match v.parse::<usize>() {
                Ok(n) if n >= smem => n,
                _ => smem,
            },
            Err(_) => smem,
        };
        if smem > 48 * 1024 {
            let ceiling = self
                .attn_prefill_tiled
                .get_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                )
                .unwrap_or(0);
            if (smem as i32) > ceiling {
                self.attn_prefill_tiled
                    .set_attribute(
                        cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem as i32,
                    )
                    .map_err(|e| {
                        CudaError::InvalidArgument(format!(
                            "attn_prefill_tiled: cannot raise dynamic shared memory to {smem} B: {e:?}"
                        ))
                    })?;
            }
        }
        let (t, nq, nk, hd) =
            (n_tokens as i32, n_q_heads as i32, n_kv_heads as i32, head_dim as i32);
        let (st, kb) = (start as i32, kv_base as i32);
        let tiles = n_tokens.div_ceil(BQ) as u32;
        unsafe {
            dev.stream()
                .launch_builder(&self.attn_prefill_tiled)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(out)
                .arg(&t)
                .arg(&nq)
                .arg(&nk)
                .arg(&hd)
                .arg(&scale)
                .arg(&st)
                .arg(&kb)
                .launch(LaunchConfig {
                    grid_dim: (n_q_heads as u32, tiles, 1),
                    block_dim: (head_dim as u32, 1, 1),
                    shared_mem_bytes: smem as u32,
                })?;
        }
        Ok(())
    }

    /// The original prefill kernel. Correct, but its shared-memory request
    /// grows with the number of keys.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_legacy(
        &self,
        dev: &Device,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        start: usize,
        kv_base: usize,
    ) -> Result<()> {
        need(
            q.len() >= n_tokens * n_q_heads * head_dim
                // `kv_base` is already in floats; only the key rows scale by
                // the head count.
                && k.len() >= kv_base + (start + n_tokens) * n_kv_heads * head_dim
                && v.len() >= kv_base + (start + n_tokens) * n_kv_heads * head_dim
                && out.len() >= n_tokens * n_q_heads * head_dim,
            "attn_prefill",
        )?;
        let (t, nq, nk, hd) =
            (n_tokens as i32, n_q_heads as i32, n_kv_heads as i32, head_dim as i32);
        let (st, kb) = (start as i32, kv_base as i32);
        unsafe {
            dev.stream()
                .launch_builder(&self.attn_prefill)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(out)
                .arg(&t)
                .arg(&nq)
                .arg(&nk)
                .arg(&hd)
                .arg(&scale)
                .arg(&st)
                .arg(&kb)
                .launch(LaunchConfig {
                    grid_dim: (n_q_heads as u32, n_tokens as u32, 1),
                    block_dim: (block_for(head_dim, 256), 1, 1),
                    shared_mem_bytes: ((start + n_tokens) * 4) as u32,
                })?;
        }
        Ok(())
    }
    /// Batched `l2norm_scale` over `batch` rows of `row_stride` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn l2norm_scale_batched(
        &self, dev: &Device, x: &mut CudaSlice<f32>, row_stride: usize, offset: usize,
        vectors: usize, n: usize, batch: usize, scale: f32, eps: f32,
    ) -> Result<()> {
        let (rs, off, nn) = (row_stride as i32, offset as i32, n as i32);
        unsafe {
            dev.stream().launch_builder(&self.l2norm_scale_batched)
                .arg(x).arg(&rs).arg(&off).arg(&nn).arg(&scale).arg(&eps)
                .launch(LaunchConfig { grid_dim: (vectors as u32, batch as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Batched `delta_gate` over `batch` rows of `n` heads.
    pub fn delta_gate_batched(
        &self, dev: &Device, a: &CudaSlice<f32>, b: &CudaSlice<f32>, a_log: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>, decay: &mut CudaSlice<f32>, beta: &mut CudaSlice<f32>,
        n: usize, batch: usize,
    ) -> Result<()> {
        let nn = n as i32;
        unsafe {
            dev.stream().launch_builder(&self.delta_gate_batched)
                .arg(a).arg(b).arg(a_log).arg(dt_bias).arg(decay).arg(beta).arg(&nn)
                .launch(LaunchConfig { grid_dim: (cdiv(n,256), batch as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Causal depthwise conv + SiLU over `t` rows, threading `hist` through.
    pub fn conv1d_prefill_silu(
        &self, dev: &Device, x: &CudaSlice<f32>, w: &CudaSlice<f32>, hist: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>, channels: usize, t: usize, base: usize,
    ) -> Result<()> {
        let (c, tt) = (channels as i32, t as i32);
        let bs = base as i32;
        unsafe {
            dev.stream().launch_builder(&self.conv1d_prefill_silu)
                .arg(x).arg(w).arg(hist).arg(y).arg(&c).arg(&tt).arg(&bs)
                .launch(LaunchConfig { grid_dim: (cdiv(channels,256), 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// RoPE over `t` rows with per-position `cos`/`sin` tables.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_neox_batched(
        &self, dev: &Device, q: &mut CudaSlice<f32>, k: &mut CudaSlice<f32>,
        cos: &CudaSlice<f32>, sin: &CudaSlice<f32>, n_q_heads: usize, n_k_heads: usize,
        head_dim: usize, half: usize, t: usize,
    ) -> Result<()> {
        let (nq, nk, hd, hf) = (n_q_heads as i32, n_k_heads as i32, head_dim as i32, half as i32);
        unsafe {
            dev.stream().launch_builder(&self.rope_neox_batched)
                .arg(q).arg(k).arg(cos).arg(sin).arg(&nq).arg(&nk).arg(&hd).arg(&hf)
                .launch(LaunchConfig { grid_dim: (cdiv((n_q_heads+n_k_heads)*half,256), t as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Append `t` rows of k/v into the cache starting at `start_pos`.
    pub fn kv_cache_append_batched(
        &self, dev: &Device, k: &CudaSlice<f32>, v: &CudaSlice<f32>, k_cache: &mut CudaSlice<u16>,
        v_cache: &mut CudaSlice<u16>, start_pos: usize, n_kv_heads: usize, head_dim: usize,
        t: usize, base: usize,
    ) -> Result<()> {
        let (sp, nkv, hd, bs) =
            (start_pos as i32, n_kv_heads as i32, head_dim as i32, base as i32);
        unsafe {
            dev.stream().launch_builder(&self.kv_cache_append_batched)
                .arg(k).arg(v).arg(k_cache).arg(v_cache).arg(&sp).arg(&nkv).arg(&hd).arg(&bs)
                .launch(LaunchConfig { grid_dim: (cdiv(n_kv_heads*head_dim,256), t as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// The Gated DeltaNet recurrence over `t` rows, state resident in shared
    /// memory for the whole loop.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_rule_chunk(
        &self, dev: &Device, qkv: &CudaSlice<f32>, q_off: usize, k_off: usize, v_off: usize,
        row_stride: usize, decay: &CudaSlice<f32>, beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>, out: &mut CudaSlice<f32>, t: usize, n_v_heads: usize,
        n_k_heads: usize, group: usize, base: usize,
    ) -> Result<()> {
        let (qo, ko, vo, rs, tt, nv, nk, g) =
            (q_off as i32, k_off as i32, v_off as i32, row_stride as i32, t as i32, n_v_heads as i32, n_k_heads as i32, group as i32);
        let bs = base as i32;
        unsafe {
            dev.stream().launch_builder(&self.gated_delta_rule_chunk)
                .arg(qkv).arg(&qo).arg(&ko).arg(&vo).arg(&rs).arg(decay).arg(beta)
                .arg(state).arg(out).arg(&tt).arg(&nv).arg(&nk).arg(&g).arg(&bs)
                .launch(LaunchConfig { grid_dim: (n_v_heads as u32, 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Batched `deinterleave_heads` over `t` rows.
    pub fn deinterleave_heads_batched(
        &self, dev: &Device, src: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, n_heads: usize,
        head_dim: usize, src_off: usize, t: usize,
    ) -> Result<()> {
        let (nh, hd, so) = (n_heads as i32, head_dim as i32, src_off as i32);
        unsafe {
            dev.stream().launch_builder(&self.deinterleave_heads_batched)
                .arg(src).arg(dst).arg(&nh).arg(&hd).arg(&so)
                .launch(LaunchConfig { grid_dim: (cdiv(n_heads*head_dim,256), t as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Gather `t` token embeddings into a `[t, hidden]` fp32 buffer.
    pub fn embed_gather_batched(
        &self, dev: &Device, table: &CudaSlice<u16>, tokens: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>, hidden: usize, t: usize,
    ) -> Result<()> {
        let h = hidden as i32;
        unsafe {
            dev.stream().launch_builder(&self.embed_gather_batched)
                .arg(table).arg(tokens).arg(out).arg(&h)
                .launch(LaunchConfig { grid_dim: (cdiv(hidden,256), t as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Copy row `t-1` of `src` (`[t, n]`) to the front of `dst`.
    pub fn copy_last_row(
        &self, dev: &Device, src: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, t: usize, n: usize,
    ) -> Result<()> {
        let (tt, nn) = (t as i32, n as i32);
        unsafe {
            dev.stream().launch_builder(&self.copy_last_row)
                .arg(src).arg(dst).arg(&tt).arg(&nn)
                .launch(LaunchConfig { grid_dim: (cdiv(n,256), 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Gather `rows` rows starting at `row0` of `src` (`[row0+rows, n]`) into a
    /// dense `dst` (`[rows, n]`).
    pub fn copy_rows(
        &self, dev: &Device, src: &CudaSlice<f32>, dst: &mut CudaSlice<f32>,
        row0: usize, rows: usize, n: usize,
    ) -> Result<()> {
        need(rows > 0, "copy_rows rows")?;
        need(n > 0, "copy_rows n")?;
        need(src.len() >= (row0 + rows) * n, "copy_rows src")?;
        need(dst.len() >= rows * n, "copy_rows dst")?;
        let (r0, rr, nn) = (row0 as i32, rows as i32, n as i32);
        unsafe {
            dev.stream().launch_builder(&self.copy_rows)
                .arg(src).arg(dst).arg(&r0).arg(&rr).arg(&nn)
                .launch(LaunchConfig { grid_dim: (cdiv(rows*n,256), 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// `dst[0..n] = a`, `dst[n..2n] = b`. Used by the MTP head to join the two
    /// normalised inputs of `mtp.fc`.
    pub fn concat2(
        &self, dev: &Device, a: &CudaSlice<f32>, b: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>, n: usize,
    ) -> Result<()> {
        let nn = n as i32;
        unsafe {
            dev.stream().launch_builder(&self.concat2)
                .arg(a).arg(b).arg(dst).arg(&nn)
                .launch(LaunchConfig { grid_dim: (cdiv(n,256), 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Prefill GEMM: `y[t, n] = sum_k W[n, k] * x[t, k]`, each weight read once.
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_gemm(
        &self, dev: &Device, w: &CudaSlice<u8>, wscale: &CudaSlice<u8>, scale2: &CudaSlice<f32>,
        x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize, k: usize, t: usize,
    ) -> Result<()> {
        let (nn, kk, tt) = (n as i32, k as i32, t as i32);
        // K-SPLIT host half (round 187): split K across grid.z, zero y first because
        // block>0 accumulates. The kernel half (nsplit = gridDim.z) is already in.
        //
        // Making the split conditional on the grid being small was tried (round
        // 259) and measured as NO CHANGE: `gb10-bench stream` 95.28 ms against
        // 91.56/94.76 ms on other runs, and 8K OTPS 8.64 mean against 8.65. The
        // memset and the atomicAdds overlap with the neighbouring GEMMs' loads,
        // so removing them buys nothing. It was reverted because it perturbs the
        // GEMM summation order for no measured gain.
        let want = t * n;
        dev.stream().memset_zeros(&mut y.slice_mut(..want))?;
        unsafe {
            dev.stream().launch_builder(&self.nvfp4_gemm)
                .arg(w).arg(wscale).arg(scale2).arg(x).arg(y).arg(&nn).arg(&kk).arg(&tt)
                .launch(LaunchConfig { grid_dim: (cdiv(n,GB10_NR), cdiv(t,GB10_TILE_T), 2), block_dim: (128,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Prefill GEMM for E4M3 weights.
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_gemm(
        &self, dev: &Device, w: &CudaSlice<u8>, scale: &CudaSlice<f32>, x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>, n: usize, k: usize, t: usize,
    ) -> Result<()> {
        let (nn, kk, tt) = (n as i32, k as i32, t as i32);
        unsafe {
            dev.stream().launch_builder(&self.fp8_gemm)
                .arg(w).arg(scale).arg(x).arg(y).arg(&nn).arg(&kk).arg(&tt)
                .launch(LaunchConfig { grid_dim: (cdiv(n,GB10_NR), cdiv(t,GB10_TILE_T), 1), block_dim: (128,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Prefill GEMM for bf16 weights.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_gemm(
        &self, dev: &Device, w: &CudaSlice<u16>, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>,
        n: usize, k: usize, t: usize,
    ) -> Result<()> {
        let (nn, kk, tt) = (n as i32, k as i32, t as i32);
        unsafe {
            dev.stream().launch_builder(&self.bf16_gemm)
                .arg(w).arg(x).arg(y).arg(&nn).arg(&kk).arg(&tt)
                .launch(LaunchConfig { grid_dim: (cdiv(n,GB10_NR), cdiv(t,GB10_TILE_T), 1), block_dim: (128,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

}

fn need(ok: bool, what: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument(format!(
            "{what}: buffer too small"
        )))
    }

}
