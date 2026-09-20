//! Typed launch wrappers for the non-GEMV kernels: norms, elementwise ops,
//! RoPE, the depthwise conv, the Gated DeltaNet recurrence and prefill
//! attention.
//!
//! Every wrapper validates its shapes before launching. These kernels are not
//! the bandwidth bottleneck — the 17.6 GB of quantized weights streamed by the
//! GEMV kernels dominates by ~100x — so they are written for obvious
//! correctness rather than peak occupancy.

use crate::{CudaError, CudaSlice, Device, LaunchConfig, Result};
use cudarc::driver::{CudaFunction, PushKernelArg};
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
    "attn_decode_kernel",
    "kv_cache_append_kernel",
    "embed_gather_kernel",
    "argmax_kernel",
];

/// Gated DeltaNet key/value head geometry (fixed by the checkpoint).
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
    attn_decode: CudaFunction,
    kv_cache_append: CudaFunction,
    embed_gather: CudaFunction,
    argmax: CudaFunction,
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
            attn_decode: take(map, "attn_decode_kernel")?,
            kv_cache_append: take(map, "kv_cache_append_kernel")?,
            embed_gather: take(map, "embed_gather_kernel")?,
            argmax: take(map, "argmax_kernel")?,
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
        unsafe {
            dev.stream()
                .launch_builder(&self.conv1d_step_silu)
                .arg(x)
                .arg(w)
                .arg(hist)
                .arg(y)
                .arg(&c)
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
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_keys: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        need(n_keys > 0, "attn_decode: no keys")?;
        need(
            k_cache.len() >= n_keys * n_kv_heads * head_dim
                && v_cache.len() >= n_keys * n_kv_heads * head_dim,
            "attn_decode cache",
        )?;
        let (nk, nq, nkv, hd) =
            (n_keys as i32, n_q_heads as i32, n_kv_heads as i32, head_dim as i32);
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
                .launch(LaunchConfig {
                    grid_dim: (n_q_heads as u32, 1, 1),
                    block_dim: (block_for(head_dim, 256), 1, 1),
                    shared_mem_bytes: (n_keys * 4) as u32,
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
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
        pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let n = n_kv_heads * head_dim;
        need(k.len() >= n && v.len() >= n, "kv append src")?;
        need(
            k_cache.len() >= (pos + 1) * n && v_cache.len() >= (pos + 1) * n,
            "kv append cache",
        )?;
        let (p, nkv, hd) = (pos as i32, n_kv_heads as i32, head_dim as i32);
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
    pub fn attn_prefill(
        &self,
        dev: &Device,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        need(
            q.len() >= n_tokens * n_q_heads * head_dim
                && k.len() >= n_tokens * n_kv_heads * head_dim
                && v.len() >= n_tokens * n_kv_heads * head_dim
                && out.len() >= n_tokens * n_q_heads * head_dim,
            "attn_prefill",
        )?;
        let (t, nq, nk, hd) =
            (n_tokens as i32, n_q_heads as i32, n_kv_heads as i32, head_dim as i32);
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
                .launch(LaunchConfig {
                    grid_dim: (n_q_heads as u32, n_tokens as u32, 1),
                    block_dim: (block_for(head_dim, 256), 1, 1),
                    shared_mem_bytes: (n_tokens * 4) as u32,
                })?;
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
