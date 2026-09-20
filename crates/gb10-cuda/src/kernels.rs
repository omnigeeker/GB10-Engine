//! Kernel handles and typed launch wrappers.

use crate::{CudaError, CudaSlice, Device, LaunchConfig, Result};
use cudarc::driver::{CudaFunction, PushKernelArg};
use std::collections::HashMap;

/// Every kernel symbol the engine expects to find in the PTX modules.
pub const KERNEL_NAMES: &[&str] = &[
    "nvfp4_gemv_kernel",
    "fp8_gemv_kernel",
    "bf16_gemv_kernel",
    "nvfp4_gemv_batch_kernel",
    "fp8_gemv_batch_kernel",
    "bf16_gemv_batch_kernel",
];

/// Largest batch the multi-sequence GEMV kernels accept. Must match
/// `GB10_BATCH_MAX` in `kernels/gemv.cu`. Larger batches fall back to the
/// per-sequence kernels, which are correct but re-read the weights per
/// sequence; only tiny projections are ever called that way.
pub const GEMV_BATCH_MAX: usize = 16;

/// Threads per block for the GEMV kernels (8 warps).
pub const GEMV_BLOCK: u32 = 256;
const WARPS_PER_BLOCK: u32 = GEMV_BLOCK / 32;

/// Output rows each warp accumulates simultaneously. Must match the `ROWS`
/// template argument used by the exported kernels in `kernels/gemv.cu`.
pub const GEMV_ROWS_PER_WARP: u32 = 1;

/// Loaded kernel functions.
pub struct Kernels {
    nvfp4_gemv: CudaFunction,
    nvfp4_gemv_batch: CudaFunction,
    fp8_gemv: CudaFunction,
    bf16_gemv: CudaFunction,
    fp8_gemv_batch: CudaFunction,
    bf16_gemv_batch: CudaFunction,
}

impl Kernels {
    pub(crate) fn from_map(
        map: &mut HashMap<String, CudaFunction>,
    ) -> Result<Self> {
        let take = |m: &mut HashMap<String, CudaFunction>, n: &str| {
            m.remove(n)
                .ok_or_else(|| CudaError::KernelNotFound(n.to_string()))
        };
        Ok(Self {
            nvfp4_gemv: take(map, "nvfp4_gemv_kernel")?,
            nvfp4_gemv_batch: take(map, "nvfp4_gemv_batch_kernel")?,
            fp8_gemv: take(map, "fp8_gemv_kernel")?,
            bf16_gemv: take(map, "bf16_gemv_kernel")?,
            fp8_gemv_batch: take(map, "fp8_gemv_batch_kernel")?,
            bf16_gemv_batch: take(map, "bf16_gemv_batch_kernel")?,
        })
    }

    pub fn has_all(&self) -> bool {
        true
    }

    // ---------------------------------------------------------------- NVFP4 --
    /// `y[b,n] = scale2 * sum_k dequant_nvfp4(W[n,k]) * x[b,k]`
    ///
    /// * `w`      packed E2M1, `[N, K/2]` bytes
    /// * `wscale` E4M3 group scales, `[N, K/16]` bytes
    /// * `scale2` per-tensor global scale, `[1]`
    pub fn nvfp4_gemv(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        w: &CudaSlice<u8>,
        wscale: &CudaSlice<u8>,
        scale2: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<()> {
        check_gemv_shapes("nvfp4_gemv", x, w, y, n, k, batch, 1)?;
        let (n_i32, k_i32) = (n as i32, k as i32);
        if batch > 1 && batch <= GEMV_BATCH_MAX {
            let b_i32 = batch as i32;
            let grid = (((n as u32) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);
            unsafe {
                dev.stream()
                    .launch_builder(&self.nvfp4_gemv_batch)
                    .arg(x).arg(w).arg(wscale).arg(scale2).arg(y)
                    .arg(&n_i32).arg(&k_i32).arg(&b_i32)
                    .launch(LaunchConfig { grid_dim: grid, block_dim: (GEMV_BLOCK, 1, 1), shared_mem_bytes: 0 })?;
            }
            return Ok(());
        }
        let (grid, block, smem) = gemv_launch_config(n, k, batch);
        let f = &self.nvfp4_gemv;
        unsafe {
            dev.stream()
                .launch_builder(f)
                .arg(x)
                .arg(w)
                .arg(wscale)
                .arg(scale2)
                .arg(y)
                .arg(&n_i32)
                .arg(&k_i32)
                .launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: block,
                    shared_mem_bytes: smem,
                })?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------ FP8 --
    /// `y[b,n] = wscale * sum_k W[n,k] * x[b,k]` with `W` stored as E4M3.
    pub fn fp8_gemv(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        w: &CudaSlice<u8>,
        wscale: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<()> {
        check_gemv_shapes("fp8_gemv", x, w, y, n, k, batch, 1)?;
        let (n_i32, k_i32) = (n as i32, k as i32);
        if batch > 1 && batch <= GEMV_BATCH_MAX {
            let b_i32 = batch as i32;
            let grid = (((n as u32) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);
            unsafe {
                dev.stream()
                    .launch_builder(&self.fp8_gemv_batch)
                    .arg(x).arg(w).arg(wscale).arg(y)
                    .arg(&n_i32).arg(&k_i32).arg(&b_i32)
                    .launch(LaunchConfig { grid_dim: grid, block_dim: (GEMV_BLOCK, 1, 1), shared_mem_bytes: 0 })?;
            }
            return Ok(());
        }
        let (grid, block, smem) = gemv_launch_config(n, k, batch);
        let f = &self.fp8_gemv;
        unsafe {
            dev.stream()
                .launch_builder(f)
                .arg(x)
                .arg(w)
                .arg(wscale)
                .arg(y)
                .arg(&n_i32)
                .arg(&k_i32)
                .launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: block,
                    shared_mem_bytes: smem,
                })?;
        }
        Ok(())
    }

    // ----------------------------------------------------------------- bf16 --
    /// `y[b,n] = sum_k W[n,k] * x[b,k]` with `W` stored as bf16.
    pub fn bf16_gemv(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        w: &CudaSlice<u16>,
        y: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<()> {
        let (n_i32, k_i32) = (n as i32, k as i32);
        if batch > 1 && batch <= GEMV_BATCH_MAX {
            let b_i32 = batch as i32;
            let grid = (((n as u32) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);
            unsafe {
                dev.stream()
                    .launch_builder(&self.bf16_gemv_batch)
                    .arg(x).arg(w).arg(y)
                    .arg(&n_i32).arg(&k_i32).arg(&b_i32)
                    .launch(LaunchConfig { grid_dim: grid, block_dim: (GEMV_BLOCK, 1, 1), shared_mem_bytes: 0 })?;
            }
            return Ok(());
        }
        let (grid, block, smem) = gemv_launch_config(n, k, batch);
        let f = &self.bf16_gemv;
        unsafe {
            dev.stream()
                .launch_builder(f)
                .arg(x)
                .arg(w)
                .arg(y)
                .arg(&n_i32)
                .arg(&k_i32)
                .launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: block,
                    shared_mem_bytes: smem,
                })?;
        }
        Ok(())
    }
}

/// Grid/block configuration for a GEMV.
///
/// No dynamic shared memory is used: the activation tile lives in registers
/// and is reused across `GEMV_ROWS_PER_WARP` output rows. GB10 caps shared
/// memory at 100 KB per SM, so a shared-memory staging scheme would have
/// limited the kernel to a single block per SM.
pub fn gemv_launch_config(
    n: usize,
    _k: usize,
    batch: usize,
) -> ((u32, u32, u32), (u32, u32, u32), u32) {
    let rows_per_block = WARPS_PER_BLOCK * GEMV_ROWS_PER_WARP;
    let gx = ((n as u32) + rows_per_block - 1) / rows_per_block;
    let gy = batch.max(1) as u32;
    ((gx.max(1), gy, 1), (GEMV_BLOCK, 1, 1), 0)
}

fn check_gemv_shapes(
    which: &str,
    x: &CudaSlice<f32>,
    w: &CudaSlice<u8>,
    y: &CudaSlice<f32>,
    n: usize,
    k: usize,
    batch: usize,
    bytes_per_elem_num: usize,
) -> Result<()> {
    let _ = bytes_per_elem_num;
    if n == 0 || k == 0 || batch == 0 {
        return Err(CudaError::InvalidArgument(format!(
            "{which}: degenerate shape n={n} k={k} batch={batch}"
        )));
    }
    if x.len() < batch * k {
        return Err(CudaError::InvalidArgument(format!(
            "{which}: x has {} elements, need {}",
            x.len(),
            batch * k
        )));
    }
    if y.len() < batch * n {
        return Err(CudaError::InvalidArgument(format!(
            "{which}: y has {} elements, need {}",
            y.len(),
            batch * n
        )));
    }
    if w.is_empty() {
        return Err(CudaError::InvalidArgument(format!("{which}: empty weights")));
    }
    Ok(())
}
