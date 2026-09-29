//! # gb10-cuda
//!
//! CUDA backend for GB10-Engine.
//!
//! Kernels are compiled to PTX for `sm_121` by `build.rs` and JIT-loaded here.
//! The decode path is bandwidth-bound (see `docs/PHYSICS.md`), so the GEMV
//! kernels are shaped to keep the weight stream perfectly coalesced.

pub mod kernels;
pub mod ops;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaModule, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub use cudarc::driver::sys::CUevent_flags;
pub use cudarc::driver::{CudaEvent, CudaSlice, DeviceRepr, LaunchConfig, ValidAsZeroBits};
pub use kernels::Kernels;
pub use ops::Ops;

/// A CUDA event with timing enabled.
///
/// `CudaContext::new_event` defaults to `CU_EVENT_DISABLE_TIMING`, and under
/// that flag `CudaEvent::elapsed_ms` returns a meaningless number rather than
/// failing. Anything that reports a duration must get its events from here.
pub fn timing_event(dev: &Device) -> std::result::Result<CudaEvent, CudaError> {
    Ok(dev
        .stream()
        .context()
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?)
}

/// PTX files produced by `build.rs`, colon separated.
const KERNEL_PTX: &str = env!("GB10_KERNEL_PTX");
pub const CUDA_ARCH: &str = env!("GB10_CUDA_ARCH");

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("cuda driver error: {0}")]
    Driver(#[from] DriverError),
    #[error("kernel {0} not found in any loaded module")]
    KernelNotFound(String),
    #[error("no PTX kernels were compiled")]
    NoKernels,
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("cublas error: {0}")]
    Cublas(String),
}

pub type Result<T> = std::result::Result<T, CudaError>;

/// Growable bf16 buffers shared by every tensor-core prefill GEMM.
///
/// The prefill path used to allocate these per matrix, which measured at 39% of
/// the whole pipeline (`gb10-bench tc-phase`) -- not because of the memset, which
/// a bandwidth estimate would put at ~1.2 ms, but because the measured 4.46 ms
/// works out to ~60 GB/s once the per-call allocation and launch overhead is
/// included. Held here so each buffer is allocated once and then only grown.
#[derive(Default)]
pub struct TcScratch {
    /// Dequantised weights, `[n, k]`, fp16. See `cublas_gemm_f16_f32`.
    pub w: Option<CudaSlice<half::f16>>,
    /// fp16 activations, `[t, k]`.
    ///
    /// fp16 and not bf16: bf16's 8 mantissa bits were not enough for the
    /// long-context greedy argmax. See `cublas_gemm_bf16_f16_f32`.
    pub x: Option<CudaSlice<half::f16>>,
    /// bf16 GEMM output, `[t, n]`.
    pub y: Option<CudaSlice<half::bf16>>,
    /// One fp32 element, passed to the epilogue when there is no `s2` to apply.
    pub one: Option<CudaSlice<f32>>,
}

/// A CUDA device with the engine's kernels loaded.
pub struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: Kernels,
    ops: Ops,
    modules: Vec<Arc<CudaModule>>,
    blas: CudaBlas,
    tc_scratch: Mutex<TcScratch>,
}

impl Device {
    /// Initialise the device at `ordinal` and load every compiled kernel.
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();

        let mut modules = Vec::new();
        let mut by_name: HashMap<String, cudarc::driver::CudaFunction> = HashMap::new();
        for path in KERNEL_PTX.split(':').filter(|p| !p.is_empty()) {
            let module = ctx.load_module(Ptx::from_file(path))?;
            for name in kernels::KERNEL_NAMES
                .iter()
                .chain(ops::OP_KERNEL_NAMES.iter())
            {
                if let Ok(f) = module.load_function(name) {
                    by_name.insert((*name).to_string(), f);
                }
            }
            modules.push(module);
        }
        if by_name.is_empty() {
            return Err(CudaError::NoKernels);
        }
        let kernels = Kernels::from_map(&mut by_name)?;
        let ops = Ops::from_map(&mut by_name)?;

        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| CudaError::Cublas(format!("{e:?}")))?;

        Ok(Self {
            ctx,
            stream,
            kernels,
            ops,
            modules,
            blas,
            tc_scratch: Mutex::new(TcScratch::default()),
        })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn ops(&self) -> &Ops {
        &self.ops
    }

    /// The cuBLAS handle used by the bf16 tensor-core prefill GEMM.
    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// The shared tensor-core prefill scratch. Poisoning is ignored: these are
    /// plain device buffers and a panic elsewhere must not make the device
    /// permanently unusable.
    pub fn tc_scratch(&self) -> std::sync::MutexGuard<'_, TcScratch> {
        self.tc_scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    /// Raise any CUDA error the driver recorded and cudarc swallowed.
    ///
    /// `CudaSlice::drop` synchronises the stream and hands the result to
    /// `CudaContext::record_err`, which stores it in an atomic rather than
    /// raising it. An asynchronous failure -- an illegal access, a launch
    /// abort, a watchdog kill under contention -- therefore sets a sticky
    /// error that nothing reports, and the engine carries on reading whatever
    /// the failed kernel left in its output buffer. On the argmax that is a
    /// wrong token that still decodes to fluent text, so it is invisible.
    ///
    /// Calling this before a result is trusted turns that into a hard failure.
    /// It is an atomic swap on the host, with no device work.
    pub fn check_err(&self) -> Result<()> {
        Ok(self.ctx.check_err()?)
    }

    /// Device name, e.g. `NVIDIA GB10`.
    pub fn name(&self) -> Result<String> {
        Ok(self.ctx.name()?)
    }

    /// Compute capability as `(major, minor)`.
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        use cudarc::driver::sys;
        let major = self
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok((major, minor))
    }

    pub fn total_memory(&self) -> Result<usize> {
        Ok(self.ctx.total_mem()?)
    }

    /// Number of PTX modules kept alive for the lifetime of the device.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<Device> {
        match Device::new(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipping: no usable CUDA device ({e})");
                None
            }
        }
    }

    #[test]
    fn initialises_device_and_loads_kernels() {
        let Some(dev) = device() else { return };
        let name = dev.name().unwrap();
        assert!(!name.is_empty());
        assert_eq!(dev.compute_capability().unwrap(), (12, 1), "sm_121");
        assert!(dev.module_count() >= 1);
        assert!(dev.total_memory().unwrap() > 100usize << 30);
    }

    #[test]
    fn every_declared_kernel_is_present() {
        let Some(dev) = device() else { return };
        // Kernels::from_map already errors on a missing symbol; this asserts
        // the set is the one we expect.
        assert!(dev.kernels().has_all());
    }
}
