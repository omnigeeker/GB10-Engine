//! # gb10-cuda
//!
//! CUDA backend for GB10-Engine.
//!
//! Kernels are compiled to PTX for `sm_121` by `build.rs` and JIT-loaded here.
//! The decode path is bandwidth-bound (see `docs/PHYSICS.md`), so the GEMV
//! kernels are shaped to keep the weight stream perfectly coalesced.

pub mod kernels;

use cudarc::driver::{CudaContext, CudaModule, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::Arc;

pub use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, ValidAsZeroBits};
pub use kernels::Kernels;

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
}

pub type Result<T> = std::result::Result<T, CudaError>;

/// A CUDA device with the engine's kernels loaded.
pub struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: Kernels,
    modules: Vec<Arc<CudaModule>>,
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
            for name in kernels::KERNEL_NAMES {
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

        Ok(Self {
            ctx,
            stream,
            kernels,
            modules,
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

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
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
