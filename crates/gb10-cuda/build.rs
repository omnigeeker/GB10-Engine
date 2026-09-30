// Compile kernels/*.cu to PTX for the GB10 target and expose the paths to the
// crate as a colon-separated compile-time env var.
//
// PTX (not cubin) is used so the driver JITs for the exact device it finds;
// sm_121 is the measured compute capability of the DGX Spark GB10.
//
// `sm_121a` (not `sm_121`) is required for the architecture-specific
// `kind::mxf4nvf4.block_scale` tensor-core mma used by the NVFP4 MLP GEMM
// (`kernels/nvfp4_gemm.cu`). The plain `sm_121` target cannot assemble that
// instruction at all. `sm_121a` is a strict superset of `sm_121`, so every
// pre-existing kernel compiles and runs identically -- verified by an exact
// `generate` match before the FP4 kernel was written.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CUDA_ARCH: &str = "sm_121a";

fn find_nvcc() -> PathBuf {
    if let Ok(p) = env::var("NVCC") {
        return PathBuf::from(p);
    }
    for cand in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"] {
        if Path::new(cand).exists() {
            return PathBuf::from(cand);
        }
    }
    PathBuf::from("nvcc")
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest.join("../../kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", kernel_dir.display());
    println!("cargo:rerun-if-env-changed=NVCC");

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&kernel_dir)
        .unwrap_or_else(|e| panic!("kernels/ dir {}: {e}", kernel_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "cu").unwrap_or(false))
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no .cu files in {}", kernel_dir.display());

    let nvcc = find_nvcc();
    let mut ptx_paths = Vec::new();

    for cu in &entries {
        let stem = cu.file_stem().unwrap().to_str().unwrap().to_string();
        let ptx = out_dir.join(format!("{stem}.ptx"));
        let status = Command::new(&nvcc)
            .arg("-arch")
            .arg(CUDA_ARCH)
            .arg("-ptx")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-lineinfo")
            .arg("-I")
            .arg(&kernel_dir)
            .arg("-o")
            .arg(&ptx)
            .arg(cu)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
        assert!(status.success(), "nvcc failed for {}", cu.display());
        ptx_paths.push(ptx);
    }

    println!(
        "cargo:rustc-env=GB10_KERNEL_PTX={}",
        ptx_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(":")
    );
    println!("cargo:rustc-env=GB10_CUDA_ARCH={CUDA_ARCH}");
}
