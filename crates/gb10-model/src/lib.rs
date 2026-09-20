//! # gb10-model
//!
//! The Qwen3.5 model graph: 64 decoder layers alternating Gated DeltaNet
//! (linear attention) and full attention, plus the MTP head.
//!
//! Weights stay in their checkpoint quantization (NVFP4 / FP8 / bf16) and are
//! dequantized on the fly by the GEMV kernels — see `docs/PHYSICS.md` for why
//! materialising them as fp32 would destroy the roofline.
//!
//! Exact semantics (zero-centered norms, sigmoid output gate, per-head q/gate
//! interleaving) are documented with their source line numbers in
//! `docs/ARCHITECTURE.md`.

pub mod layer;
pub mod rope;
pub mod weights;

pub use layer::{FullAttnLayer, Layer, LayerState, Mlp, Scratch};
pub use rope::rope_tables;
pub use weights::{Linear, LinearData, Store};
