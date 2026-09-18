//! # gb10-core
//!
//! Device-independent foundation for GB10-Engine: checkpoint configuration,
//! a memory-mapped safetensors reader, the tokenizer, and chat-template
//! rendering.
//!
//! Nothing here touches CUDA, so it builds and tests on any host.

pub mod chat;
pub mod config;
pub mod error;
pub mod safetensors;
pub mod tokenizer;

pub use chat::{ChatMessage, ChatTemplate, ChatTemplateOptions};
pub use config::{
    LayerType, ModelConfig, QuantAlgo, QuantizationConfig, RopeParameters, TextConfig,
};
pub use error::{CoreError, Result};
pub use safetensors::{DType, ShardedSafeTensors, TensorInfo};
pub use tokenizer::QwenTokenizer;

/// Shape of a tensor as used by the weight loader.
pub type Shape = Vec<usize>;
