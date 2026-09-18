//! Error type for the device-independent core.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid safetensors file {path}: {msg}")]
    BadSafetensors { path: PathBuf, msg: String },

    #[error("tensor {name} not found in checkpoint")]
    TensorNotFound { name: String },

    #[error("tensor {name} has {got} bytes but shape/dtype imply {want}")]
    TensorSizeMismatch {
        name: String,
        got: usize,
        want: usize,
    },

    #[error("config error: {0}")]
    Config(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("chat template error: {0}")]
    ChatTemplate(String),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, CoreError>;
