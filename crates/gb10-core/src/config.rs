//! Model configuration for `qwen3_5` (Qwen3.8-27B), including the MTP head.
//!
//! Parsed from the checkpoint's `config.json`. Field names mirror the HF
//! config exactly so the two can be diffed; defaults are only used for keys
//! that are genuinely optional in the HF schema.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Which attention a decoder layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Gated DeltaNet linear attention (constant-size recurrent state).
    LinearAttention,
    /// Standard grouped-query causal attention with paged KV cache.
    FullAttention,
}

/// RoPE / mRoPE parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}
fn default_partial_rotary_factor() -> f32 {
    1.0
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
            mrope_interleaved: false,
            mrope_section: Vec::new(),
            rope_type: None,
        }
    }
}

/// The text backbone configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    pub layer_types: Vec<LayerType>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,

    // Gated DeltaNet (linear attention) geometry.
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,

    /// `true` when the full-attention q projection emits a gate alongside q.
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub output_gate_type: Option<String>,

    #[serde(default)]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_parameters: RopeParameters,

    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub hidden_act: Option<String>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_full_attention_interval() -> usize {
    4
}
fn default_linear_conv_kernel_dim() -> usize {
    4
}
fn default_linear_key_head_dim() -> usize {
    128
}
fn default_linear_num_key_heads() -> usize {
    16
}
fn default_linear_num_value_heads() -> usize {
    48
}
fn default_linear_value_head_dim() -> usize {
    128
}

/// Map a tensor name onto the module path used by `quantization_config`.
///
/// `model...gate_proj.weight`        -> `model...gate_proj`
/// `model...gate_proj.weight_scale`  -> `model...gate_proj`
/// `model...gate_proj.weight_scale_2`-> `model...gate_proj`
/// `lm_head.weight`                  -> `lm_head`
pub fn strip_weight_suffix(tensor_name: &str) -> &str {
    for suffix in [
        ".weight_scale_2",
        ".weight_scale",
        ".input_scale",
        ".weight",
    ] {
        if let Some(stripped) = tensor_name.strip_suffix(suffix) {
            return stripped;
        }
    }
    tensor_name
}

/// Top-level checkpoint configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
    pub text_config: TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub video_token_id: Option<u32>,
}

/// modelopt quantization description (`quant_method: "modelopt"`).
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    #[serde(default)]
    pub quant_algo: Option<String>,
    #[serde(default)]
    pub quant_method: Option<String>,
    #[serde(default)]
    pub quantized_layers: std::collections::HashMap<String, QuantLayer>,
    #[serde(default)]
    pub ignore: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantLayer {
    #[serde(default)]
    pub quant_algo: Option<String>,
    #[serde(default)]
    pub group_size: Option<usize>,
}

/// The quantization applied to one weight, resolved from the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantAlgo {
    /// 8-bit float (E4M3) weights with a per-tensor scale.
    Fp8,
    /// NVFP4: 4-bit E2M1 packed two-per-byte, group-16 E4M3 scales, plus a
    /// per-tensor fp32 global scale.
    NvFp4 { group_size: usize },
    /// Unquantized (bf16/fp16/fp32) — norms, embeddings, MTP layers, vision.
    None,
}

impl TextConfig {
    /// Number of layers using full attention.
    pub fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count()
    }

    /// Number of layers using Gated DeltaNet.
    pub fn num_linear_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count()
    }

    /// Full-attention q projection output width. With `attn_output_gate` the
    /// projection emits `2 * num_attention_heads * head_dim` (query + gate).
    pub fn q_proj_out_features(&self) -> usize {
        let base = self.num_attention_heads * self.head_dim;
        if self.attn_output_gate {
            base * 2
        } else {
            base
        }
    }

    pub fn kv_proj_out_features(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Total qk dimension inside a Gated DeltaNet layer
    /// (`num_key_heads * key_head_dim`, used for both q and k).
    pub fn linear_qk_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// Value dimension inside a Gated DeltaNet layer.
    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Rotary dimension actually rotated (partial rotary).
    pub fn rotary_dim(&self) -> usize {
        let factor = if self.rope_parameters.partial_rotary_factor > 0.0 {
            self.rope_parameters.partial_rotary_factor
        } else {
            self.partial_rotary_factor
        };
        let d = (self.head_dim as f32 * factor).round() as usize;
        // RoPE needs an even number of rotated channels.
        d - (d % 2)
    }

    /// Resolve the quantization scheme for a named weight.
    ///
    /// `ignore` patterns from the config (`mtp*`) are honoured, so MTP and
    /// vision weights stay unquantized even though the global algo is
    /// MIXED_PRECISION.
    ///
    /// `quantized_layers` is keyed by *module* path
    /// (`model.language_model.layers.0.mlp.gate_proj`), while tensors carry a
    /// suffix (`...gate_proj.weight`, `...weight_scale`, `...weight_scale_2`).
    /// The suffix is therefore stripped before lookup, so a scale tensor
    /// resolves to the same scheme as the weight it belongs to.
    pub fn quant_for(&self, cfg: &ModelConfig, tensor_name: &str) -> QuantAlgo {
        let module = strip_weight_suffix(tensor_name);
        if let Some(q) = &cfg.quantization_config {
            for pat in &q.ignore {
                let pat = pat.trim_end_matches('*');
                if !pat.is_empty()
                    && (tensor_name.starts_with(pat) || module.starts_with(pat))
                {
                    return QuantAlgo::None;
                }
            }
            if let Some(layer) = q.quantized_layers.get(module) {
                return match layer.quant_algo.as_deref() {
                    Some("NVFP4") => QuantAlgo::NvFp4 {
                        group_size: layer.group_size.unwrap_or(16),
                    },
                    Some("FP8") => QuantAlgo::Fp8,
                    _ => QuantAlgo::None,
                };
            }
        }
        QuantAlgo::None
    }
}

impl ModelConfig {
    pub fn from_json_str(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path.as_ref())?;
        Self::from_json_str(&s)
    }

    /// Effective EOS set. `text_config.eos_token_id` may be a scalar or a
    /// list, and `generation_config.json` may add more.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        let mut out = Vec::new();
        match &self.text_config.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                if let Some(v) = n.as_u64() {
                    out.push(v as u32)
                }
            }
            Some(serde_json::Value::Array(a)) => {
                for v in a {
                    if let Some(v) = v.as_u64() {
                        out.push(v as u32)
                    }
                }
            }
            _ => {}
        }
        out
    }

    /// Load and merge `generation_config.json` EOS ids into the set.
    pub fn eos_token_ids_with_generation_config(&self, gen_cfg: &serde_json::Value) -> Vec<u32> {
        let mut out = self.eos_token_ids();
        if let Some(v) = gen_cfg.get("eos_token_id") {
            let add = |x: &serde_json::Value, out: &mut Vec<u32>| {
                if let Some(u) = x.as_u64() {
                    if !out.contains(&(u as u32)) {
                        out.push(u as u32);
                    }
                }
            };
            match v {
                serde_json::Value::Array(a) => a.iter().for_each(|x| add(x, &mut out)),
                other => add(other, &mut out),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_config_path() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new("../../models/Qwen3.8-27B-NVFP4/config.json");
        if p.exists() {
            Some(p.to_path_buf())
        } else {
            None
        }
    }

    #[test]
    fn parses_real_checkpoint_config() {
        let Some(p) = real_config_path() else {
            eprintln!("skipping: model not present");
            return;
        };
        let cfg = ModelConfig::from_file(&p).expect("parse config.json");
        let t = &cfg.text_config;
        assert_eq!(t.hidden_size, 5120);
        assert_eq!(t.intermediate_size, 17408);
        assert_eq!(t.num_hidden_layers, 64);
        assert_eq!(t.num_attention_heads, 24);
        assert_eq!(t.num_key_value_heads, 4);
        assert_eq!(t.head_dim, 256);
        assert_eq!(t.vocab_size, 248320);
        assert_eq!(t.layer_types.len(), 64);
        assert_eq!(t.num_linear_attention_layers(), 48);
        assert_eq!(t.num_full_attention_layers(), 16);
        assert_eq!(t.mtp_num_hidden_layers, 1);
        assert!(t.attn_output_gate);
        // partial rotary 0.25 of head_dim 256 -> 64 channels, even.
        assert_eq!(t.rotary_dim(), 64);
        assert_eq!(t.q_proj_out_features(), 24 * 256 * 2);
        assert_eq!(t.kv_proj_out_features(), 4 * 256);
        assert_eq!(t.linear_qk_dim(), 16 * 128);
        assert_eq!(t.linear_value_dim(), 48 * 128);
    }

    #[test]
    fn layer_pattern_is_period_4_ending_in_full_attention() {
        let Some(p) = real_config_path() else { return };
        let cfg = ModelConfig::from_file(&p).unwrap();
        for (i, lt) in cfg.text_config.layer_types.iter().enumerate() {
            let expect_full = (i + 1) % cfg.text_config.full_attention_interval == 0;
            assert_eq!(
                *lt == LayerType::FullAttention,
                expect_full,
                "layer {i} layer_type mismatch"
            );
        }
    }

    #[test]
    fn quantization_resolution_matches_modelopt_config() {
        let Some(p) = real_config_path() else { return };
        let cfg = ModelConfig::from_file(&p).unwrap();
        let t = &cfg.text_config;

        // NVFP4 group-16 MLP
        assert_eq!(
            t.quant_for(&cfg, "model.language_model.layers.0.mlp.gate_proj.weight"),
            QuantAlgo::NvFp4 { group_size: 16 }
        );
        // FP8 attention projections
        assert_eq!(
            t.quant_for(&cfg, "model.language_model.layers.3.self_attn.q_proj.weight"),
            QuantAlgo::Fp8
        );
        assert_eq!(
            t.quant_for(&cfg, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
            QuantAlgo::Fp8
        );
        // NVFP4 lm_head
        assert_eq!(
            t.quant_for(&cfg, "lm_head.weight"),
            QuantAlgo::NvFp4 { group_size: 16 }
        );
        // `ignore: ["mtp*"]` keeps the MTP head unquantized
        assert_eq!(t.quant_for(&cfg, "mtp.layers.0.mlp.gate_proj.weight"), QuantAlgo::None);
        // norms are never quantized
        assert_eq!(
            t.quant_for(&cfg, "model.language_model.layers.0.input_layernorm.weight"),
            QuantAlgo::None
        );
    }

    #[test]
    fn eos_set_includes_generation_config_ids() {
        let Some(p) = real_config_path() else { return };
        let cfg = ModelConfig::from_file(&p).unwrap();
        let gen: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(p.parent().unwrap().join("generation_config.json")).unwrap(),
        )
        .unwrap();
        let eos = cfg.eos_token_ids_with_generation_config(&gen);
        assert!(eos.contains(&248046), "im_end must be EOS, got {eos:?}");
        assert!(eos.contains(&248044));
    }
}
