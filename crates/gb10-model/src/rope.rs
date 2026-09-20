//! Rotary position embeddings.
//!
//! Qwen3.5 uses partial RoPE over `int(head_dim * partial_rotary_factor)` = 64
//! of 256 channels. The mRoPE grid machinery in the reference
//! (`mrope_section = [11, 11, 10]`) only differs between the T/H/W grids for
//! image input; for text all three position-id rows are identical, so the
//! recomposition is a no-op and the frequency table is simply duplicated. The
//! result is a plain GPT-NeoX rotation — see `docs/ARCHITECTURE.md`.

use gb10_core::config::TextConfig;

/// `(cos, sin)` tables of length `rotary_dim / 2` for a single position.
pub fn rope_tables(cfg: &TextConfig, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let rotary = cfg.rotary_dim();
    let half = rotary / 2;
    let theta = cfg.rope_parameters.rope_theta as f64;

    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for (i, (c, s)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
        // inv_freq[i] = 1 / theta^(2i / rotary_dim)
        let inv = 1.0 / theta.powf((2 * i) as f64 / rotary as f64);
        let angle = pos as f64 * inv;
        *c = angle.cos() as f32;
        *s = angle.sin() as f32;
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn config() -> Option<TextConfig> {
        let dir = Path::new("../../models/Qwen3.8-27B-NVFP4");
        if !dir.exists() {
            return None;
        }
        let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
        let cfg: gb10_core::config::ModelConfig = serde_json::from_str(&raw).ok()?;
        Some(cfg.text_config)
    }

    #[test]
    fn rotary_dim_is_a_quarter_of_head_dim() {
        let Some(cfg) = config() else { return };
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.rotary_dim(), 64);
    }

    #[test]
    fn position_zero_is_identity_rotation() {
        let Some(cfg) = config() else { return };
        let (cos, sin) = rope_tables(&cfg, 0);
        assert_eq!(cos.len(), 32);
        for (c, s) in cos.iter().zip(sin.iter()) {
            assert!((c - 1.0).abs() < 1e-6, "cos[0] should be 1, got {c}");
            assert!(s.abs() < 1e-6, "sin[0] should be 0, got {s}");
        }
    }

    #[test]
    fn frequencies_match_the_reference_formula() {
        let Some(cfg) = config() else { return };
        let (cos, _) = rope_tables(&cfg, 1);
        let theta = cfg.rope_parameters.rope_theta as f64;
        for i in [0usize, 1, 15, 31] {
            let inv = 1.0 / theta.powf((2 * i) as f64 / 64.0);
            let want = (inv).cos() as f32;
            assert!(
                (cos[i] - want).abs() < 1e-7,
                "i={i}: got {} want {want}",
                cos[i]
            );
        }
    }
}
