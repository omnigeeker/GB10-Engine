//! Tokenizer wrapper around the checkpoint's own `tokenizer.json`.
//!
//! We deliberately load the HF `tokenizer.json` rather than re-deriving the
//! byte-level BPE from `vocab.json` + `merges.txt`: the pre-tokenizer regex
//! uses a negative lookahead (`\s+(?!\S)`) that plain `regex` cannot express,
//! and token-exactness is a hard correctness gate (see `docs/TARGETS.md` T7).

use std::path::Path;
use tokenizers::Tokenizer;

pub struct QwenTokenizer {
    inner: Tokenizer,
    eos_ids: Vec<u32>,
}

impl QwenTokenizer {
    /// Load `tokenizer.json` from a checkpoint directory.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = dir.as_ref().join("tokenizer.json");
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))?;
        Ok(Self {
            inner,
            eos_ids: Vec::new(),
        })
    }

    /// Build directly from a `tokenizer.json` string.
    pub fn from_json_str(s: &str) -> anyhow::Result<Self> {
        let inner = Tokenizer::from_bytes(s.as_bytes())
            .map_err(|e| anyhow::anyhow!("parsing tokenizer.json: {e}"))?;
        Ok(Self {
            inner,
            eos_ids: Vec::new(),
        })
    }

    pub fn with_eos_ids(mut self, ids: Vec<u32>) -> Self {
        self.eos_ids = ids;
        self
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_ids.contains(&id)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode text to token ids. `add_special` applies the post-processor
    /// (which for this model is a no-op ByteLevel pass).
    pub fn encode(&self, text: &str, add_special: bool) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode ids back to text, skipping special tokens by default.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> anyhow::Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
    }

    /// Look up a token id by its literal piece.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Expose the raw tokenizer for the chat-template renderer.
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn model_dir() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from("../../models/Qwen3.8-27B-NVFP4");
        p.exists().then_some(p)
    }

    #[test]
    fn loads_real_tokenizer_with_expected_vocab() {
        let Some(dir) = model_dir() else {
            eprintln!("skipping: model not present");
            return;
        };
        let tok = QwenTokenizer::from_model_dir(&dir).unwrap();
        // tokenizer.json holds 248044 base BPE entries + 33 added specials
        // = 248077 usable ids. `config.json` declares vocab_size 248320
        // because the embedding matrix is padded out to a rounder number;
        // ids 248077..248320 are never produced by the tokenizer.
        assert_eq!(tok.vocab_size(), 248077, "vocab size");
        assert_eq!(tok.token_to_id("<|im_end|>"), Some(248046));
        assert_eq!(tok.id_to_token(248076).is_some(), true);
    }

    #[test]
    fn roundtrips_ascii_text() {
        let Some(dir) = model_dir() else { return };
        let tok = QwenTokenizer::from_model_dir(&dir).unwrap();
        let text = "Hello, world! This is a tokenizer round-trip test.";
        let ids = tok.encode(text, false).unwrap();
        assert!(!ids.is_empty());
        let back = tok.decode(&ids, true).unwrap();
        assert_eq!(back, text);
    }

    #[test]
    fn roundtrips_unicode_and_whitespace() {
        let Some(dir) = model_dir() else { return };
        let tok = QwenTokenizer::from_model_dir(&dir).unwrap();
        for text in [
            "你好，世界！GB10 推理引擎",
            "tabs\tand\nnewlines\n\n  trailing spaces   ",
            "emoji 🚀🔥 and math ∑∫≈",
            "numbers 1234567890 and code `fn main() {}`",
        ] {
            let ids = tok.encode(text, false).unwrap();
            let back = tok.decode(&ids, true).unwrap();
            assert_eq!(back, text, "roundtrip failed for {text:?}");
        }
    }

    #[test]
    fn special_tokens_resolve_to_declared_ids() {
        let Some(dir) = model_dir() else { return };
        let tok = QwenTokenizer::from_model_dir(&dir).unwrap();
        assert_eq!(tok.token_to_id("<|im_start|>"), Some(248045));
        assert_eq!(tok.token_to_id("<|im_end|>"), Some(248046));
        assert_eq!(tok.token_to_id("<|endoftext|>"), Some(248044));
        assert_eq!(tok.id_to_token(248045).as_deref(), Some("<|im_start|>"));
    }

    #[test]
    fn eos_set_from_config_is_honoured() {
        let Some(dir) = model_dir() else { return };
        let cfg = ModelConfig::from_file(dir.join("config.json")).unwrap();
        let gen: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("generation_config.json")).unwrap())
                .unwrap();
        let eos = cfg.eos_token_ids_with_generation_config(&gen);
        let tok = QwenTokenizer::from_model_dir(&dir).unwrap().with_eos_ids(eos);
        assert!(tok.is_eos(248046));
        assert!(tok.is_eos(248044));
        assert!(!tok.is_eos(248045));
    }
}
