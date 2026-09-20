//! Multi-Token Prediction head (`mtp_num_hidden_layers: 1`).
//!
//! The head predicts position `t+2` from two things the decoder already has:
//! its hidden state at `t`, and the embedding of the token it emitted at
//! `t+1`. Both are normalised, concatenated to `2 * hidden`, projected back
//! down by `mtp.fc`, and pushed through a single full-attention layer.
//!
//! Every MTP tensor is BF16 -- the quantizer left the head unquantized -- so
//! `Store::linear` routes them to the bf16 GEMV path unchanged.
//!
//! `mtp_use_dedicated_embeddings` is false, so the head shares the decoder's
//! `embed_tokens` and `lm_head` rather than owning copies.

use crate::layer::{FullAttnLayer, Mlp};
use crate::weights::{Linear, Store};
use anyhow::{Context, Result};
use gb10_core::config::TextConfig;
use gb10_cuda::{CudaSlice, Device};

pub struct Mtp {
    /// Norm applied to the next token's embedding before the concatenation.
    pub pre_norm_embed: CudaSlice<f32>,
    /// Norm applied to the decoder's hidden state before the concatenation.
    pub pre_norm_hidden: CudaSlice<f32>,
    /// `[hidden, 2 * hidden]`: the concatenation back down to `hidden`.
    pub fc: Linear,
    /// One full-attention decoder layer (`mtp.layers.0.*`).
    pub layer: FullAttnLayer,
    /// Final norm before the shared `lm_head`.
    pub norm: CudaSlice<f32>,
}

impl Mtp {
    pub fn load(store: &Store, dev: &Device, _cfg: &TextConfig) -> Result<Self> {
        let lp = "mtp.layers.0.";
        Ok(Self {
            pre_norm_embed: store
                .bf16_f32(dev, "mtp.pre_fc_norm_embedding.weight")
                .context("loading mtp.pre_fc_norm_embedding")?,
            pre_norm_hidden: store
                .bf16_f32(dev, "mtp.pre_fc_norm_hidden.weight")
                .context("loading mtp.pre_fc_norm_hidden")?,
            fc: store.linear(dev, "mtp.fc").context("loading mtp.fc")?,
            layer: FullAttnLayer {
                input_ln: store.bf16_f32(dev, &format!("{lp}input_layernorm.weight"))?,
                post_ln: store.bf16_f32(dev, &format!("{lp}post_attention_layernorm.weight"))?,
                q_proj: store.linear(dev, &format!("{lp}self_attn.q_proj"))?,
                k_proj: store.linear(dev, &format!("{lp}self_attn.k_proj"))?,
                v_proj: store.linear(dev, &format!("{lp}self_attn.v_proj"))?,
                o_proj: store.linear(dev, &format!("{lp}self_attn.o_proj"))?,
                q_norm: store.bf16_f32(dev, &format!("{lp}self_attn.q_norm.weight"))?,
                k_norm: store.bf16_f32(dev, &format!("{lp}self_attn.k_norm.weight"))?,
                mlp: Mlp::load(store, dev, lp)?,
            },
            norm: store.bf16_f32(dev, "mtp.norm.weight")?,
        })
    }

    /// Weight bytes the head reads per drafted token. Worth knowing because
    /// the head is unquantized BF16 while the decoder it drafts for is FP4:
    /// it is a meaningful fraction of a full decoder step, which is exactly
    /// what limits the payoff of speculative decoding here.
    pub fn traffic_bytes(&self) -> usize {
        let l = &self.layer;
        let norms = self.pre_norm_embed.len()
            + self.pre_norm_hidden.len()
            + self.norm.len()
            + l.input_ln.len()
            + l.post_ln.len()
            + l.q_norm.len()
            + l.k_norm.len();
        self.fc.traffic_bytes()
            + l.q_proj.traffic_bytes()
            + l.k_proj.traffic_bytes()
            + l.v_proj.traffic_bytes()
            + l.o_proj.traffic_bytes()
            + l.mlp.gate.traffic_bytes()
            + l.mlp.up.traffic_bytes()
            + l.mlp.down.traffic_bytes()
            + norms * 4
    }
}
