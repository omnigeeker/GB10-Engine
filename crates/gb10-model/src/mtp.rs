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

use crate::layer::{FullAttnLayer, LayerState, Mlp, Scratch};
use crate::weights::{Linear, Store};
use anyhow::{Context, Result};
use gb10_core::config::TextConfig;
use gb10_cuda::{CudaSlice, Device};

/// Per-call scratch for the head. Sized once; the norm/attention buffers are
/// the same shape as the decoder's because `mtp.layers.0` has the decoder's
/// full-attention geometry.
pub struct MtpState {
    /// Gathered embedding of the token at `t+1`.
    embed: CudaSlice<f32>,
    /// The two normalised inputs, and the concatenation that feeds `mtp.fc`.
    ne: CudaSlice<f32>,
    nh: CudaSlice<f32>,
    fc_in: CudaSlice<f32>,
    /// `fc` output, the attention layer's output, and the final norm.
    x: CudaSlice<f32>,
    /// Layer output; the draft chain feeds this back as the next hidden.
    pub out: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    pub logits: CudaSlice<f32>,
    /// The head keeps its own KV cache; it must not share the decoder's.
    pub layer_state: LayerState,
    pub scratch: Scratch,
}

impl MtpState {
    pub fn new(dev: &Device, cfg: &TextConfig, max_seq: usize, vocab: usize) -> Result<Self> {
        let h = cfg.hidden_size;
        let z = |n: usize| -> Result<CudaSlice<f32>> { Ok(dev.stream().alloc_zeros::<f32>(n)?) };
        let kvd = max_seq * cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            embed: z(h)?,
            ne: z(h)?,
            nh: z(h)?,
            fc_in: z(2 * h)?,
            x: z(h)?,
            out: z(h)?,
            normed: z(h)?,
            logits: z(vocab)?,
            layer_state: LayerState {
                conv_hist: z(1)?,
                rec: z(1)?,
                k_cache: z(kvd)?,
                v_cache: z(kvd)?,
                n_seq: 1,
                n_keys: vec![0],
                positions: dev.stream().alloc_zeros::<i32>(1)?,
            },
            scratch: Scratch::new(dev, cfg, max_seq)?,
        })
    }
}

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

    /// Drafts position `t+2` from the decoder's hidden state at `t` and the
    /// token the decoder emitted at `t+1`. Leaves the result in `st.logits`;
    /// the caller runs the shared `lm_head` (or not) as it sees fit.
    ///
    /// `hidden_t` is the post-final-norm hidden state the decoder already
    /// computed for position `t`, which is why drafting is so cheap: the head
    /// never re-reads the decoder's 17.6 GB of weights.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        dev: &Device,
        cfg: &TextConfig,
        hidden_t: &CudaSlice<f32>,
        next_token: u32,
        embed: &CudaSlice<u16>,
        lm_head: &Linear,
        st: &mut MtpState,
    ) -> Result<()> {
        let ops = dev.ops();
        let eps = cfg.rms_norm_eps as f32;
        let h = cfg.hidden_size;

        ops.embed_gather(dev, embed, next_token, &mut st.embed, h)?;
        ops.rmsnorm_zero_centered(dev, &st.embed, &self.pre_norm_embed, &mut st.ne, 1, h, eps)?;
        ops.rmsnorm_zero_centered(dev, hidden_t, &self.pre_norm_hidden, &mut st.nh, 1, h, eps)?;
        ops.concat2(dev, &st.ne, &st.nh, &mut st.fc_in, h)?;
        self.fc.forward(dev, &st.fc_in, &mut st.x, 1)?;
        self.layer
            .forward(dev, cfg, &st.x, &mut st.out, &mut st.layer_state, &mut st.scratch, 0)?;
        ops.rmsnorm_zero_centered(dev, &st.out, &self.norm, &mut st.normed, 1, h, eps)?;
        lm_head.forward(dev, &st.normed, &mut st.logits, 1)?;
        Ok(())
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
