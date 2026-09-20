//! Decoder-layer weights and the decode-path forward pass.
//!
//! The forward here is the **decode** path: one token at a time, with the
//! Gated DeltaNet recurrent state and the attention KV cache carried between
//! steps. Running a whole prompt through it token by token reproduces the
//! reference's chunked prefill exactly — verified numerically in
//! `docs/ARCHITECTURE.md` and by the `layer-parity` gate — so it is also the
//! correctness path. Batched prefill is a throughput optimisation (M6), not a
//! different numerical result.

use anyhow::{Context, Result};
use gb10_core::config::TextConfig;
use gb10_cuda::{CudaSlice, Device};
use gb10_cuda::ops::{DELTA_KEY_HEAD_DIM, DELTA_VALUE_HEAD_DIM};

use crate::rope::{rope_tables, rope_tables_range};
use crate::weights::{Linear, Store};

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl Mlp {
    fn load(store: &Store, dev: &Device, p: &str) -> Result<Self> {
        Ok(Self {
            gate: store.linear(dev, &format!("{p}mlp.gate_proj"))?,
            up: store.linear(dev, &format!("{p}mlp.up_proj"))?,
            down: store.linear(dev, &format!("{p}mlp.down_proj"))?,
        })
    }

    /// Reads `x`, writes `out`. `a`/`b` are scratch of `intermediate_size`.
    fn forward(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        a: &mut CudaSlice<f32>,
        b: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<()> {
        self.gate.forward(dev, x, a, batch)?;
        self.up.forward(dev, x, b, batch)?;
        // In-place is safe: element i is read before it is written.
        dev.ops().swiglu_inplace(dev, a, b, self.gate.n * batch)?;
        self.down.forward(dev, a, out, batch)?;
        Ok(())
    }

    /// Batched prefill: all `t` tokens in one GEMM per projection.
    fn forward_prefill(
        &self,
        dev: &Device,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        a: &mut CudaSlice<f32>,
        b: &mut CudaSlice<f32>,
        t: usize,
    ) -> Result<()> {
        self.gate.forward_prefill(dev, x, a, t)?;
        self.up.forward_prefill(dev, x, b, t)?;
        dev.ops().swiglu_inplace(dev, a, b, self.gate.n * t)?;
        self.down.forward_prefill(dev, a, out, t)?;
        Ok(())
    }

    fn traffic_bytes(&self) -> usize {
        self.gate.traffic_bytes() + self.up.traffic_bytes() + self.down.traffic_bytes()
    }
}

/// Gated DeltaNet (linear attention) layer.
pub struct DeltaNetLayer {
    pub input_ln: CudaSlice<f32>,
    pub post_ln: CudaSlice<f32>,
    pub in_proj_qkv: Linear,
    pub in_proj_z: Linear,
    pub in_proj_a: Linear,
    pub in_proj_b: Linear,
    pub conv1d: CudaSlice<f32>,
    pub a_log: CudaSlice<f32>,
    pub dt_bias: CudaSlice<f32>,
    pub norm: CudaSlice<f32>,
    pub out_proj: Linear,
    pub mlp: Mlp,
}

/// Full (softmax) attention layer.
pub struct FullAttnLayer {
    pub input_ln: CudaSlice<f32>,
    pub post_ln: CudaSlice<f32>,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: CudaSlice<f32>,
    pub k_norm: CudaSlice<f32>,
    pub mlp: Mlp,
}

pub enum Layer {
    Delta(DeltaNetLayer),
    Attn(FullAttnLayer),
}

impl Layer {
    pub fn is_delta(&self) -> bool {
        matches!(self, Layer::Delta(_))
    }

    pub fn traffic_bytes(&self) -> usize {
        match self {
            Layer::Delta(l) => {
                l.in_proj_qkv.traffic_bytes()
                    + l.in_proj_z.traffic_bytes()
                    + l.in_proj_a.traffic_bytes()
                    + l.in_proj_b.traffic_bytes()
                    + l.out_proj.traffic_bytes()
                    + l.mlp.traffic_bytes()
            }
            Layer::Attn(l) => {
                l.q_proj.traffic_bytes()
                    + l.k_proj.traffic_bytes()
                    + l.v_proj.traffic_bytes()
                    + l.o_proj.traffic_bytes()
                    + l.mlp.traffic_bytes()
            }
        }
    }

    pub fn load(store: &Store, dev: &Device, cfg: &TextConfig, idx: usize) -> Result<Self> {
        let p = format!("layers.{idx}.");
        let input_ln = store.bf16_f32(dev, &format!("{p}input_layernorm.weight"))?;
        let post_ln = store.bf16_f32(dev, &format!("{p}post_attention_layernorm.weight"))?;
        let mlp = Mlp::load(store, dev, &p)?;

        Ok(match cfg.layer_types[idx] {
            gb10_core::config::LayerType::LinearAttention => Layer::Delta(DeltaNetLayer {
                input_ln,
                post_ln,
                in_proj_qkv: store.linear(dev, &format!("{p}linear_attn.in_proj_qkv"))?,
                in_proj_z: store.linear(dev, &format!("{p}linear_attn.in_proj_z"))?,
                in_proj_a: store.linear(dev, &format!("{p}linear_attn.in_proj_a"))?,
                in_proj_b: store.linear(dev, &format!("{p}linear_attn.in_proj_b"))?,
                conv1d: store.bf16_f32(dev, &format!("{p}linear_attn.conv1d.weight"))?,
                a_log: store.bf16_f32(dev, &format!("{p}linear_attn.A_log"))?,
                dt_bias: store.bf16_f32(dev, &format!("{p}linear_attn.dt_bias"))?,
                norm: store.bf16_f32(dev, &format!("{p}linear_attn.norm.weight"))?,
                out_proj: store.linear(dev, &format!("{p}linear_attn.out_proj"))?,
                mlp,
            }),
            gb10_core::config::LayerType::FullAttention => Layer::Attn(FullAttnLayer {
                input_ln,
                post_ln,
                q_proj: store.linear(dev, &format!("{p}self_attn.q_proj"))?,
                k_proj: store.linear(dev, &format!("{p}self_attn.k_proj"))?,
                v_proj: store.linear(dev, &format!("{p}self_attn.v_proj"))?,
                o_proj: store.linear(dev, &format!("{p}self_attn.o_proj"))?,
                q_norm: store.bf16_f32(dev, &format!("{p}self_attn.q_norm.weight"))?,
                k_norm: store.bf16_f32(dev, &format!("{p}self_attn.k_norm.weight"))?,
                mlp,
            }),
        })
    }

    /// One decode step. `x` is the incoming residual stream, `out` receives the
    /// layer output. `state` persists across steps.
    pub fn forward(
        &self,
        dev: &Device,
        cfg: &TextConfig,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        state: &mut LayerState,
        sc: &mut Scratch,
    ) -> Result<()> {
        match self {
            Layer::Delta(l) => l.forward(dev, cfg, x, out, state, sc),
            Layer::Attn(l) => l.forward(dev, cfg, x, out, state, sc),
        }
    }

    /// Batched prefill over `t` tokens; `x` is `[t, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill(
        &self,
        dev: &Device,
        cfg: &TextConfig,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        state: &mut LayerState,
        sc: &mut Scratch,
        t: usize,
    ) -> Result<()> {
        match self {
            Layer::Delta(l) => l.forward_prefill(dev, cfg, x, out, state, sc, t),
            Layer::Attn(l) => l.forward_prefill(dev, cfg, x, out, state, sc, t),
        }
    }
}

impl DeltaNetLayer {
    /// Batched prefill over `t` prompt tokens. `x` is `[t, hidden]`.
    ///
    /// Everything except the recurrence is parallel across `t`; the
    /// recurrence runs as a `t`-iteration loop inside one launch, so the
    /// 3.1 MB state is read and written once for the whole prompt instead of
    /// once per token.
    #[allow(clippy::too_many_arguments)]
    fn forward_prefill(
        &self,
        dev: &Device,
        cfg: &TextConfig,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        state: &mut LayerState,
        sc: &mut Scratch,
        t: usize,
    ) -> Result<()> {
        let ops = dev.ops();
        let eps = cfg.rms_norm_eps as f32;
        let hidden = cfg.hidden_size;
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let qk_dim = cfg.linear_qk_dim();
        let conv_dim = qk_dim * 2 + cfg.linear_value_dim();
        let group = nv / nk;

        ops.rmsnorm_zero_centered(dev, x, &self.input_ln, &mut sc.hidden, t, hidden, eps)?;
        self.in_proj_qkv.forward_prefill(dev, &sc.hidden, &mut sc.qkv, t)?;
        self.in_proj_z.forward_prefill(dev, &sc.hidden, &mut sc.z, t)?;
        self.in_proj_a.forward_prefill(dev, &sc.hidden, &mut sc.a, t)?;
        self.in_proj_b.forward_prefill(dev, &sc.hidden, &mut sc.b, t)?;

        ops.conv1d_prefill_silu(
            dev,
            &sc.qkv,
            &self.conv1d,
            &mut state.conv_hist,
            &mut sc.conv,
            conv_dim,
            t,
        )?;

        dev.stream().memcpy_dtod(&sc.conv, &mut sc.conv_ln)?;
        let q_scale = 1.0 / (kd as f32).sqrt();
        ops.l2norm_scale_batched(dev, &mut sc.conv_ln, conv_dim, 0, nk, kd, t, q_scale, 1e-6)?;
        ops.l2norm_scale_batched(dev, &mut sc.conv_ln, conv_dim, qk_dim, nk, kd, t, 1.0, 1e-6)?;

        ops.delta_gate_batched(
            dev,
            &sc.a,
            &sc.b,
            &self.a_log,
            &self.dt_bias,
            &mut sc.decay,
            &mut sc.beta,
            nv,
            t,
        )?;

        ops.gated_delta_rule_chunk(
            dev,
            &sc.conv_ln,
            0,
            qk_dim,
            2 * qk_dim,
            conv_dim,
            &sc.decay,
            &sc.beta,
            &mut state.rec,
            &mut sc.attn,
            t,
            nv,
            nk,
            group,
        )?;

        ops.rmsnorm_gated(dev, &sc.attn, &sc.z, &self.norm, &mut sc.gnorm, t * nv, vd, eps)?;
        self.out_proj.forward_prefill(dev, &sc.gnorm, &mut sc.proj, t)?;

        ops.add(dev, x, &sc.proj, &mut sc.res, t * hidden)?;
        ops.rmsnorm_zero_centered(dev, &sc.res, &self.post_ln, &mut sc.mlp_in, t, hidden, eps)?;
        self.mlp
            .forward_prefill(dev, &sc.mlp_in, &mut sc.down, &mut sc.inter, &mut sc.inter2, t)?;
        ops.add(dev, &sc.res, &sc.down, out, t * hidden)?;
        Ok(())
    }
}

impl FullAttnLayer {
    /// Batched prefill over `t` prompt tokens. `x` is `[t, hidden]`.
    ///
    /// Valid only when the cache is empty, because `attn_prefill_kernel`
    /// hardcodes the causal window as `0..=t` rather than `0..=start+t`.
    #[allow(clippy::too_many_arguments)]
    fn forward_prefill(
        &self,
        dev: &Device,
        cfg: &TextConfig,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        state: &mut LayerState,
        sc: &mut Scratch,
        t: usize,
    ) -> Result<()> {
        let ops = dev.ops();
        let eps = cfg.rms_norm_eps as f32;
        let hidden = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let rotary = cfg.rotary_dim();

        ops.rmsnorm_zero_centered(dev, x, &self.input_ln, &mut sc.hidden, t, hidden, eps)?;
        self.q_proj.forward_prefill(dev, &sc.hidden, &mut sc.fused, t)?;
        self.k_proj.forward_prefill(dev, &sc.hidden, &mut sc.kb, t)?;
        self.v_proj.forward_prefill(dev, &sc.hidden, &mut sc.vb, t)?;

        ops.deinterleave_heads_batched(dev, &sc.fused, &mut sc.q, nh, hd, 0, t)?;
        ops.deinterleave_heads_batched(dev, &sc.fused, &mut sc.gate, nh, hd, hd, t)?;

        ops.rmsnorm_zero_centered_inplace(dev, &mut sc.q, &self.q_norm, t * nh, hd, eps)?;
        ops.rmsnorm_zero_centered(dev, &sc.kb, &self.k_norm, &mut sc.kb_ln, t * nkv, hd, eps)?;

        let pos = state.n_keys;
        let (cos, sin) = rope_tables_range(cfg, pos, t);
        dev.stream().memcpy_htod(&cos, &mut sc.cos)?;
        dev.stream().memcpy_htod(&sin, &mut sc.sin)?;
        ops.rope_neox_batched(
            dev,
            &mut sc.q,
            &mut sc.kb_ln,
            &sc.cos,
            &sc.sin,
            nh,
            nkv,
            hd,
            rotary / 2,
            t,
        )?;

        ops.kv_cache_append_batched(
            dev,
            &sc.kb_ln,
            &sc.vb,
            &mut state.k_cache,
            &mut state.v_cache,
            pos,
            nkv,
            hd,
            t,
        )?;
        state.n_keys = pos + t;

        let scale = 1.0 / (hd as f32).sqrt();
        ops.attn_prefill(
            dev,
            &sc.q,
            &state.k_cache,
            &state.v_cache,
            &mut sc.attn,
            state.n_keys,
            nh,
            nkv,
            hd,
            scale,
        )?;

        ops.sigmoid_mul(dev, &mut sc.attn, &sc.gate, t * nh * hd)?;
        self.o_proj.forward_prefill(dev, &sc.attn, &mut sc.proj, t)?;

        ops.add(dev, x, &sc.proj, &mut sc.res, t * hidden)?;
        ops.rmsnorm_zero_centered(dev, &sc.res, &self.post_ln, &mut sc.mlp_in, t, hidden, eps)?;
        self.mlp
            .forward_prefill(dev, &sc.mlp_in, &mut sc.down, &mut sc.inter, &mut sc.inter2, t)?;
        ops.add(dev, &sc.res, &sc.down, out, t * hidden)?;
        Ok(())
    }
}

/// Per-sequence state. For DeltaNet layers this is the conv history plus the
/// fp32 `[48, 128, 128]` recurrent state (3.1 MB per layer); for attention
/// layers it is the KV cache.
pub struct LayerState {
    pub conv_hist: CudaSlice<f32>,
    pub rec: CudaSlice<f32>,
    pub k_cache: CudaSlice<f32>,
    pub v_cache: CudaSlice<f32>,
    pub n_keys: usize,
}

impl LayerState {
    pub fn new(dev: &Device, cfg: &TextConfig, layer: &Layer, max_seq: usize) -> Result<Self> {
        let zeros = |n: usize| -> Result<CudaSlice<f32>> {
            Ok(dev.stream().alloc_zeros::<f32>(n)?)
        };
        match layer {
            Layer::Delta(_) => {
                let conv_dim = cfg.linear_qk_dim() * 2 + cfg.linear_value_dim();
                Ok(Self {
                    conv_hist: zeros(conv_dim * (cfg.linear_conv_kernel_dim - 1))?,
                    rec: zeros(cfg.linear_num_value_heads * DELTA_KEY_HEAD_DIM * DELTA_VALUE_HEAD_DIM)?,
                    k_cache: zeros(1)?,
                    v_cache: zeros(1)?,
                    n_keys: 0,
                })
            }
            Layer::Attn(_) => {
                let n = max_seq * cfg.num_key_value_heads * cfg.head_dim;
                Ok(Self {
                    conv_hist: zeros(1)?,
                    rec: zeros(1)?,
                    k_cache: zeros(n)?,
                    v_cache: zeros(n)?,
                    n_keys: 0,
                })
            }
        }
    }

    pub fn reset(&mut self, dev: &Device) -> Result<()> {
        for b in [
            &mut self.conv_hist,
            &mut self.rec,
            &mut self.k_cache,
            &mut self.v_cache,
        ] {
            dev.stream().memset_zeros(b)?;
        }
        self.n_keys = 0;
        Ok(())
    }
}

/// Reusable activation buffers, sized for the widest tensor in the model.
pub struct Scratch {
    pub hidden: CudaSlice<f32>,
    /// MLP input: `post_attention_layernorm` of the residual. Separate from
    /// `hidden` so the `input_layernorm` output stays inspectable.
    pub mlp_in: CudaSlice<f32>,
    pub fused: CudaSlice<f32>,
    pub qkv: CudaSlice<f32>,
    pub conv: CudaSlice<f32>,
    pub conv_ln: CudaSlice<f32>,
    pub down: CudaSlice<f32>,
    pub z: CudaSlice<f32>,
    pub q: CudaSlice<f32>,
    pub gate: CudaSlice<f32>,
    pub kb: CudaSlice<f32>,
    /// k_proj after `k_norm` + RoPE. Kept separate from `kb` so the raw
    /// projection stays comparable in the parity gate.
    pub kb_ln: CudaSlice<f32>,
    pub vb: CudaSlice<f32>,
    pub attn: CudaSlice<f32>,
    pub gnorm: CudaSlice<f32>,
    pub proj: CudaSlice<f32>,
    pub res: CudaSlice<f32>,
    pub a: CudaSlice<f32>,
    pub b: CudaSlice<f32>,
    pub decay: CudaSlice<f32>,
    pub beta: CudaSlice<f32>,
    pub inter: CudaSlice<f32>,
    pub inter2: CudaSlice<f32>,
    pub cos: CudaSlice<f32>,
    pub sin: CudaSlice<f32>,
}

impl Scratch {
    /// One token's worth of scratch, for the per-token parity gate and decode.
    pub fn new_single(dev: &Device, cfg: &TextConfig) -> Result<Self> {
        Self::new(dev, cfg, 1)
    }

    pub fn new(dev: &Device, cfg: &TextConfig, max_seq: usize) -> Result<Self> {
        // Every buffer below is per-token, so prefill needs `max_seq` copies
        // of each. At 512 tokens that is ~300 MB, which is cheap next to the
        // 17 GB of weights and buys a single batched pass over the prompt.
        let z = |n: usize| -> Result<CudaSlice<f32>> {
            Ok(dev.stream().alloc_zeros::<f32>(n * max_seq)?)
        };
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let vdim = cfg.linear_value_dim();
        let nv = cfg.linear_num_value_heads;
        Ok(Self {
            hidden: z(hidden)?,
            mlp_in: z(hidden)?,
            fused: z(cfg.q_proj_out_features())?,
            qkv: z(cfg.linear_qk_dim() * 2 + vdim)?,
            conv: z(cfg.linear_qk_dim() * 2 + vdim)?,
            conv_ln: z(cfg.linear_qk_dim() * 2 + vdim)?,
            down: z(hidden)?,
            z: z(vdim)?,
            q: z(cfg.num_attention_heads * cfg.head_dim)?,
            gate: z(cfg.num_attention_heads * cfg.head_dim)?,
            kb: z(cfg.num_key_value_heads * cfg.head_dim)?,
            kb_ln: z(cfg.num_key_value_heads * cfg.head_dim)?,
            vb: z(cfg.num_key_value_heads * cfg.head_dim)?,
            attn: z(vdim)?,
            gnorm: z(vdim)?,
            proj: z(hidden)?,
            res: z(hidden)?,
            a: z(nv)?,
            b: z(nv)?,
            decay: z(nv)?,
            beta: z(nv)?,
            inter: z(inter)?,
            inter2: z(inter)?,
            cos: z(cfg.rotary_dim() / 2)?,
            sin: z(cfg.rotary_dim() / 2)?,
        })
    }
}

impl DeltaNetLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        dev: &Device,
        cfg: &TextConfig,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        state: &mut LayerState,
        sc: &mut Scratch,
    ) -> Result<()> {
        let ops = dev.ops();
        let eps = cfg.rms_norm_eps as f32;
        let hidden = cfg.hidden_size;
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let qk_dim = cfg.linear_qk_dim();
        let vdim = cfg.linear_value_dim();
        let conv_dim = qk_dim * 2 + vdim;
        let group = nv / nk;

        ops.rmsnorm_zero_centered(dev, x, &self.input_ln, &mut sc.hidden, 1, hidden, eps)?;

        self.in_proj_qkv.forward(dev, &sc.hidden, &mut sc.qkv, 1)?;
        self.in_proj_z.forward(dev, &sc.hidden, &mut sc.z, 1)?;
        self.in_proj_a.forward(dev, &sc.hidden, &mut sc.a, 1)?;
        self.in_proj_b.forward(dev, &sc.hidden, &mut sc.b, 1)?;

        // Causal depthwise conv + SiLU, updating the 3-token history in place.
        ops.conv1d_step_silu(
            dev,
            &sc.qkv,
            &self.conv1d,
            &mut state.conv_hist,
            &mut sc.conv,
            conv_dim,
        )?;

        // L2-normalise q and k. The reference applies head_dim^-0.5 to the
        // QUERY ONLY (`query = query / query.shape[-1]**0.5` in
        // torch_recurrent_gated_delta_rule); the key is normalised and left
        // alone. Scaling both is a silent 1/sqrt(128) error on every key.
        dev.stream().memcpy_dtod(&sc.conv, &mut sc.conv_ln)?;
        let q_scale = 1.0 / (kd as f32).sqrt();
        ops.l2norm_scale(dev, &mut sc.conv_ln, 0, nk, kd, q_scale, 1e-6)?;
        ops.l2norm_scale(dev, &mut sc.conv_ln, qk_dim, nk, kd, 1.0, 1e-6)?;

        ops.delta_gate(dev, &sc.a, &sc.b, &self.a_log, &self.dt_bias, &mut sc.decay, &mut sc.beta, nv)?;

        // Recurrence. q/k/v are contiguous regions of the conv output.
        ops.gated_delta_rule_step(
            dev,
            &sc.conv_ln,
            0,
            qk_dim,
            2 * qk_dim,
            conv_dim,
            &sc.decay,
            &sc.beta,
            &mut state.rec,
            &mut sc.attn,
            1,
            nv,
            nk,
            group,
        )?;

        // Gated RMSNorm over each value head, then out_proj. Written to a
        // separate buffer so `sc.attn` keeps the raw recurrence output, which
        // the stage-parity gate compares against the reference.
        ops.rmsnorm_gated(dev, &sc.attn, &sc.z, &self.norm, &mut sc.gnorm, nv, vd, eps)?;
        self.out_proj.forward(dev, &sc.gnorm, &mut sc.proj, 1)?;

        ops.add(dev, x, &sc.proj, &mut sc.res, hidden)?;
        // The MLP reads the *normalised* residual, not the residual itself.
        ops.rmsnorm_zero_centered(dev, &sc.res, &self.post_ln, &mut sc.mlp_in, 1, hidden, eps)?;
        self.mlp
            .forward(dev, &sc.mlp_in, &mut sc.down, &mut sc.inter, &mut sc.inter2, 1)?;
        ops.add(dev, &sc.res, &sc.down, out, hidden)?;
        Ok(())
    }
}

impl FullAttnLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        dev: &Device,
        cfg: &TextConfig,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        state: &mut LayerState,
        sc: &mut Scratch,
    ) -> Result<()> {
        let ops = dev.ops();
        let eps = cfg.rms_norm_eps as f32;
        let hidden = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let rotary = cfg.rotary_dim();

        ops.rmsnorm_zero_centered(dev, x, &self.input_ln, &mut sc.hidden, 1, hidden, eps)?;

        self.q_proj.forward(dev, &sc.hidden, &mut sc.fused, 1)?;
        self.k_proj.forward(dev, &sc.hidden, &mut sc.kb, 1)?;
        self.v_proj.forward(dev, &sc.hidden, &mut sc.vb, 1)?;

        // q_proj emits [n_heads, 2*head_dim]; head h holds query then gate.
        ops.deinterleave_heads(dev, &sc.fused, &mut sc.q, nh, hd, 0)?;
        ops.deinterleave_heads(dev, &sc.fused, &mut sc.gate, nh, hd, hd)?;

        // Per-head RMSNorm over head_dim (zero-centered), then RoPE on the
        // first `rotary_dim` channels of each head.
        ops.rmsnorm_zero_centered_inplace(dev, &mut sc.q, &self.q_norm, nh, hd, eps)?;
        ops.rmsnorm_zero_centered(dev, &sc.kb, &self.k_norm, &mut sc.kb_ln, nkv, hd, eps)?;

        // RoPE on the first `rotary` channels of each head.
        let pos = state.n_keys;
        let (cos, sin) = rope_tables(cfg, pos);
        dev.stream().memcpy_htod(&cos, &mut sc.cos)?;
        dev.stream().memcpy_htod(&sin, &mut sc.sin)?;
        ops.rope_neox(dev, &mut sc.q, &mut sc.kb_ln, &sc.cos, &sc.sin, nh, nkv, hd, rotary / 2)?;

        ops.kv_cache_append(
            dev,
            &sc.kb_ln,
            &sc.vb,
            &mut state.k_cache,
            &mut state.v_cache,
            pos,
            nkv,
            hd,
        )?;
        state.n_keys = pos + 1;

        let scale = 1.0 / (hd as f32).sqrt();
        ops.attn_decode(
            dev,
            &sc.q,
            &state.k_cache,
            &state.v_cache,
            &mut sc.attn,
            state.n_keys,
            nh,
            nkv,
            hd,
            scale,
        )?;

        // The output gate is sigmoid, NOT the swish in config.json.
        ops.sigmoid_mul(dev, &mut sc.attn, &sc.gate, nh * hd)?;
        self.o_proj.forward(dev, &sc.attn, &mut sc.proj, 1)?;

        ops.add(dev, x, &sc.proj, &mut sc.res, hidden)?;
        // The MLP reads the *normalised* residual, not the residual itself.
        ops.rmsnorm_zero_centered(dev, &sc.res, &self.post_ln, &mut sc.mlp_in, 1, hidden, eps)?;
        self.mlp
            .forward(dev, &sc.mlp_in, &mut sc.down, &mut sc.inter, &mut sc.inter2, 1)?;
        ops.add(dev, &sc.res, &sc.down, out, hidden)?;
        Ok(())
    }
}

impl Store {
    /// Load one decoder layer by index.
    pub fn layer(&self, dev: &Device, cfg: &TextConfig, idx: usize) -> Result<Layer> {
        Layer::load(self, dev, cfg, idx)
            .with_context(|| format!("loading decoder layer {idx}"))
    }
}
