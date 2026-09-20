//! The full Qwen3.5 model: embedding gather, 64 decoder layers, final norm and
//! the NVFP4 `lm_head`, driven one token at a time.
//!
//! Decode is memory-bandwidth-bound: each token streams ~17.6 GB of quantized
//! weights (`docs/PHYSICS.md`), so the only thing that matters for throughput
//! is that every weight byte is read exactly once per token and that the
//! kernels stay coalesced. Nothing here materialises a dequantized weight.

use anyhow::{Context, Result};
use gb10_core::config::ModelConfig;
use gb10_cuda::{CudaSlice, Device};

use crate::layer::{Layer, LayerState, Scratch};
use crate::weights::{Linear, Store};

/// Weight prefix for the text decoder stack.
const TEXT: &str = "model.language_model.";

pub struct Model {
    pub cfg: ModelConfig,
    /// `embed_tokens` stays bf16 in device memory (2.5 GB). Widen on gather.
    pub embed: CudaSlice<u16>,
    pub layers: Vec<Layer>,
    /// Final zero-centered RMSNorm.
    pub norm: CudaSlice<f32>,
    pub lm_head: Linear,
}

impl Model {
    pub fn load(dev: &Device, cfg: ModelConfig) -> Result<Self> {
        Self::load_from(dev, cfg, "models/Qwen3.8-27B-NVFP4")
    }

    pub fn load_from(
        dev: &Device,
        cfg: ModelConfig,
        dir: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let store = Store::open(dir, TEXT)?;
        let text = &cfg.text_config;

        let embed = store
            .bf16_u16(dev, "embed_tokens.weight")
            .context("loading embed_tokens")?;
        let norm = store
            .bf16_f32(dev, "norm.weight")
            .context("loading final norm")?;

        // lm_head sits at the top level, outside the language_model prefix,
        // and is quantized NVFP4 (tie_word_embeddings is false).
        let top = Store::open(dir, "")?;
        let lm_head = top.linear(dev, "lm_head").context("loading lm_head")?;

        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for i in 0..text.num_hidden_layers {
            layers.push(store.layer(dev, text, i)?);
        }

        Ok(Self {
            cfg,
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn text(&self) -> &gb10_core::config::TextConfig {
        &self.cfg.text_config
    }

    /// Sum of the weight bytes the decoder reads per token, excluding the
    /// embedding gather (which touches one row) and the vision tower.
    pub fn traffic_bytes(&self) -> usize {
        let layers: usize = self.layers.iter().map(|l| l.traffic_bytes()).sum();
        layers + self.lm_head.traffic_bytes()
    }

    pub fn vocab_size(&self) -> usize {
        self.lm_head.n
    }
}

/// Everything that persists across decode steps for one sequence.
pub struct ModelState {
    pub layers: Vec<LayerState>,
    pub logits: CudaSlice<f32>,
    pub idx: CudaSlice<i32>,
    /// Ping-pong residual-stream buffers.
    pub a: CudaSlice<f32>,
    pub b: CudaSlice<f32>,
    pub normed: CudaSlice<f32>,
    /// Final-norm row of the last prompt token, consumed by `lm_head`.
    pub last: CudaSlice<f32>,
    pub n_tokens: usize,
}

impl ModelState {
    pub fn new(dev: &Device, model: &Model, max_seq: usize) -> Result<Self> {
        let text = model.text();
        let mut layers = Vec::with_capacity(model.layers.len());
        for l in &model.layers {
            layers.push(LayerState::new(dev, text, l, max_seq)?);
        }
        let z = |n: usize| -> Result<CudaSlice<f32>> { Ok(dev.stream().alloc_zeros::<f32>(n)?) };
        Ok(Self {
            layers,
            logits: z(model.vocab_size())?,
            idx: dev.stream().alloc_zeros::<i32>(1)?,
            // Per-token buffers are `max_seq` rows so prefill can run the
            // whole prompt through the stack in one batched pass.
            a: z(text.hidden_size * max_seq)?,
            b: z(text.hidden_size * max_seq)?,
            normed: z(text.hidden_size * max_seq)?,
            last: z(text.hidden_size)?,
            n_tokens: 0,
        })
    }

    /// Drop all context: KV caches, conv history and recurrent state.
    pub fn reset(&mut self, dev: &Device) -> Result<()> {
        for l in &mut self.layers {
            l.reset(dev)?;
        }
        self.n_tokens = 0;
        Ok(())
    }
}

impl Model {
    /// One decode step: consume `token`, return the greedy next token id.
    pub fn step(
        &self,
        dev: &Device,
        token: u32,
        state: &mut ModelState,
        sc: &mut Scratch,
    ) -> Result<u32> {
        let text = self.text();
        let hidden = text.hidden_size;
        let eps = text.rms_norm_eps as f32;

        dev.ops()
            .embed_gather(dev, &self.embed, token, &mut state.a, hidden)?;

        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(dev, text, &state.a, &mut state.b, &mut state.layers[i], sc)?;
            std::mem::swap(&mut state.a, &mut state.b);
        }

        dev.ops().rmsnorm_zero_centered(
            dev,
            &state.a,
            &self.norm,
            &mut state.normed,
            1,
            hidden,
            eps,
        )?;
        self.lm_head
            .forward(dev, &state.normed, &mut state.logits, 1)?;
        dev.ops()
            .argmax(dev, &state.logits, &mut state.idx, self.vocab_size())?;

        state.n_tokens += 1;
        let v = dev.stream().memcpy_dtov(&state.idx)?;
        Ok(v[0] as u32)
    }

    /// Feed a prompt, return the greedy token after the last prompt token.
    pub fn prefill(
        &self,
        dev: &Device,
        tokens: &[u32],
        state: &mut ModelState,
        sc: &mut Scratch,
    ) -> Result<u32> {
        let t = tokens.len();
        if t == 0 {
            anyhow::bail!("prefill: empty prompt");
        }
        let text = self.text();
        let hidden = text.hidden_size;
        let eps = text.rms_norm_eps as f32;

        let ids: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        let ids_dev = dev.stream().clone_htod(&ids)?;
        dev.ops()
            .embed_gather_batched(dev, &self.embed, &ids_dev, &mut state.a, hidden, t)?;

        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_prefill(dev, text, &state.a, &mut state.b, &mut state.layers[i], sc, t)?;
            std::mem::swap(&mut state.a, &mut state.b);
        }

        // Only the final prompt row needs logits; running lm_head over all `t`
        // rows would re-read 715 MB of weights per prompt token.
        dev.ops().rmsnorm_zero_centered(
            dev,
            &state.a,
            &self.norm,
            &mut state.normed,
            t,
            hidden,
            eps,
        )?;
        dev.ops()
            .copy_last_row(dev, &state.normed, &mut state.last, t, hidden)?;
        self.lm_head.forward(dev, &state.last, &mut state.logits, 1)?;
        dev.ops()
            .argmax(dev, &state.logits, &mut state.idx, self.vocab_size())?;
        state.n_tokens += t;
        let v = dev.stream().memcpy_dtov(&state.idx)?;
        Ok(v[0] as u32)
    }
}

/// Per-phase wall time for one decode step, in milliseconds.
///
/// Every phase is synchronised before it is timed, so this measures the GPU
/// rather than the queue. It is a diagnostic for finding where the gap to the
/// bandwidth roofline lives, not a fast path.
#[derive(Default, Debug, Clone, Copy)]
pub struct PhaseTimes {
    pub embed: f64,
    pub delta_layers: f64,
    pub attn_layers: f64,
    pub final_norm: f64,
    pub lm_head: f64,
    pub argmax: f64,
}

impl PhaseTimes {
    pub fn total(&self) -> f64 {
        self.embed
            + self.delta_layers
            + self.attn_layers
            + self.final_norm
            + self.lm_head
            + self.argmax
    }
}

impl Model {
    /// `step`, with each phase timed. Only for `--profile`.
    pub fn step_timed(
        &self,
        dev: &Device,
        token: u32,
        state: &mut ModelState,
        sc: &mut Scratch,
    ) -> Result<(u32, PhaseTimes)> {
        let text = self.text();
        let hidden = text.hidden_size;
        let eps = text.rms_norm_eps as f32;
        let mut pt = PhaseTimes::default();
        let mut t = std::time::Instant::now();

        dev.ops()
            .embed_gather(dev, &self.embed, token, &mut state.a, hidden)?;
        dev.synchronize()?;
        pt.embed = t.elapsed().as_secs_f64() * 1e3;

        for (i, layer) in self.layers.iter().enumerate() {
            t = std::time::Instant::now();
            layer.forward(dev, text, &state.a, &mut state.b, &mut state.layers[i], sc)?;
            dev.synchronize()?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if layer.is_delta() {
                pt.delta_layers += ms;
            } else {
                pt.attn_layers += ms;
            }
            std::mem::swap(&mut state.a, &mut state.b);
        }

        t = std::time::Instant::now();
        dev.ops().rmsnorm_zero_centered(
            dev,
            &state.a,
            &self.norm,
            &mut state.normed,
            1,
            hidden,
            eps,
        )?;
        dev.synchronize()?;
        pt.final_norm = t.elapsed().as_secs_f64() * 1e3;

        t = std::time::Instant::now();
        self.lm_head
            .forward(dev, &state.normed, &mut state.logits, 1)?;
        dev.synchronize()?;
        pt.lm_head = t.elapsed().as_secs_f64() * 1e3;

        t = std::time::Instant::now();
        dev.ops()
            .argmax(dev, &state.logits, &mut state.idx, self.vocab_size())?;
        dev.synchronize()?;
        pt.argmax = t.elapsed().as_secs_f64() * 1e3;

        state.n_tokens += 1;
        let v = dev.stream().memcpy_dtov(&state.idx)?;
        Ok((v[0] as u32, pt))
    }
}
