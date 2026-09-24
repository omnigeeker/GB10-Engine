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

    /// Scoring for the MTP verify path: `lm_head` over *every* row of
    /// `state.normed`, argmaxed into `state.idx_all`.
    ///
    /// `prefill_seq` scores only the last row on purpose -- over a full prompt
    /// that is 715 MB of weights per token. Verification is the opposite case:
    /// a handful of rows, each of which reuses a weight read the single-row
    /// path would have had to repeat.
    pub fn all_logits(&self, dev: &Device, state: &mut ModelState, t: usize) -> Result<()> {
        anyhow::ensure!(t >= 1 && t <= VERIFY_MAX, "all_logits rows {t}");
        self.lm_head
            .forward(dev, &state.normed, &mut state.logits_all, t)?;
        dev.ops()
            .argmax_multi(dev, &state.logits_all, &mut state.idx_all, self.vocab_size(), t)?;
        Ok(())
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
    /// Per-row logits for the MTP verify path, which needs the prediction of
    /// *every* drafted row rather than just the last one. Sized for
    /// `VERIFY_MAX` rows: at 248320 vocab a `max_seq`-sized buffer would be
    /// 2 GB, and verification never covers more than a handful of tokens.
    pub logits_all: CudaSlice<f32>,
    /// Argmax of each of those rows.
    pub idx_all: CudaSlice<i32>,
    /// Copies of every layer's recurrent state, taken before a speculative
    /// verification pass.
    ///
    /// The KV cache is append-ordered, so rejected drafts can be discarded just
    /// by rewinding `n_keys` -- the slots get overwritten. The Gated DeltaNet
    /// recurrence is *not* reversible: its 48 layers carry 150 MB per sequence
    /// of running state that a rejected draft has already folded in. So
    /// verification snapshots it first and restores it afterwards. That costs
    /// ~300 MB of traffic per round against the 17.6 GB a decoder step reads,
    /// about 1.7%, which is affordable for the multi-token verify it enables.
    pub rec_snapshot: Vec<CudaSlice<f32>>,
    pub conv_snapshot: Vec<CudaSlice<f32>>,
    /// `n_keys` before the verification pass, per sequence.
    pub n_keys_snapshot: Vec<Vec<usize>>,
    /// Token count before the verification pass. `prefill_seq` advances it, so
    /// leaving it un-restored makes every later round read a stale row.
    pub n_tokens_snapshot: usize,
    /// Token ids staged on the device for a batched decode step.
    pub tokens_dev: CudaSlice<i32>,
    /// Number of sequences this state is sliced into.
    pub n_seq: usize,
    pub n_tokens: usize,
}

impl ModelState {
    pub fn new(dev: &Device, model: &Model, max_seq: usize, n_seq: usize) -> Result<Self> {
        let text = model.text();
        let mut layers = Vec::with_capacity(model.layers.len());
        for l in &model.layers {
            layers.push(LayerState::new(dev, text, l, max_seq, n_seq)?);
        }
        let z = |n: usize| -> Result<CudaSlice<f32>> { Ok(dev.stream().alloc_zeros::<f32>(n)?) };
        let vocab_hint = model.vocab_size();
        // Built before `layers` is moved into the struct below.
        let rec_snapshot: Vec<CudaSlice<f32>> = layers
            .iter()
            .map(|l| z(l.rec.len()))
            .collect::<Result<_>>()?;
        let conv_snapshot: Vec<CudaSlice<f32>> = layers
            .iter()
            .map(|l| z(l.conv_hist.len()))
            .collect::<Result<_>>()?;
        let n_keys_snapshot: Vec<Vec<usize>> =
            layers.iter().map(|l| l.n_keys.clone()).collect();
        Ok(Self {
            layers,
            logits: z(model.vocab_size() * n_seq)?,
            idx: dev.stream().alloc_zeros::<i32>(n_seq)?,
            // Per-token buffers are `max_seq` rows so prefill can run the
            // whole prompt through the stack in one batched pass.
            a: z(text.hidden_size * max_seq)?,
            b: z(text.hidden_size * max_seq)?,
            normed: z(text.hidden_size * max_seq)?,
            last: z(text.hidden_size)?,
            logits_all: z(vocab_hint * VERIFY_MAX)?,
            idx_all: dev.stream().alloc_zeros::<i32>(VERIFY_MAX)?,
            rec_snapshot,
            conv_snapshot,
            n_keys_snapshot,
            tokens_dev: dev.stream().alloc_zeros::<i32>(n_seq)?,
            n_seq,
            n_tokens: 0,
            n_tokens_snapshot: 0,
        })
    }

    /// Drop all context: KV caches, conv history and recurrent state.
    /// Capture the recurrent state of every layer, plus the cache lengths.
    pub fn snapshot_recurrent(&mut self, dev: &Device) -> Result<()> {
        for (i, l) in self.layers.iter().enumerate() {
            dev.stream().memcpy_dtod(&l.rec, &mut self.rec_snapshot[i])?;
            dev.stream().memcpy_dtod(&l.conv_hist, &mut self.conv_snapshot[i])?;
            self.n_keys_snapshot[i].copy_from_slice(&l.n_keys);
        }
        self.n_tokens_snapshot = self.n_tokens;
        Ok(())
    }

    /// Undo everything after `snapshot_recurrent`. `positions` is rebuilt from
    /// `n_keys` rather than snapshotted, since it is a pure function of it.
    pub fn restore_recurrent(&mut self, dev: &Device) -> Result<()> {
        for (i, l) in self.layers.iter_mut().enumerate() {
            dev.stream().memcpy_dtod(&self.rec_snapshot[i], &mut l.rec)?;
            dev.stream().memcpy_dtod(&self.conv_snapshot[i], &mut l.conv_hist)?;
            l.n_keys.copy_from_slice(&self.n_keys_snapshot[i]);
            l.sync_positions(dev)?;
        }
        self.n_tokens = self.n_tokens_snapshot;
        Ok(())
    }

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
            layer.forward(dev, text, &state.a, &mut state.b, &mut state.layers[i], sc, 0)?;
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
        dev.check_err()?;
        Ok(v[0] as u32)
    }

    /// One batched decode step: consume one token per sequence and return the
    /// greedy next token for each.
    ///
    /// This is the throughput path. The projections and norms run once over all
    /// `n_seq` rows, so the 17.6 GB of weights are streamed once per step
    /// rather than once per sequence. Only the state-touching ops (conv
    /// history, the recurrence, the KV cache) are per-sequence, and they are
    /// indexed by `seq * stride` inside the shared state allocation.
    pub fn step_batch(
        &self,
        dev: &Device,
        tokens: &[u32],
        state: &mut ModelState,
        sc: &mut Scratch,
    ) -> Result<Vec<u32>> {
        let text = self.text();
        let hidden = text.hidden_size;
        let eps = text.rms_norm_eps as f32;
        let n_seq = tokens.len();
        anyhow::ensure!(n_seq > 0, "step_batch: empty batch");
        anyhow::ensure!(n_seq <= state.n_seq, "step_batch: batch exceeds state");

        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        dev.stream().memcpy_htod(&ids, &mut state.tokens_dev)?;
        dev.ops().embed_gather_batched(
            dev,
            &self.embed,
            &state.tokens_dev,
            &mut state.a,
            hidden,
            n_seq,
        )?;

        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_batch(dev, text, &state.a, &mut state.b, &mut state.layers[i], sc, n_seq)?;
            std::mem::swap(&mut state.a, &mut state.b);
        }

        dev.ops().rmsnorm_zero_centered(
            dev,
            &state.a,
            &self.norm,
            &mut state.normed,
            n_seq,
            hidden,
            eps,
        )?;
        self.lm_head.forward(dev, &state.normed, &mut state.logits, n_seq)?;
        dev.ops()
            .argmax_multi(dev, &state.logits, &mut state.idx, self.vocab_size(), n_seq)?;

        state.n_tokens += 1;
        let v = dev.stream().memcpy_dtov(&state.idx)?;
        dev.check_err()?;
        Ok(v.iter().map(|&x| x as u32).collect())
    }

    /// Feed a prompt, return the greedy token after the last prompt token.
    pub fn prefill(
        &self,
        dev: &Device,
        tokens: &[u32],
        state: &mut ModelState,
        sc: &mut Scratch,
    ) -> Result<u32> {
        self.prefill_seq(dev, tokens, state, sc, 0)
    }

    /// `prefill` into one sequence slot of a multi-sequence state.
    ///
    /// The residual buffers are transient -- only the conv history, the
    /// recurrence and the KV cache persist -- so `seq` affects where state
    /// lands, not how the prompt is computed.
    pub fn prefill_seq(
        &self,
        dev: &Device,
        tokens: &[u32],
        state: &mut ModelState,
        sc: &mut Scratch,
        seq: usize,
    ) -> Result<u32> {
        let t = tokens.len();
        self.forward_normed(dev, tokens, state, sc, seq)?;

        // Only the final prompt row needs logits; running lm_head over all `t`
        // rows would re-read 715 MB of weights per prompt token.
        let hidden = self.text().hidden_size;
        dev.ops()
            .copy_last_row(dev, &state.normed, &mut state.last, t, hidden)?;
        self.lm_head.forward(dev, &state.last, &mut state.logits, 1)?;
        dev.ops()
            .argmax(dev, &state.logits, &mut state.idx, self.vocab_size())?;
        let v = dev.stream().memcpy_dtov(&state.idx)?;
        dev.check_err()?;
        Ok(v[0] as u32)
    }

    /// Run `tokens` through the whole stack and leave the final-norm rows of
    /// *every* position in `state.normed` (`[t, hidden]`), with no `lm_head`.
    ///
    /// This is the shared body of `prefill_seq`, which then scores only the
    /// last row, and of the perplexity path, which has to score a whole window
    /// of rows and therefore drives `lm_head` itself.
    ///
    /// `seq` selects which sequence slot of a multi-sequence state the conv
    /// history, recurrence and KV cache land in. The residual buffers are
    /// transient, so it does not change how the prompt is computed.
    pub fn forward_normed(
        &self,
        dev: &Device,
        tokens: &[u32],
        state: &mut ModelState,
        sc: &mut Scratch,
        seq: usize,
    ) -> Result<()> {
        let t = tokens.len();
        if t == 0 {
            anyhow::bail!("forward_normed: empty prompt");
        }
        let text = self.text();
        let hidden = text.hidden_size;
        anyhow::ensure!(
            t <= state.normed.len() / hidden,
            "forward_normed: {t} rows exceeds the state's max_seq of {}",
            state.normed.len() / hidden
        );

        let ids: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        let ids_dev = dev.stream().clone_htod(&ids)?;
        dev.ops()
            .embed_gather_batched(dev, &self.embed, &ids_dev, &mut state.a, hidden, t)?;

        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_prefill(dev, text, &state.a, &mut state.b, &mut state.layers[i], sc, t, seq)?;
            std::mem::swap(&mut state.a, &mut state.b);
        }

        dev.ops().rmsnorm_zero_centered(
            dev,
            &state.a,
            &self.norm,
            &mut state.normed,
            t,
            hidden,
            text.rms_norm_eps as f32,
        )?;
        state.n_tokens += t;
        // The temporary `ids_dev` above is dropped at the end of this function,
        // and `CudaSlice::drop` synchronises the stream and swallows whatever
        // that sync reports. Surface it here, before the caller scores these
        // rows, so an aborted kernel cannot become a quiet wrong answer.
        dev.check_err()?;
        Ok(())
    }
}

/// Rows the MTP verify path can score at once. Each extra verified row is one
/// more drafted token, bought for the ~0.85 GB the head costs to draft it
/// instead of the 17.6 GB a decoder step costs.
pub const VERIFY_MAX: usize = 64;

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
            layer.forward(dev, text, &state.a, &mut state.b, &mut state.layers[i], sc, 0)?;
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
        dev.check_err()?;
        Ok((v[0] as u32, pt))
    }
}
