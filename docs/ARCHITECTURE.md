# Architecture — exact Qwen3.5 semantics as implemented

Every rule below was read out of the reference implementation that produced our
oracle traces:
`oracle/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`
(transformers 5.17.0). Getting these details wrong is the main correctness risk,
so they are written down with the line numbers they came from.

## Model shape

`Qwen3_5ForConditionalGeneration` -> `qwen3_5` text backbone.

| | |
|---|---|
| layers | 64: 48 `linear_attention` (Gated DeltaNet) + 16 `full_attention` |
| pattern | layer `i` is full attention iff `(i + 1) % 4 == 0` |
| hidden | 5120 |
| attention | 24 query heads, 4 KV heads, head_dim 256 (GQA group 6) |
| DeltaNet | 16 key heads x 128, 48 value heads x 128, conv kernel 4 |
| mlp | intermediate 17408, SiLU SwiGLU |
| vocab | 248320 (tokenizer emits at most 248076) |
| MTP | 1 layer |
| rope | theta 1e7, partial factor 0.25 -> 64 of 256 dims rotated |

## Quantization

modelopt `MIXED_PRECISION`:

| tensors | format |
|---|---|
| all `mlp.{gate,up,down}_proj`, `lm_head` | NVFP4, group 16 (packed U8 + E4M3 scales + fp32 global) |
| all attention projections (both layer types) | FP8 E4M3, per-tensor fp32 scale |
| norms, `conv1d`, `in_proj_a/b`, `A_log`, `dt_bias`, embeddings, MTP, vision | BF16 |

## Norms: **zero-centered** RMSNorm

`Qwen3_5RMSNorm` (line 841) stores `weight` initialised to **zeros** and
computes

```
y = (x * rsqrt(mean(x^2) + eps)) * (1 + weight)
```

This is *not* the usual `x_norm * weight`. Using the standard form would be
silently wrong on every norm in the model. There are 130 such norms:
2 per layer x 64, plus the final `model.norm`.

**Verified against the checkpoint, not just the source.** The stored weight
distributions are only coherent under the `1 + w` reading. Measured means, and
the implied per-channel gain under each hypothesis:

| tensor | stored mean | if gain = `w` | if gain = `1 + w` |
|---|---|---|---|
| `layers.0.input_layernorm.weight` | -0.0334 | **-0.03** | 0.967 |
| `layers.0.post_attention_layernorm.weight` | -0.2173 | **-0.22** | 0.783 |
| `layers.3.self_attn.q_norm.weight` | +0.2304 | 0.230 | 1.230 |
| `layers.3.self_attn.k_norm.weight` | +0.2203 | 0.220 | 1.220 |
| `model.language_model.norm.weight` | +0.9441 | 0.944 | 1.944 |
| `layers.0.linear_attn.norm.weight` (Gated) | +0.8686 | **0.869** | 1.869 |

Under the standard reading the two block norms would apply a **negative** mean
gain immediately before the token mixer and the MLP, which is not a trained
configuration. Under the `1 + w` reading every gain lands in a plausible band.
`RMSNormGated` is the exception — it really does use `w` directly, and its mean
of 0.869 confirms it.

The BF16 and NVFP4 checkpoints store **byte-identical** norm weights, so the
quantization conversion did not fold the `+1` into the stored tensor. It must
be applied at inference time.

`Qwen3_5RMSNormGated` (line 218), used only inside Gated DeltaNet, is the
ordinary form — `weight` is ones-initialised and there is no `+1`:

```
y = (x_fp32 * rsqrt(mean(x_fp32^2) + eps)) * weight * silu(gate)
```

The normalisation is applied **before** the gate, and the gate is `silu`.

## Full attention layer

From `Qwen3_5Attention.forward` (line 776):

```
qg   = q_proj(x)                      # [B,S,12288]
qg   = qg.view(B, S, 24, 512)
q, g = chunk(qg, 2, dim=-1)           # each [B,S,24,256]
q    = q_norm(q);  k = k_norm(k_proj(x))
q, k = rope(q, k)                     # first 64 of 256 dims
attn = softmax(q @ k^T * (1/sqrt(256))) @ v      # GQA 24:4
attn = attn.reshape(B, S, 6144)
attn = attn * sigmoid(g.reshape(B, S, 6144))     # SIGMOID
out  = o_proj(attn)
```

Two traps:

1. **The output gate is `sigmoid`, not `swish`.** `config.json` advertises
   `output_gate_type: "swish"` and `attn_output_gate: true`, but neither key is
   read anywhere in the modeling file — line 811 hardcodes `torch.sigmoid`.
   Following the config instead of the code would corrupt every full-attention
   layer.
2. **q/gate interleaving is per head.** The view is `(..., 24, 512)` then a
   chunk on the last axis, so for head `h` the q_proj weight rows
   `[512h, 512h+256)` are query and `[512h+256, 512h+512)` are gate. It is not
   "all queries then all gates".

## Gated DeltaNet layer

From `Qwen3_5GatedDeltaNet.forward` (line 550) and
`torch_recurrent_gated_delta_rule` (line 438).

Projections:

```
qkv = in_proj_qkv(x)   # [B,S,10240] = q(2048) + k(2048) + v(6144)
z   = in_proj_z(x)     # [B,S,6144], reshaped to [B,S,48,128]
b   = in_proj_b(x)     # [B,S,48]
a   = in_proj_a(x)     # [B,S,48]
```

Causal depthwise conv1d over the 10240 channels, kernel 4, `padding = 3`, then
`silu`, then take the **last `seq_len` positions** (the padding-produced prefix
is dropped):

```
qkv = silu(conv1d(qkv))[:, :, -S:]
```

Per-token gating (line 604-606):

```
beta  = sigmoid(b)                                   # [B,S,48]
g     = -exp(A_log) * softplus(a + dt_bias)          # [B,S,48], fp32
```

Key/value heads are expanded **before** the recurrence:
`repeat_interleave(48/16 = 3, dim=2)`, so value head `h` reads key head `h/3`.

Recurrence, per token, per value head `h`, with state
`S in R^{128 x 128}` (`k_head_dim x v_head_dim`) held in **fp32**:

```
q = l2norm(q, eps=1e-6);  k = l2norm(k, eps=1e-6)
q = q / sqrt(128)                       # head_dim normalisation
S = S * exp(g)                          # decay
kv_mem = sum_i S[i,:] * k[i]            # -> [128] over v_head_dim
delta  = (v - kv_mem) * beta
S = S + outer(k, delta)
out = sum_i S[i,:] * q[i]               # -> [128]
```

Note `decay_t = exp(g)`, and `g` is already negative, so this is a decay in
`(0,1)`. Both `l2norm` and the `/sqrt(head_dim)` happen inside the kernel
(`use_qk_l2norm_in_kernel=True`).

Then (line 641):

```
out = out.reshape(-1, 128)
z   = z.reshape(-1, 128)
out = RMSNormGated(out, z)              # norm first, then * silu(z)
out = out.reshape(B, S, 6144)
out = out_proj(out)
```

State to keep per sequence per layer: `conv_state` (10240 x 3 bf16 history) and
the fp32 recurrent state (48 x 128 x 128 = 3.1 MB per layer per sequence).
48 DeltaNet layers -> **150 MB of recurrent state per sequence**. At 16
concurrent that is 2.4 GB, which fits comfortably in 121 GiB but must be
budgeted, unlike a conventional KV cache.

## MLP

`down_proj(silu(gate_proj(x)) * up_proj(x))`, plain SwiGLU.

## Decoder layer

```
h = h + token_mixer(input_layernorm(h))
h = h + mlp(post_attention_layernorm(h))
```

## RoPE: text is plain NeoX rotation

`Qwen3_5TextRotaryEmbedding` (line 143) computes
`inv_freq[i] = 1 / theta^(2i/64)` for `i in 0..32`, giving 32 frequencies over
the 64 rotated channels.

The mRoPE machinery (`mrope_section = [11, 11, 10]`,
`recomposition_frequencies`, line 204) exists for the vision grids. For
**text-only** input all three grid position-id rows are identical, so the
recomposition is a no-op and `cat((f, f))` merely duplicates the 32 frequencies
to 64. The result is the standard GPT-NeoX rotation:

```
rotary_dim = 64
a, b = q[..., :32], q[..., 32:64]
q_embed = cat([a*cos - b*sin, b*cos + a*sin])
q_out   = cat([q_embed, q[..., 64:]])   # remaining 192 dims pass through
```

Implementing the full mRoPE is only needed if vision input is ever supported,
which is out of scope (`docs/TARGETS.md`).

## Sampling

`generation_config.json` ships `do_sample: true, temperature 1.0, top_k 20,
top_p 0.95`, but the oracle and every correctness gate use **greedy argmax**
over a fixed length. EOS is `{248046 (<|im_end|>), 248044 (<|endoftext|>)}`.
