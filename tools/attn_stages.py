"""Dump full-attention intermediate stages for layer 3.

Same idea as `delta_stages.py`: the layer-level fixture says the attention
block is wrong but not which part. This captures q/k/v projections, the
per-head q/gate split, the per-head RMSNorm, RoPE, the softmax attention
output, the sigmoid gate, and o_proj.

Usage:  python3 tools/attn_stages.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_dequant import Shards  # noqa: E402

from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    apply_rotary_pos_emb,
    repeat_kv,
)

MODEL_DIR = os.environ.get("GB10_MODEL_DIR", "models/Qwen3.8-27B-NVFP4")
LAYER = 3
OUT = "fixtures/oracle"
P = f"model.language_model.layers.{LAYER}.self_attn."


def main() -> int:
    s = Shards(MODEL_DIR)
    npz = np.load(f"{OUT}/layer{LAYER:02d}.npz")
    x_np = npz["input_ln"]  # the engine already matches this
    T, _ = x_np.shape
    dev = "cuda"
    x = torch.tensor(x_np, dtype=torch.float32, device=dev)

    cfg = json.load(open(f"{MODEL_DIR}/config.json"))["text_config"]
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    partial = cfg["rope_parameters"].get("partial_rotary_factor", 1.0)
    rotary = int(hd * partial)
    base = cfg["rope_parameters"]["rope_theta"]

    def W(name, bf16=False):
        a = s.bf16(P + name + ".weight") if bf16 else s.dequant(P + name)
        return torch.tensor(a, dtype=torch.float32, device=dev)

    stages: dict[str, torch.Tensor] = {}

    q_out = x @ W("q_proj").T  # (T, 12288)
    k_out = x @ W("k_proj").T  # (T, 1024)
    v_out = x @ W("v_proj").T  # (T, 1024)
    stages["q_proj"] = q_out
    stages["k_proj"] = k_out
    stages["v_proj"] = v_out

    # q_proj emits [n_heads, 2*head_dim]: query then gate, per head.
    qg = q_out.view(T, nh, 2 * hd)
    q = qg[..., :hd]
    gate = qg[..., hd:]
    stages["gate"] = gate.reshape(T, nh * hd)

    qw = torch.tensor(s.bf16(P + "q_norm.weight"), dtype=torch.float32, device=dev)
    kw = torch.tensor(s.bf16(P + "k_norm.weight"), dtype=torch.float32, device=dev)
    k = k_out.view(T, nkv, hd)

    def rmsnorm_zero_centered(t, w):
        t = t.float()
        return (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)) * (1.0 + w)

    qn = rmsnorm_zero_centered(q, qw)
    kn = rmsnorm_zero_centered(k, kw)
    stages["q_norm"] = qn.reshape(T, nh * hd)
    stages["k_norm"] = kn.reshape(T, nkv * hd)

    # RoPE over the first `rotary` channels of each head (NeoX rotation).
    inv = 1.0 / (base ** (torch.arange(0, rotary, 2, dtype=torch.float, device=dev) / rotary))
    pos = torch.arange(T, dtype=torch.float, device=dev)
    freqs = torch.outer(pos, inv)  # (T, rotary/2)
    # `position_embeddings` are (B, T, rotary); apply_rotary_pos_emb unsqueezes
    # dim 1 so they broadcast against (B, heads, T, head_dim).
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1).unsqueeze(0)
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1).unsqueeze(0)
    qr, kr = apply_rotary_pos_emb(
        qn.unsqueeze(0).transpose(1, 2), kn.unsqueeze(0).transpose(1, 2), cos, sin
    )
    qr = qr.transpose(1, 2)[0]  # (T, nh, hd)
    kr = kr.transpose(1, 2)[0]
    stages["q_rope"] = qr.reshape(T, nh * hd)
    stages["k_rope"] = kr.reshape(T, nkv * hd)

    # Causal softmax attention, exactly as eager_attention_forward computes it.
    scale = hd**-0.5
    qh = qr.transpose(0, 1).unsqueeze(0)  # (1, nh, T, hd)
    kh = repeat_kv(kr.transpose(0, 1).unsqueeze(0), nh // nkv)
    vh = repeat_kv(v_out.view(T, nkv, hd).transpose(0, 1).unsqueeze(0), nh // nkv)
    scores = (qh @ kh.transpose(-1, -2)) * scale
    if os.environ.get("GB10_NONCAUSAL") == "1":
        pass  # reproduce the fixture, which was run with attention_mask=None
    else:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    attn = probs @ vh  # (1, nh, T, hd)
    attn = attn[0].transpose(0, 1)  # (T, nh, hd)
    stages["attn_out"] = attn.reshape(T, nh * hd)
    stages["attn_gated"] = (attn * torch.sigmoid(gate)).reshape(T, nh * hd)

    stages["o_proj"] = (attn * torch.sigmoid(gate)).reshape(T, nh * hd) @ W("o_proj").T

    ref = npz["self_attn"]
    d = np.abs(stages["o_proj"].cpu().numpy() - ref).max()
    print(f"self-check: recomputed self_attn vs fixture max|diff| = {d:.3e}")

    index = {}
    off = 0
    with open(f"{OUT}/layer{LAYER:02d}_stages.bin", "wb") as f:
        for name in sorted(stages):
            arr = np.ascontiguousarray(stages[name].detach().cpu().numpy(), dtype=np.float32)
            index[name] = {"shape": list(arr.shape), "offset": off, "count": int(arr.size)}
            f.write(arr.tobytes())
            off += arr.size * 4
    with open(f"{OUT}/layer{LAYER:02d}_stages.json", "w") as f:
        json.dump(index, f, indent=2)

    for name in sorted(stages):
        t = stages[name]
        print(f"  {name:12s} {tuple(t.shape)}  mean={t.mean():+.5f} std={t.std():.5f} "
              f"max|.|={t.abs().max():.4f}")
    print(f"\nwrote {OUT}/layer{LAYER:02d}_stages.{{bin,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
