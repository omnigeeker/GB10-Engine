"""Dump Gated DeltaNet intermediate stages for layer 0, computed with the
reference `transformers` functions and weights dequantized from the NVFP4
checkpoint.

The layer-level fixture tells us the block is wrong but not where. This splits
it: conv+SiLU, the L2-normalised q/k, the per-head decay/beta, the recurrence
output, and the gated norm. Compare against the engine's own scratch buffers to
find the first diverging stage.

Usage:  python3 tools/delta_stages.py
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
    causal_conv1d_fn,
    l2norm,
    torch_recurrent_gated_delta_rule,
)

MODEL_DIR = os.environ.get("GB10_MODEL_DIR", "models/Qwen3.8-27B-NVFP4")
LAYER = 0
OUT = "fixtures/oracle"
P = f"model.language_model.layers.{LAYER}.linear_attn."


def main() -> int:
    s = Shards(MODEL_DIR)
    npz = np.load(f"{OUT}/layer{LAYER:02d}.npz")
    x_np = npz["input_ln"]  # (T, 5120) — the norm output the engine already matches
    T, H = x_np.shape
    dev = "cuda"
    x = torch.tensor(x_np, dtype=torch.float32, device=dev)

    cfg = json.load(open(f"{MODEL_DIR}/config.json"))["text_config"]
    nk = cfg["linear_num_key_heads"]
    nv = cfg["linear_num_value_heads"]
    kd = cfg["linear_key_head_dim"]
    vd = cfg["linear_value_head_dim"]
    qk = nk * kd
    vdim = nv * vd
    conv_dim = 2 * qk + vdim
    group = nv // nk

    def W(name, is_bf16=False):
        # `dequant` takes the module path; the raw bf16 reader needs `.weight`.
        a = s.bf16(P + name + ".weight") if is_bf16 else s.dequant(P + name)
        return torch.tensor(a, dtype=torch.float32, device=dev)

    stages: dict[str, torch.Tensor] = {}

    qkv = x @ W("in_proj_qkv").T  # (T, 10240)
    z = x @ W("in_proj_z").T  # (T, 6144)
    b = x @ W("in_proj_b", True).T  # (T, 48)
    a = x @ W("in_proj_a", True).T  # (T, 48)
    stages["proj_qkv"] = qkv
    stages["proj_z"] = z

    conv_w = torch.tensor(s.bf16(P + "conv1d.weight"), dtype=torch.float32, device=dev)
    conv_out = causal_conv1d_fn(
        qkv.T.unsqueeze(0).contiguous(), conv_w.squeeze(1), None, activation="silu"
    )[0].T  # (T, 10240)
    stages["conv_out"] = conv_out

    q = conv_out[:, :qk].reshape(T, nk, kd)
    k = conv_out[:, qk : 2 * qk].reshape(T, nk, kd)
    v = conv_out[:, 2 * qk :].reshape(T, nv, vd)

    q_ln = l2norm(q, dim=-1, eps=1e-6) / (kd**0.5)
    k_ln = l2norm(k, dim=-1, eps=1e-6)
    stages["q_ln"] = q_ln
    stages["k_ln"] = k_ln

    beta = b.sigmoid()
    A_log = torch.tensor(s.bf16(P + "A_log"), dtype=torch.float32, device=dev)
    dt_bias = torch.tensor(s.bf16(P + "dt_bias"), dtype=torch.float32, device=dev)
    g = -A_log.exp() * torch.nn.functional.softplus(a + dt_bias)
    decay = g.exp()
    stages["decay"] = decay
    stages["beta"] = beta

    # NOTE: the reference kernel normalises q/k *internally* (including the
    # head_dim^-0.5 factor). Passing pre-normalised vectors would scale them
    # twice. `q_ln`/`k_ln` above are only kept for stage comparison.
    qr = q.repeat_interleave(group, dim=1)
    kr = k.repeat_interleave(group, dim=1)
    out, _ = torch_recurrent_gated_delta_rule(
        qr.unsqueeze(0),
        kr.unsqueeze(0),
        v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    delta_out = out[0].reshape(T * nv, vd)
    stages["delta_out"] = delta_out

    norm_w = torch.tensor(s.bf16(P + "norm.weight"), dtype=torch.float32, device=dev)
    zz = z.reshape(T * nv, vd)
    var = delta_out.pow(2).mean(-1, keepdim=True)
    gnorm = (delta_out * torch.rsqrt(var + 1e-6)) * norm_w
    gnorm = gnorm * torch.nn.functional.silu(zz)
    stages["gnorm"] = gnorm

    # Cross-check: the recomputed gnorm must reproduce the fixture.
    ref = npz["gated_norm"]
    d = np.abs(gnorm.cpu().numpy() - ref).max()
    print(f"self-check: recomputed gated_norm vs fixture max|diff| = {d:.3e}")

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
