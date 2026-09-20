"""Generate layer-level reference fixtures from the NVFP4 checkpoint.

`transformers` cannot load a pre-quantized `modelopt` checkpoint, so we
dequantize the weights ourselves, load them into the real
`Qwen3_5DecoderLayer` class, and run the authoritative forward pass in fp32.

Running the reference in fp32 (rather than bf16) is deliberate: it isolates
kernel bugs from dtype and quantization noise. Agreement with the bf16
behaviour of the full model is a separate, looser check.

Output: fixtures/oracle/layer{NN}.npz with the layer input, the layer output,
and selected intermediates for debugging.

Usage:  python3 tools/make_layer_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_dequant import Shards  # noqa: E402

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)

MODEL_DIR = os.environ.get("GB10_MODEL_DIR", "models/Qwen3.8-27B-NVFP4")
OUT_DIR = "fixtures/oracle"
SEQ_LEN = 8
LAYERS = [0, 3]  # 0 = Gated DeltaNet, 3 = full attention
SEED = 20260918


def build_config() -> Qwen3_5TextConfig:
    raw = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    cfg = Qwen3_5TextConfig(**raw["text_config"])
    # Eager attention is the reference implementation and is what
    # `docs/ARCHITECTURE.md` documents. SDPA may dispatch to a different
    # kernel depending on the mask.
    cfg._attn_implementation = "eager"
    return cfg


def load_layer(module: torch.nn.Module, shards: Shards, prefix: str) -> None:
    """Fill `module`'s parameters from the checkpoint under `prefix`.

    Quantized weights go through `Shards.dequant`; everything else is bf16 and
    is loaded verbatim (the module applies the `1 + w` convention itself).
    """
    sd = module.state_dict()
    missing = []
    for key in sd:
        full = f"{prefix}{key}"
        if not shards.has(full):
            missing.append(full)
            continue
        if full.endswith(".weight") and not shards.headers[full][0]["dtype"] in ("BF16", "F16", "F32"):
            arr = shards.dequant(full[: -len(".weight")])
        else:
            arr = shards.bf16(full)
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
        if t.shape != sd[key].shape:
            raise RuntimeError(f"{full}: checkpoint {tuple(t.shape)} != module {tuple(sd[key].shape)}")
        sd[key] = t
    if missing:
        raise RuntimeError(f"missing checkpoint tensors: {missing[:8]} (of {len(missing)})")
    module.load_state_dict(sd)
    module.eval()


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    cfg = build_config()
    shards = Shards(MODEL_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Deterministic pseudo-random layer input, scaled like a real residual
    # stream (unit-ish variance, not tiny) so the comparison is meaningful.
    g = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(1, SEQ_LEN, cfg.hidden_size, generator=g, dtype=torch.float32)
    x = (x * 0.5).to(dev)

    position_ids = torch.arange(SEQ_LEN, device=dev).unsqueeze(0).unsqueeze(0)
    position_ids = position_ids.expand(3, 1, SEQ_LEN)

    summary = {}
    for layer_idx in LAYERS:
        torch.manual_seed(SEED)
        layer = Qwen3_5DecoderLayer(cfg, layer_idx).to(dev, dtype=torch.float32)
        prefix = f"model.language_model.layers.{layer_idx}."
        load_layer(layer, shards, prefix)

        captured: dict[str, torch.Tensor] = {}

        def hook(name):
            def fn(_mod, _inp, out):
                captured[name] = out[0] if isinstance(out, tuple) else out

            return fn

        handles = []
        if cfg.layer_types[layer_idx] == "linear_attention":
            handles.append(layer.linear_attn.register_forward_hook(hook("linear_attn")))
            handles.append(layer.linear_attn.norm.register_forward_hook(hook("gated_norm")))
            handles.append(layer.linear_attn.out_proj.register_forward_hook(hook("out_proj")))
        else:
            handles.append(layer.self_attn.register_forward_hook(hook("self_attn")))
            handles.append(layer.self_attn.o_proj.register_forward_hook(hook("o_proj")))
        handles.append(layer.mlp.register_forward_hook(hook("mlp")))
        handles.append(layer.input_layernorm.register_forward_hook(hook("input_ln")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(hook("post_ln")))

        pos_emb = None
        if cfg.layer_types[layer_idx] == "full_attention":
            rotary = Qwen3_5TextRotaryEmbedding(cfg).to(dev)
            with torch.no_grad():
                pos_emb = rotary(x, position_ids)

        # A decoder-only layer must be run CAUSALLY. Passing
        # `attention_mask=None` makes eager/SDPA attention fully
        # bidirectional, so the earlier version of this fixture encoded
        # non-causal semantics that no correct decoder can reproduce.
        # Only the attention layer takes a 4-D additive mask; the DeltaNet
        # path expects a 2-D padding mask and must get None here.
        neg = torch.finfo(torch.float32).min
        causal = None
        if cfg.layer_types[layer_idx] == "full_attention":
            causal = torch.triu(
                torch.full((1, 1, SEQ_LEN, SEQ_LEN), neg, device=dev), diagonal=1
            )

        with torch.no_grad():
            out = layer(
                hidden_states=x,
                position_embeddings=pos_emb,
                attention_mask=causal,
                position_ids=position_ids[0],
                past_key_values=None,
            )
        for h in handles:
            h.remove()

        arrays = {
            "input": x[0].cpu().numpy(),
            "output": out[0].cpu().numpy(),
            "layer_index": np.array(layer_idx),
            "layer_type": np.array(cfg.layer_types[layer_idx]),
        }
        for k, v in captured.items():
            arrays[k] = v.reshape(-1, v.shape[-1]).cpu().numpy()

        path = os.path.join(OUT_DIR, f"layer{layer_idx:02d}.npz")
        np.savez_compressed(path, **arrays)

        # Also emit a flat float32 blob plus an index. The Rust verifier then
        # needs no zip/npy dependency, and the layout is trivial to inspect.
        index = {}
        offset = 0
        with open(os.path.join(OUT_DIR, f"layer{layer_idx:02d}.bin"), "wb") as fb:
            for k in sorted(arrays):
                if not np.issubdtype(np.asarray(arrays[k]).dtype, np.number):
                    continue  # layer_type is a string; the npz keeps it
                a = np.ascontiguousarray(arrays[k], dtype=np.float32)
                index[k] = {"shape": list(a.shape), "offset": offset, "count": int(a.size)}
                fb.write(a.tobytes())
                offset += a.size * 4
        with open(os.path.join(OUT_DIR, f"layer{layer_idx:02d}.json"), "w") as fj:
            json.dump(index, fj, indent=2)

        fin = np.isfinite(arrays["output"]).all()
        summary[layer_idx] = {
            "type": cfg.layer_types[layer_idx],
            "path": path,
            "out_mean": float(arrays["output"].mean()),
            "out_std": float(arrays["output"].std()),
            "finite": bool(fin),
            "captured": sorted(captured),
        }
        print(
            f"layer {layer_idx:2d} ({cfg.layer_types[layer_idx]:16s}) -> {path}  "
            f"out mean={arrays['output'].mean():+.5f} std={arrays['output'].std():.5f} "
            f"finite={fin}  captured={sorted(captured)}"
        )
        if not fin:
            print("  !! non-finite output — reference is broken, not the kernel")
            return 1

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(
            {
                "seq_len": SEQ_LEN,
                "seed": SEED,
                "dtype": "float32",
                "model": MODEL_DIR,
                "hidden": cfg.hidden_size,
                "layers": summary,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {OUT_DIR}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
