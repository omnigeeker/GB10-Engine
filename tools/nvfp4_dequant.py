"""Dequantization helpers for the nv-community/Qwen3.8-27B-NVFP4 checkpoint.

Used to rebuild an fp32 reference of individual layers, because `transformers`
cannot load a pre-quantized `modelopt` checkpoint directly and we specifically
want to test against the *quantized* weights the engine actually runs.

Formats (verified against the checkpoint's safetensors headers):

  NVFP4  weight         U8    [N, K/2]   two E2M1 nibbles per byte, low nibble first
         weight_scale   E4M3  [N, K/16]  one scale per 16-element group
         weight_scale_2 F32   []         per-tensor global scale
  FP8    weight         E4M3  [N, K]
         weight_scale   F32   []         per-tensor scale

The E2M1 magnitude table is 0, 0.5, 1, 1.5, 2, 3, 4, 6 with bit 3 as sign.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

# E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit.
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _build_e2m1_table() -> np.ndarray:
    t = np.zeros(16, dtype=np.float32)
    for b in range(16):
        mag = E2M1_MAG[b & 0x7]
        t[b] = -mag if (b & 0x8) else mag
    return t


def _build_e4m3_table() -> np.ndarray:
    """Decode all 256 E4M3 patterns the way the hardware does.

    exp == 0 is subnormal (value = m/8 * 2^-6); exp == 15 and m == 7 is NaN,
    which we map to 0.0 since it never appears in a valid checkpoint.
    """
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0:
            val = (man / 8.0) * (2.0**-6)
        elif exp == 15 and man == 7:
            val = 0.0  # NaN
        else:
            val = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
        t[b] = sign * val
    return t


E2M1 = _build_e2m1_table()
E4M3 = _build_e4m3_table()


class Shards:
    """Lazy reader over a sharded safetensors checkpoint."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        index = os.path.join(model_dir, "model.safetensors.index.json")
        self.map = json.load(open(index))["weight_map"]
        self.headers: dict[str, tuple[dict, str, int]] = {}
        for fn in sorted(set(self.map.values())):
            with open(os.path.join(model_dir, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            for k, v in hdr.items():
                self.headers[k] = (v, fn, 8 + n)

    def has(self, name: str) -> bool:
        return name in self.headers

    def raw(self, name: str) -> tuple[np.ndarray, dict]:
        meta, fn, base = self.headers[name]
        o = base + meta["data_offsets"][0]
        length = meta["data_offsets"][1] - meta["data_offsets"][0]
        with open(os.path.join(self.dir, fn), "rb") as f:
            f.seek(o)
            buf = f.read(length)
        dt = {
            "BF16": np.uint16,
            "F16": np.uint16,
            "F32": np.float32,
            "F8_E4M3": np.uint8,
            "U8": np.uint8,
            "I8": np.int8,
        }[meta["dtype"]]
        return np.frombuffer(buf, dtype=dt).copy(), meta

    def bf16(self, name: str) -> np.ndarray:
        a, meta = self.raw(name)
        assert meta["dtype"] == "BF16", (name, meta["dtype"])
        return (a.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])

    def f32(self, name: str) -> np.ndarray:
        a, meta = self.raw(name)
        assert meta["dtype"] == "F32", (name, meta["dtype"])
        return a.reshape(meta["shape"])

    def dequant(self, prefix: str) -> np.ndarray:
        """Dequantize `<prefix>.weight` into fp32 [N, K] using its sibling scales."""
        w, meta = self.raw(f"{prefix}.weight")
        shape = meta["shape"]
        n, k = shape[0], int(np.prod(shape[1:]))

        if meta["dtype"] == "BF16":
            return (w.astype(np.uint32) << 16).view(np.float32).reshape(shape)

        if meta["dtype"] == "F8_E4M3":
            scale = float(self.f32(f"{prefix}.weight_scale").reshape(-1)[0])
            return (E4M3[w].reshape(n, k) * scale).astype(np.float32)

        if meta["dtype"] == "U8":
            # The packed weight is [N, K/2]; K is only recoverable from the
            # group-scale shape, which is [N, K/16].
            sraw, smeta = self.raw(f"{prefix}.weight_scale")
            assert smeta["dtype"] == "F8_E4M3", (prefix, smeta["dtype"])
            assert smeta["shape"][0] == n, (prefix, smeta["shape"], n)
            group = smeta["shape"][1]
            k = group * 16
            kk = k // 2
            assert shape[1] == kk, f"{prefix}: packed shape {shape} does not match K={k}"

            nib = np.empty((n, k), dtype=np.uint8)
            nib[:, 0::2] = w.reshape(n, kk) & 0x0F
            nib[:, 1::2] = (w.reshape(n, kk) >> 4) & 0x0F
            vals = E2M1[nib]

            scales = np.repeat(E4M3[sraw].reshape(n, group), 16, axis=1)

            s2 = float(self.f32(f"{prefix}.weight_scale_2").reshape(-1)[0])
            return (vals * scales * s2).astype(np.float32)

        raise ValueError(f"{prefix}: unsupported dtype {meta['dtype']}")


if __name__ == "__main__":
    import sys

    s = Shards(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.8-27B-NVFP4")
    for name in [
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_a",
    ]:
        w = s.dequant(name)
        print(f"{name:60s} {w.shape} mean={w.mean():+.5f} std={w.std():.5f}")
