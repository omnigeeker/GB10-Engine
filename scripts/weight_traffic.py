#!/usr/bin/env python3
"""Exact per-token weight traffic of a safetensors checkpoint.

Decode is memory-bound for a dense model: every weight byte must cross the
memory bus once per generated token. This script reads only the safetensors
headers (a few KB) and reports the byte breakdown, so the roofline can be
computed without loading the model.

Usage:
    python3 scripts/weight_traffic.py [model_dir] [--bandwidth-gbps 200]
"""
import argparse
import collections
import json
import os
import struct

BITS = {
    "F64": 64, "F32": 32, "F16": 16, "BF16": 16,
    "F8_E4M3": 8, "F8_E5M2": 8, "I8": 8, "U8": 8,
    "I16": 16, "U16": 16, "I32": 32, "U32": 32, "I64": 64,
    "BOOL": 8,
}


def read_headers(model_dir):
    """Return {tensor_name: (dtype, shape)} for every tensor in the shards."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))

    headers = {}
    for shard in shards:
        with open(os.path.join(model_dir, shard), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            blob = json.loads(fh.read(n))
        blob.pop("__metadata__", None)
        headers.update({k: (v["dtype"], v["shape"]) for k, v in blob.items()})
    return headers


def num_bytes(dtype, shape):
    n = 1
    for s in shape:
        n *= s
    if dtype not in BITS:
        raise SystemExit(f"unknown dtype {dtype!r}")
    return n * BITS[dtype] // 8


def classify(name):
    if name.startswith(("model.visual", "visual")):
        return "vision_tower"
    if "mtp" in name:
        return "mtp"
    if "embed_tokens" in name:
        return "embed_tokens"
    if name.startswith("lm_head"):
        return "lm_head"
    if "linear_attn" in name:
        return "linear_attn(GatedDeltaNet)"
    if "self_attn" in name:
        return "full_attn"
    if "mlp" in name:
        return "mlp(NVFP4)"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?", default="models/Qwen3.8-27B-NVFP4")
    ap.add_argument("--bandwidth-gbps", type=float, default=200.0,
                    help="measured usable read bandwidth (default 200 GB/s)")
    args = ap.parse_args()

    headers = read_headers(args.model_dir)
    groups = collections.Counter()
    counts = collections.Counter()
    dtypes = collections.Counter()
    for name, (dtype, shape) in headers.items():
        b = num_bytes(dtype, shape)
        groups[classify(name)] += b
        counts[classify(name)] += 1
        dtypes[dtype] += b

    total = sum(groups.values())
    text = total - groups["vision_tower"]

    print(f"model: {args.model_dir}")
    print(f"tensors: {len(headers)}\n")
    print(f"{'group':28s} {'GB':>9s} {'tensors':>8s}")
    print("-" * 47)
    for k, v in groups.most_common():
        print(f"{k:28s} {v / 1e9:9.3f} {counts[k]:8d}")
    print("-" * 47)
    print(f"{'TOTAL':28s} {total / 1e9:9.3f} {len(headers):8d}")

    print("\nby dtype:")
    for k, v in dtypes.most_common():
        print(f"  {k:10s} {v / 1e9:8.3f} GB")

    bw = args.bandwidth_gbps * 1e9
    print(f"\n--- roofline at {args.bandwidth_gbps:.0f} GB/s usable read ---")
    for label, g in (("full checkpoint", total), ("text-only", text)):
        t = g / bw
        print(f"{label:18s} {g / 1e9:6.2f} GB -> {t * 1000:6.1f} ms/token "
              f"-> {1 / t:5.2f} tok/s single-stream")

    for target in (50.0, 100.0):
        need = text / (1.0 / target)
        print(f"\nto reach {target:.0f} tok/s you need {need / 1e12:.2f} TB/s "
              f"({need / bw:.1f}x the measured bandwidth)")


if __name__ == "__main__":
    main()
