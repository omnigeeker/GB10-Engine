#!/usr/bin/env python3
"""Three-way perplexity comparison for GB10-Engine.

Reads the JSON artifacts written by the three runs and prints the table that
goes into the report. Nothing here re-runs anything; it only combines.

    llama.cpp   tools/llama.cpp/build/bin/llama-perplexity, the NVFP4 GGUF
    engine      target/release/gb10-verify perplexity, the NVFP4 checkpoint
    bf16        bench/ppl/bf16_ppl.py, the unquantized BF16 base model

All three score the same token ids over the same 512-token windows, so the
weights are the only difference between them.
"""
import argparse
import json
import math
import re
import sys


def load(path, key, default=None):
    try:
        with open(path) as fh:
            d = json.load(fh)
        return d
    except FileNotFoundError:
        return default


def parse_llamacpp(log_path):
    """Pull the final estimate and the cumulative trace out of the log."""
    ppl = None
    stderr = None
    trace = []
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = re.search(r"Final estimate: PPL = ([0-9.]+)\s*\+/-\s*([0-9.]+)", line)
            if m:
                ppl, stderr = float(m.group(1)), float(m.group(2))
                continue
            m = re.match(r"\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
            if m:
                trace.append((int(m.group(1)), float(m.group(2))))
    return {"ppl": ppl, "stderr": stderr, "trace": trace}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llamacpp", default="bench/ppl/llamacpp-nvfp4.log")
    p.add_argument("--engine", default="bench/ppl/engine.json")
    p.add_argument("--bf16", default="bench/ppl/bf16.json")
    p.add_argument("--out", default="bench/ppl/compare.json")
    args = p.parse_args()

    lc = parse_llamacpp(args.llamacpp)
    eng = load(args.engine, None)
    bf = load(args.bf16, None)

    rows = []
    if lc["ppl"]:
        rows.append(("llama.cpp", "NVFP4 (same GGUF)", lc["ppl"], lc["stderr"]))
    if eng:
        rows.append(("gb10-engine", "NVFP4 (checkpoint)", eng["ppl"], eng["ppl_stderr_window"]))
    if bf:
        rows.append(("transformers", "BF16 (base model)", bf["ppl"], bf["ppl_stderr_chunk"]))

    print("wikitext-2 test, n_ctx=512, identical token ids (297054 tokens)")
    print()
    print(f"{'implementation':<16} {'weights':<22} {'PPL':>9} {'+-':>8}")
    print("-" * 58)
    for name, w, ppl, se in rows:
        se_s = f"{se:.4f}" if se else "-"
        print(f"{name:<16} {w:<22} {ppl:>9.4f} {se_s:>8}")
    print()

    if lc["ppl"] and eng:
        d = eng["ppl"] - lc["ppl"]
        print(f"engine vs llama.cpp (same NVFP4 weights):")
        print(f"  delta   {d:+.4f} PPL  ({100*d/lc['ppl']:+.3f}%)")
        print(f"  ratio   {eng['ppl']/lc['ppl']:.5f}")
    if lc["ppl"] and bf:
        d = bf["ppl"] - lc["ppl"]
        print(f"BF16 vs llama.cpp (quantization cost):")
        print(f"  delta   {d:+.4f} PPL  ({100*d/lc['ppl']:+.3f}%)")
    if eng and bf:
        d = bf["ppl"] - eng["ppl"]
        print(f"BF16 vs engine (quantization cost):")
        print(f"  delta   {d:+.4f} PPL  ({100*d/eng['ppl']:+.3f}%)")

    out = {"rows": [{"impl": n, "weights": w, "ppl": ppl, "stderr": se}
                    for n, w, ppl, se in rows]}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
