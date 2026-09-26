#!/usr/bin/env python3
"""Multi-prompt, long-generation token-exactness check for GB10-Engine.

The single committed fixture (`fixtures/oracle/greedy_tokens.json`) pins one
prompt and 16 tokens. This drives the engine over every oracle produced by

    python3 tools/full_oracle.py --n 64 --prompts-file tools/oracle_prompts.json \
        --out-dir fixtures/oracle-multi

and requires exact agreement on all of them. Because those oracles come from the
same NVFP4 checkpoint dequantized to bf16, the weights are the engine's weights:
any disagreement is an implementation defect, not a quantization effect.

    python3 bench/ppl/multi_prompt_check.py [--n 64] [--json out.json]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

RE_PROMPT = re.compile(r"prompt: (\d+) tokens")
RE_TTFT = re.compile(r"TTFT ([\d.]+) ms")
RE_AGREE = re.compile(r"oracle agreement: (\d+)/(\d+) \(([\d.]+)%\)")
RE_FIRSTBAD = re.compile(r"first divergence at index (\d+): engine (\d+) vs oracle (\d+)")
RE_DECODED = re.compile(r"decoded (\d+) tokens in ([\d.]+)s")


def run_one(binary, model, oracle_dir, n):
    cmd = [binary, "generate", "--n", str(n), "--oracle", oracle_dir, "--model", model]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    text = p.stdout + p.stderr

    rec = {
        "oracle": os.path.basename(oracle_dir.rstrip("/")),
        "exit": p.returncode,
        "wall_s": round(wall, 1),
        "exact": "exact match" in text,
        "first_bad": None,
    }
    for key, rx, cast in (
        ("prompt_tokens", RE_PROMPT, int),
        ("ttft_ms", RE_TTFT, float),
    ):
        m = rx.search(text)
        if m:
            rec[key] = cast(m.group(1))
    m = RE_AGREE.search(text)
    if m:
        rec["agree"] = int(m.group(1))
        rec["compared"] = int(m.group(2))
        rec["rate"] = float(m.group(3))
    m = RE_DECODED.search(text)
    if m:
        rec["decoded"] = int(m.group(1))
        rec["decode_s"] = float(m.group(2))
    m = RE_FIRSTBAD.search(text)
    if m:
        rec["first_bad"] = {
            "index": int(m.group(1)),
            "engine": int(m.group(2)),
            "oracle": int(m.group(3)),
        }
    if rec["exit"] != 0 and not m:
        rec["tail"] = text.strip().splitlines()[-6:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="target/release/gb10-verify")
    ap.add_argument("--model", default="models/Qwen3.8-27B-NVFP4")
    ap.add_argument("--oracles", default="fixtures/oracle-multi")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--json", default="bench/ppl/multi-prompt.json")
    args = ap.parse_args()

    if not os.path.isdir(args.oracles):
        raise SystemExit(
            f"{args.oracles} missing — run tools/full_oracle.py --prompts-file first"
        )
    dirs = sorted(
        d
        for d in os.listdir(args.oracles)
        if os.path.isfile(os.path.join(args.oracles, d, "greedy_tokens.json"))
    )
    if not dirs:
        raise SystemExit(f"{args.oracles}: no oracle directories")

    print(f"{len(dirs)} prompts, up to {args.n} greedy tokens each, weights = NVFP4")
    print()
    print(f"{'prompt':<14} {'tok':>4} {'out':>4} {'agree':>7} {'ttft_ms':>8} {'wall_s':>7}  result")
    print("-" * 72)

    rows = []
    for d in dirs:
        rec = run_one(args.binary, args.model, os.path.join(args.oracles, d), args.n)
        rows.append(rec)
        agree = f"{rec.get('agree','?')}/{rec.get('compared','?')}"
        result = "exact" if rec["exact"] else (
            f"DIVERGE@{rec['first_bad']['index']} "
            f"engine={rec['first_bad']['engine']} oracle={rec['first_bad']['oracle']}"
            if rec["first_bad"] else f"FAILED exit={rec['exit']}"
        )
        print(
            f"{rec['oracle']:<14} {rec.get('prompt_tokens',0):>4} {rec.get('decoded',0):>4} "
            f"{agree:>7} {rec.get('ttft_ms',0):>8.1f} {rec['wall_s']:>7.1f}  {result}"
        )
        sys.stdout.flush()

    n_exact = sum(1 for r in rows if r["exact"])
    total = sum(r.get("compared", 0) for r in rows)
    agreed = sum(r.get("agree", 0) for r in rows)
    print()
    print(f"exact prompts : {n_exact}/{len(rows)}")
    print(f"tokens agreed : {agreed}/{total}" + (f" ({100*agreed/total:.2f}%)" if total else ""))

    summary = {
        "kind": "multi-prompt-token-exactness",
        "model": args.model,
        "n": args.n,
        "prompts": len(rows),
        "exact_prompts": n_exact,
        "tokens_agreed": agreed,
        "tokens_compared": total,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {args.json}")
    return 0 if n_exact == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
