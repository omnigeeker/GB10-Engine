#!/usr/bin/env python3
"""Is the bf16 oracle deterministic at all?

`divergence_analysis.py` re-ran the same greedy decode that produced the
committed fixtures and got different tokens — at step 1 for one prompt. If the
reference is not reproducible, then "token-exact agreement with the reference"
is not a well-defined criterion, and an engine divergence from a fixture is not
by itself evidence of an engine defect.

This runs the identical decode N times in one process and compares the runs with
each other, so nothing but the model's own reproducibility is under test.

  --deterministic sets torch.use_deterministic_algorithms(True) and
  CUBLAS_WORKSPACE_CONFIG, to test whether non-deterministic kernels are the
  cause rather than to paper over it.

    python3 bench/ppl/oracle_determinism.py --tags p01-capital,p04-chinese --reps 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "tools")
import full_oracle as fo  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402


def greedy(model, ids, n, dev):
    out = []
    cur = torch.tensor([ids], device=dev)
    with torch.no_grad():
        for _ in range(n):
            nxt = int(model(input_ids=cur).logits[:, -1, :].float().argmax(-1).item())
            if nxt in fo.EOS_IDS:
                break
            out.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=dev)], dim=1)
    return out


def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="p01-capital,p02-math,p04-chinese,p06-kyoto")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        print("torch.use_deterministic_algorithms(True)")

    tags = [t for t in args.tags.split(",") if t]
    cfg, shards = fo.build()
    model = Qwen3_5ForCausalLM._from_config(cfg, torch_dtype=torch.bfloat16)
    model.eval()
    print("loading weights ...", flush=True)
    with torch.no_grad():
        fo.load_weights(model, shards)
    model.to("cuda")
    torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(fo.MODEL_DIR)

    results = []
    for tag in tags:
        fx = json.load(open(f"fixtures/oracle-multi/{tag}/greedy_tokens.json"))
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": fx["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(rendered, add_special_tokens=False)["input_ids"]

        runs = [greedy(model, ids, args.n, "cuda") for _ in range(args.reps)]
        # Compare every run against the first, and against the committed fixture.
        diffs = [first_diff(runs[0], r) for r in runs[1:]]
        fx_diff = first_diff(runs[0], fx["tokens"])
        stable = all(d is None for d in diffs)
        rec = {
            "tag": tag,
            "reps": args.reps,
            "stable": stable,
            "run_diffs": diffs,
            "vs_fixture_first_diff": fx_diff,
            "runs": runs,
        }
        results.append(rec)
        print(
            f"{tag:<14} stable={str(stable):<5} run-vs-run diffs={diffs}  "
            f"vs fixture first diff={fx_diff}",
            flush=True,
        )
        for i, r in enumerate(runs):
            print(f"    run{i}: {r}", flush=True)

    n_stable = sum(1 for r in results if r["stable"])
    print()
    print(f"stable prompts: {n_stable}/{len(results)} at n={args.n}, reps={args.reps}")
    out = args.out or (
        "bench/ppl/oracle-determinism-deterministic.json"
        if args.deterministic
        else "bench/ppl/oracle-determinism.json"
    )
    with open(out, "w") as fh:
        json.dump({"deterministic_flag": args.deterministic, "n": args.n,
                   "reps": args.reps, "stable": n_stable, "results": results}, fh, indent=2)
    print(f"wrote {out}")
    return 0 if n_stable == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
