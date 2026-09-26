#!/usr/bin/env python3
"""Reference MMLU accuracy, scored exactly like `gb10-verify choice`.

Standard MMLU scoring: one forward pass per question, then argmax over the
logits of the four answer-letter tokens at the position after "Answer:".
" A".." D" are single tokens in this vocabulary; that is asserted, not assumed.

The prompt is built by the same rule as the Rust side. The engine records its
own token count per question, and `--engine-json` makes this script verify that
this implementation tokenised every prompt to the same length -- if it did not,
the two are not answering the same questions and the comparison is void.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import render  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--engine-json", default=None,
                   help="engine result, to cross-check per-question token counts")
    p.add_argument("--mode", choices=["bf16", "dequant"], default="bf16",
                   help="bf16: the unquantized base model (total quantization cost). "
                        "dequant: this NVFP4 checkpoint dequantized to bf16, i.e. the "
                        "same weights the engine holds (implementation fidelity).")
    args = p.parse_args()

    questions = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    if args.limit >= 0:
        questions = questions[:args.limit]
    print(f"questions: {len(questions)}", flush=True)

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)

    letters = [tok(s, add_special_tokens=False)["input_ids"] for s in [" A", " B", " C", " D"]]
    assert all(len(x) == 1 for x in letters), f"letter not a single token: {letters}"
    letters = [x[0] for x in letters]
    print(f'answer letters " A".." D" -> token ids {letters}', flush=True)

    # Cross-check: the engine's token count per question must match ours.
    engine_counts = None
    if args.engine_json:
        ej = json.load(open(args.engine_json))
        engine_counts = {r["id"]: r["prompt_tokens"] for r in ej["rows"]}
        assert ej["letter_token_ids"] == letters, (
            f"letter ids differ: engine {ej['letter_token_ids']} vs reference {letters}")

    t0 = time.time()
    if args.mode == "bf16":
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
        model.eval()
    else:
        import os
        import sys
        root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        sys.path.insert(0, os.path.join(root, "tools"))
        from full_oracle import build, load_weights
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        cfg, shards = build()
        model = Qwen3_5ForCausalLM._from_config(cfg, torch_dtype=torch.bfloat16)
        model.eval()
        with torch.no_grad():
            load_weights(model, shards)
        model.to("cuda")
        torch.cuda.empty_cache()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    correct = 0
    per_subject = {}
    rows = []
    mismatched = 0
    started = time.time()

    with torch.no_grad():
        for qi, q in enumerate(questions):
            ids = tok(render(q), add_special_tokens=False)["input_ids"]
            if engine_counts is not None and engine_counts.get(q["id"]) != len(ids):
                mismatched += 1
            inp = torch.tensor([ids], device="cuda")
            logits = model(input_ids=inp).logits[0, -1]
            scores = [float(logits[t]) for t in letters]
            pick = max(range(4), key=lambda i: scores[i])
            ok = pick == q["answer"]
            correct += ok
            e = per_subject.setdefault(q["subject"], [0, 0])
            e[1] += 1
            e[0] += ok
            rows.append({"id": q["id"], "subject": q["subject"], "gold": q["answer"],
                         "pred": pick, "scores": scores, "prompt_tokens": len(ids)})
            if qi % 100 == 0 or qi + 1 == len(questions):
                el = time.time() - started
                print(f"  [{qi}/{len(questions)}] acc={correct/(qi+1):.4f}  "
                      f"{el/(qi+1):.2f}s/q  eta {el/(qi+1)*(len(questions)-qi-1)/60:.1f} min",
                      flush=True)

    n = len(questions)
    acc = correct / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (acc + z * z / (2 * n)) / denom
    half = z * ((acc * (1 - acc) / n + z * z / (4 * n * n)) ** 0.5) / denom

    print()
    print(f"MMLU accuracy: {correct}/{n} = {acc:.4f}")
    print(f"  95% CI: [{centre-half:.4f}, {centre+half:.4f}]")
    if engine_counts is not None:
        print(f"  prompt-token cross-check: {mismatched} mismatches / {n}")
    print()
    for s in sorted(per_subject):
        c, t = per_subject[s]
        print(f"  {s:<40} {c:>4}/{t:<4} {c/t:.3f}")
    print(f"  wall {time.time()-started:.1f}s")

    if args.out:
        json.dump({
            "kind": "mmlu-choice",
            "model": args.model_path,
            "jsonl": args.jsonl,
            "questions": n,
            "correct": correct,
            "accuracy": acc,
            "ci95": [centre - half, centre + half],
            "letter_token_ids": letters,
            "prompt_token_mismatches": mismatched,
            "per_subject": {s: {"correct": c, "total": t, "accuracy": c / t}
                            for s, (c, t) in per_subject.items()},
            "rows": rows,
            "wall_s": time.time() - started,
        }, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
