#!/usr/bin/env python3
"""Score the same MMLU questions through llama.cpp's server.

llama.cpp is the other NVFP4 implementation on this machine, so it is the
useful external check: if the engine's accuracy matches the bf16 model and
llama.cpp's does not, the engine is not the one losing points.

Method. The server is asked for one token after "Answer:" with a large
`n_probs`, and the answer is the letter among A-D with the highest returned
log-probability. Log-probabilities are monotonic in logits, so argmax is the
same decision the engine and the reference make.

This server version answers with OpenAI-style keys -- `logprob` and
`top_logprobs` -- not the older `prob`/`probs` pair, and it returns
**pre-sampling** values (`post_sampling_probs` defaults to false), so the
sampler chain cannot distort the ranking.

The sampler is neutralised (`top_k: 0`, `top_p: 1`, `min_p: 0`,
`repeat_penalty: 1`, `temperature: 0`) anyway. Whether the candidate list was
wide enough is not assumed: the script reports the fraction of questions whose
top-`n_probs` list actually contained all four letter tokens. If that is not
~100 %, the comparison is invalid and the script says so.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import render  # noqa: E402


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8899")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--n-probs", type=int, default=100)
    ap.add_argument("--engine-json", default=None,
                   help="engine result, to cross-check the answer-letter token ids")
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    if args.limit >= 0:
        questions = questions[:args.limit]
    print(f"questions: {len(questions)}", flush=True)

    # The four answer letters, as llama.cpp tokenises them. " A".." D" are
    # single tokens; the ids are the same as the engine's by construction
    # (verified: engine tokenizer vs llama-tokenize, 0 mismatches / 297,054).
    letters = None

    correct = 0
    per_subject = {}
    rows = []
    missing = 0
    started = time.time()

    for qi, q in enumerate(questions):
        r = post(f"{args.url}/completion", {
            "prompt": render(q), "n_predict": 1, "n_probs": args.n_probs,
            "temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
            "repeat_penalty": 1.0, "cache_prompt": False, "stream": False,
        })
        cp = r["completion_probabilities"][0]["top_logprobs"]
        lp = {p["id"]: p["logprob"] for p in cp}
        if letters is None:
            # Map by token string, in A/B/C/D order. Sorting the ids would
            # silently permute the letters and score the wrong answer: this
            # vocabulary has " A"=357, " B"=417, " C"=351, " D"=414, so the
            # sorted order is not the letter order.
            by_tok = {p["token"]: p["id"] for p in cp
                      if p["token"] in (" A", " B", " C", " D")}
            assert len(by_tok) == 4, f"expected 4 letters, got {sorted(by_tok)}"
            letters = [by_tok[t] for t in (" A", " B", " C", " D")]
            print(f'answer letters " A".." D" -> token ids {letters}', flush=True)
            if args.engine_json:
                eng = json.load(open(args.engine_json))["letter_token_ids"]
                assert eng == letters, (
                    f"llama.cpp letter ids {letters} != engine letter ids {eng}; "
                    "the two are not scoring the same tokens")
        if not all(t in lp for t in letters):
            missing += 1
        scores = [lp.get(t, float("-inf")) for t in letters]
        pick = max(range(4), key=lambda i: scores[i])
        ok = pick == q["answer"]
        correct += ok
        e = per_subject.setdefault(q["subject"], [0, 0])
        e[1] += 1
        e[0] += ok
        rows.append({"id": q["id"], "subject": q["subject"], "gold": q["answer"],
                     "pred": pick, "scores": scores})
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
    print(f"llama.cpp MMLU accuracy: {correct}/{n} = {acc:.4f}")
    print(f"  95% CI: [{centre-half:.4f}, {centre+half:.4f}]")
    print(f"  questions where the top-{args.n_probs} list omitted a letter: "
          f"{missing}/{n} ({missing/n:.2%})")
    print()
    for s in sorted(per_subject):
        c, t = per_subject[s]
        print(f"  {s:<40} {c:>4}/{t:<4} {c/t:.3f}")
    print(f"  wall {time.time()-started:.1f}s")

    if args.out:
        json.dump({
            "kind": "mmlu-choice",
            "model": "llama.cpp " + args.url,
            "jsonl": args.jsonl,
            "questions": n,
            "correct": correct,
            "accuracy": acc,
            "ci95": [centre - half, centre + half],
            "letter_token_ids": letters,
            "letter_missing": missing,
            "n_probs": args.n_probs,
            "per_subject": {s: {"correct": c, "total": t, "accuracy": c / t}
                            for s, (c, t) in per_subject.items()},
            "rows": rows,
            "wall_s": time.time() - started,
        }, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
