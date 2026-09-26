#!/usr/bin/env python3
"""Compare MMLU runs question-by-question.

Accuracy alone is too coarse to answer "did it drop points": two runs on 3240
questions can differ by 0.5 point and still be the same model. Because every run
answers the *same* questions, the right test is paired -- McNemar's on the
discordant pairs -- which cancels the question difficulty that dominates the
unpaired variance.

  b = run A wrong, run B right      (B gained)
  c = run A right, run B wrong      (B lost)
  chi2 = (|b - c| - 1)^2 / (b + c)  (continuity corrected), 1 df
"""
import argparse
import json
import math


def mcnemar(b, c):
    """Two-sided McNemar p-value with continuity correction."""
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))


def load(path):
    d = json.load(open(path))
    return d, {r["id"]: r["pred"] for r in d["rows"]}, {r["id"]: r["gold"] for r in d["rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="result JSONs, first is the baseline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loaded = [load(p) for p in args.runs]
    labels = [p.split("/")[-1].replace(".json", "") for p in args.runs]

    print("=" * 78)
    print("MMLU accuracy")
    print("=" * 78)
    print(f"{'run':<28} {'correct':>9} {'total':>7} {'accuracy':>10} {'95% CI':>20}")
    for lab, (d, preds, _) in zip(labels, loaded):
        lo, hi = d["ci95"]
        print(f"{lab:<28} {d['correct']:>9} {d['questions']:>7} {d['accuracy']:>10.4f} "
              f"{f'[{lo:.4f}, {hi:.4f}]':>20}")

    # Every run must have answered the same questions or pairing is meaningless.
    base_d, base_preds, gold = loaded[0]
    ids = set(base_preds)
    for lab, (d, preds, g) in zip(labels[1:], loaded[1:]):
        if set(preds) != ids or g != gold:
            raise SystemExit(f"FATAL: {lab} answered a different question set")

    print()
    print("=" * 78)
    print(f"Paired comparison vs {labels[0]} (McNemar, same {len(ids)} questions)")
    print("=" * 78)
    print(f"{'run':<28} {'delta':>9} {'gained':>8} {'lost':>6} {'agree':>8} {'p':>10}")
    out = {}
    for lab, (d, preds, _) in zip(labels[1:], loaded[1:]):
        b = sum(1 for i in ids if base_preds[i] != gold[i] and preds[i] == gold[i])
        c = sum(1 for i in ids if base_preds[i] == gold[i] and preds[i] != gold[i])
        agree = sum(1 for i in ids if base_preds[i] == preds[i])
        delta = d["accuracy"] - base_d["accuracy"]
        p = mcnemar(b, c)
        print(f"{lab:<28} {delta:>+9.4f} {b:>8} {c:>6} {agree/len(ids):>8.4f} {p:>10.4f}")
        out[lab] = {"delta_vs_baseline": delta, "gained": b, "lost": c,
                    "agreement": agree / len(ids), "mcnemar_p": p}

    print()
    print("=" * 78)
    print("Per-subject accuracy")
    print("=" * 78)
    subs = sorted({r["subject"] for r in base_d["rows"]})
    hdr = f"{'subject':<38}" + "".join(f"{lab[:14]:>16}" for lab in labels)
    print(hdr)
    for s in subs:
        line = f"{s:<38}"
        for (d, _, _) in loaded:
            ps = d["per_subject"].get(s)
            line += f"{ps['accuracy']:>16.3f}" if ps else f"{'-':>16}"
        print(line)

    if args.out:
        json.dump({"labels": labels,
                   "accuracy": {lab: d["accuracy"] for lab, (d, _, _) in zip(labels, loaded)},
                   "paired_vs_baseline": out},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
