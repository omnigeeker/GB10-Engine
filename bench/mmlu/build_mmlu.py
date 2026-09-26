#!/usr/bin/env python3
"""Fetch and validate the MMLU question set.

Source is the `aviskowron/mmlu-data` GitHub mirror of the original MMLU test
split. The canonical host (hendrycks/test) has dropped its `data/` directory and
the ModelScope mirrors of `cais/mmlu` are empty stubs, so this is what is
reachable from this host; huggingface.co is not.

Rows are validated rather than trusted: a row is kept only if it has exactly six
fields and its answer is one of A-D. Anything else is counted and reported, so a
silently truncated or mis-parsed file cannot pass as a clean question set.
"""
import csv
import json
import os
import urllib.request
from collections import Counter

BASE = "https://raw.githubusercontent.com/aviskowron/mmlu-data/main"
SUBJECTS = [
    "business_ethics", "global_facts", "high_school_government_and_politics",
    "human_sexuality", "management", "marketing", "moral_disputes",
    "moral_scenarios", "philosophy", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "world_religions",
]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    os.makedirs(f"{HERE}/data", exist_ok=True)
    rows = []
    bad = 0
    for subj in SUBJECTS:
        fn = f"{subj}_test.csv"
        path = f"{HERE}/data/{fn}"
        if not os.path.exists(path):
            with urllib.request.urlopen(f"{BASE}/{fn}", timeout=60) as r:
                open(path, "wb").write(r.read())
        with open(path, newline="", encoding="utf-8") as fh:
            for i, r in enumerate(csv.reader(fh)):
                # The original files carry no header: question, A, B, C, D, answer.
                if len(r) != 6:
                    bad += 1
                    continue
                q, a, b, c, d, ans = r
                ans = ans.strip()
                if ans not in ("A", "B", "C", "D"):
                    bad += 1
                    continue
                rows.append({"id": f"{subj}-{i}", "subject": subj,
                             "question": q.strip(),
                             "options": [a.strip(), b.strip(), c.strip(), d.strip()],
                             "answer": "ABCD".index(ans)})

    out = f"{HERE}/mmlu.jsonl"
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    counts = Counter(r["subject"] for r in rows)
    print(f"total questions: {len(rows)}   malformed rows skipped: {bad}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<40} {v}")
    print(f"\nanswer distribution: {dict(sorted(Counter(r['answer'] for r in rows).items()))}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
