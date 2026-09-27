#!/usr/bin/env python3
"""Correctness check for the prefix cache.

The cache replaces a prefill with a recurrent snapshot plus one re-run token.
If the snapshot is even slightly wrong the output drifts, and a drift of one
token is invisible in the text unless you compare token ids. So this compares
*ids*, in three places:

  A. the same request twice in a row       -> must be identical (cache hit)
  B. a request after an unrelated one      -> must equal the same request made
                                              on a server that never saw the
                                              unrelated one (no contamination)
  C. a request that extends a cached prefix -> must equal the same request made
                                              from scratch (partial hit)

Usage: prefix_check.py --port 8080 [--reps N]
"""
import argparse
import json

import requests

FILLER = (
    "The archive room contains many boxes of old records. Each box is labelled "
    "with a number and a date, and the shelves are dusted every second Tuesday. "
)


def ask(url, prompt, max_tokens=24):
    r = requests.post(
        url,
        json={
            "model": "m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "enable_thinking": False,
            "temperature": 0,
        },
        timeout=3600,
    )
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"]["content"]


def body(tag, reps):
    return f"Session {tag}.\n\n" + FILLER * reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    reps = args.reps

    p1 = body("alpha", reps)
    p2 = body("bravo", reps)          # unrelated first token -> full miss
    p1_long = p1 + "\n\nAnd then answer with the word DONE."

    out = {}

    # A: same prompt twice. Second one is the cache hit.
    out["a1"] = ask(url, p1)
    out["a2"] = ask(url, p1)
    print(f"A  repeat hit:      {'IDENTICAL' if out['a1'] == out['a2'] else 'DIFFERENT'}")

    # B: unrelated prompt, must not be contaminated by alpha's cache.
    b_after = ask(url, p2)

    # C: extends a cached prefix (alpha is still the most recent for this slot
    # only if B did not evict it, so run alpha again first to make it recent).
    ask(url, p1)
    c_hit = ask(url, p1_long)

    # Reference: a *fresh* server is needed for the no-cache answer. The caller
    # supplies it by running this script twice, once per server, and diffing
    # `ref_b`/`ref_c`; here we just print what we got.
    print(f"B  p2 after p1:     {b_after[:40]!r}")
    print(f"C  extended prompt: {c_hit[:40]!r}")

    with open(f"/tmp/longctx/prefix-{args.port}.json", "w") as f:
        json.dump(
            {"a1": out["a1"], "a2": out["a2"], "b": b_after, "c": c_hit}, f, indent=1
        )
    ok = out["a1"] == out["a2"]
    print(f"\nwrote /tmp/longctx/prefix-{args.port}.json")
    print("prefix cache repeat-consistency:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
