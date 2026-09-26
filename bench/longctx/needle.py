#!/usr/bin/env python3
"""Needle-in-a-haystack over a range of context lengths, against a live server.

The needle sits at a fixed fraction of the way through a repeated filler, so a
failure at one length but not another means the context handling broke at that
scale rather than the model simply not finding it. Reports the server's own
prompt_tokens, how many prefill chunks that implies, and the wall time, because
the interesting question at long context is not just "is the answer right" but
"how long did it cost".
"""
import os
import sys
import time

import requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
NEEDLE = "The access code for the vault is 74921."
FILLER = (
    "The archive room contains many boxes of old records. Each box is labelled "
    "with a number and a date, and the shelves are dusted every second Tuesday. "
)
CHUNK = 2048

# Depths as a fraction of the document, so a pass is not an artefact of the
# needle sitting at the very end where recency alone could carry it. Override
# with `DEPTHS`: a 256K prompt costs hours, so three depths there is not
# affordable, while the cheap sizes still get all three.
DEPTHS = [float(x) for x in os.environ.get("DEPTHS", "0.1,0.5,0.9").split(",")]


def ask(prompt, max_tokens=24, timeout=172800):
    # 48 h, because the read timeout is what bounds the *whole* wait on a
    # non-streaming request: the server sends nothing until the prefill is done.
    # A 256K prefill is ~12 h and would be abandoned by the old 20000 s.
    r = requests.post(
        URL,
        json={
            "model": "m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "enable_thinking": False,
        },
        timeout=timeout,
    )
    d = r.json()
    if "choices" not in d:
        return None, None, d
    return d["choices"][0]["message"]["content"], d["usage"], d


def build(reps, depth):
    """Filler with the needle at `depth` through it."""
    cut = int(reps * depth)
    return (
        "Read the following document and answer the question at the end.\n\n"
        + FILLER * cut
        + "\n\n"
        + NEEDLE
        + "\n\n"
        + FILLER * (reps - cut)
        + "\n\nQuestion: What is the access code for the vault? "
        "Answer with just the number."
    )


def main():
    reps_list = [int(x) for x in sys.argv[1:]] or [400, 900]
    print(f"{'reps':>5} {'depth':>6} {'prompt':>8} {'chunks':>7} {'secs':>8} {'ms/tok':>7}  result")
    failures = 0
    for reps in reps_list:
        for depth in DEPTHS:
            prompt = build(reps, depth)
            t0 = time.time()
            c, u, d = ask(prompt)
            el = time.time() - t0
            if u is None:
                print(f"{reps:>5} {depth:>6} {'--':>8} {'--':>7} {el:>8.1f} {'--':>7}  ERROR {d}")
                failures += 1
                continue
            pt = u["prompt_tokens"]
            ok = "74921" in (c or "")
            failures += 0 if ok else 1
            print(
                f"{reps:>5} {depth:>6} {pt:>8} {(pt + CHUNK - 1) // CHUNK:>7} "
                f"{el:>8.1f} {el / pt * 1000:>7.0f}  {'PASS' if ok else 'MISS'} {(c or '')[:24]!r}",
                flush=True,
            )
    print(f"\n{len(reps_list) * len(DEPTHS) - failures}/{len(reps_list) * len(DEPTHS)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
