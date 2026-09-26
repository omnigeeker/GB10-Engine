#!/usr/bin/env python3
"""Time a chunked prefill at a few prompt sizes, reading the server's own
per-chunk timings from its log.

The point is the *shape*: whether the per-chunk cost is flat (linear overall) or
grows with the number of cached keys (quadratic). The server prints per-chunk
timings when started with GB10_TIMING=1.
"""
import sys
import time

import requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
FILLER = (
    "The archive room contains many boxes of old records. Each box is labelled "
    "with a number and a date, and the shelves are dusted every second Tuesday. "
)
NEEDLE = "The access code for the vault is 74921."

# ~31 tokens per filler repeat.
REPS = [400, 800]


def ask(reps):
    prompt = (
        "Read this and answer.\n\n"
        + FILLER * reps
        + "\n\n"
        + NEEDLE
        + "\n\nQuestion: What is the access code for the vault? "
        "Answer with just the number."
    )
    t0 = time.time()
    d = requests.post(
        URL,
        json={
            "model": "m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "enable_thinking": False,
        },
        timeout=30000,
    ).json()
    el = time.time() - t0
    if "choices" not in d:
        print(f"reps {reps}: ERROR {d}")
        return
    pt = d["usage"]["prompt_tokens"]
    c = d["choices"][0]["message"]["content"]
    print(
        f"reps {reps:>4}: prompt {pt:>6} tok, wall {el:>7.1f}s "
        f"({el / pt * 1000:>5.1f} ms/tok) -> {c[:16]!r} {'PASS' if '74921' in c else 'MISS'}",
        flush=True,
    )


for r in (int(x) for x in sys.argv[1:]) or REPS:
    ask(r)
