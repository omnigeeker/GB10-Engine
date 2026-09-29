#!/usr/bin/env python3
"""Needle-in-a-haystack over a range of context lengths, against a live server.

The needle sits at a fixed fraction of the way through the haystack, so a
failure at one length but not another means the context handling broke at that
scale rather than the model simply not finding it. Reports the server's own
prompt_tokens, how many prefill chunks that implies, and the wall time, because
the interesting question at long context is not just "is the answer right" but
"how long did it cost".

Two things here exist specifically to make a comparison against llama.cpp
trustworthy, because getting either wrong already produced a false "both engines
fail" conclusion once:

  * `NEEDLE_URL` selects the server. The default is gb10 on 8080; llama.cpp runs
    on 8899. Hardcoding one port made "run both engines" easy to get wrong.
  * the answer is read from `content` **or** `reasoning_content`. llama.cpp
    routes the model's text to `reasoning_content` and leaves `content` empty,
    so reading only `content` scores llama.cpp as failing every leg. gb10 puts
    everything in `content`.

And one to make the haystack a fair test:

  * `NEEDLE_HAYSTACK=wiki` uses real prose from bench/ppl/wiki.test.raw instead
    of repeated filler. Filler is a degenerate input -- it is the same sentence
    over and over -- so a failure on filler alone cannot distinguish "the model
    cannot retrieve" from "the model degenerates under repetition".
"""
import os
import sys
import time

import requests

URL = os.environ.get(
    "NEEDLE_URL", "http://127.0.0.1:8080/v1/chat/completions"
)
NEEDLE = "The access code for the vault is 74921."
FILLER = (
    "The archive room contains many boxes of old records. Each box is labelled "
    "with a number and a date, and the shelves are dusted every second Tuesday. "
)
# Must match PREFILL_CHUNK in crates/gb10-server/src/main.rs -- it is only used
# to report how many chunks the prompt is prefilled in, i.e. how many passes the
# prompt makes through the chunked prefill path under test. It was 2048 while
# the server had moved to 8192, so the column understated the count 4x.
CHUNK = int(os.environ.get("NEEDLE_CHUNK", "8192"))

# Depths as a fraction of the document, so a pass is not an artefact of the
# needle sitting at the very end where recency alone could carry it. Override
# with `DEPTHS`: a 256K prompt costs hours, so three depths there is not
# affordable, while the cheap sizes still get all three.
DEPTHS = [float(x) for x in os.environ.get("DEPTHS", "0.1,0.5,0.9").split(",")]

HAYSTACK = os.environ.get("NEEDLE_HAYSTACK", "filler")

# How many tokens the server may generate. 24 is enough for gb10, which answers
# with the number directly under `enable_thinking: False`. It is NOT enough for
# llama.cpp on this model: llama.cpp puts a reasoning preamble in
# `reasoning_content` regardless of that flag, and 24 tokens lands mid-thought
# ("The user provided a very..."), which scores as a MISS even though the model
# was retrieving fine. That produced a false "llama.cpp also fails" reading
# once already -- compare with a budget big enough for the thought to finish.
MAX_TOKENS = int(os.environ.get("NEEDLE_MAX_TOKENS", "24"))
# Only for the wiki haystack: how many characters of prose to use. Natural
# English is roughly 4 chars/token, so the default is ~35K tokens.
WIKI_CHARS = int(os.environ.get("NEEDLE_CHARS", "140000"))
WIKI_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ppl", "wiki.test.raw"
)


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
    msg = d["choices"][0]["message"]
    # llama.cpp answers in `reasoning_content`; gb10 answers in `content`.
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return text, d["usage"], d


def _wiki():
    with open(WIKI_PATH, encoding="utf-8", errors="replace") as f:
        return f.read(WIKI_CHARS)


def build(reps, depth):
    """The haystack with the needle at `depth` through it."""
    head = "Read the following document and answer the question at the end.\n\n"
    tail = (
        "\n\nQuestion: What is the access code for the vault? "
        "Answer with just the number."
    )
    if HAYSTACK == "wiki":
        doc = _wiki()
        cut = max(1, int(len(doc) * depth))
        return head + doc[:cut] + "\n\n" + NEEDLE + "\n\n" + doc[cut:] + tail
    cut = int(reps * depth)
    return head + FILLER * cut + "\n\n" + NEEDLE + "\n\n" + FILLER * (reps - cut) + tail


def main():
    reps_list = [int(x) for x in sys.argv[1:]] or [400, 900]
    print(f"# url={URL} haystack={HAYSTACK} max_tokens={MAX_TOKENS}")
    print(f"{'reps':>5} {'depth':>6} {'prompt':>8} {'chunks':>7} {'secs':>8} {'ms/tok':>7}  result")
    failures = 0
    for reps in reps_list:
        for depth in DEPTHS:
            prompt = build(reps, depth)
            t0 = time.time()
            c, u, d = ask(prompt, MAX_TOKENS)
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
