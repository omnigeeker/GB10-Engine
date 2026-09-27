#!/usr/bin/env python3
"""Cold/warm TTFT and OTPS against an OpenAI-compatible streaming endpoint.

Three numbers per trial, all from one streaming request pair:

  cold TTFT  full prefill of a prefix the server has never seen
  warm TTFT  the same request again, so a prefix cache can skip the prefill
  OTPS       content deltas after the first, over the time between them

Both trials send the *identical* request, so the only difference is what the
server already holds. That is the whole point: a server with no prefix cache
must answer warm in the same time as cold, and the gap between the two is
exactly the feature being measured.

Each trial gets a unique leading marker. Without it a second trial would be
warm by accident on any server that does cache, and the cold number would be a
cache hit. The marker is outside the repeated body, so token counts stay
comparable across trials.

Usage:
  ttft.py --port 8080 --reps 1054 --trials 2 --max-tokens 128 --label gb10
"""
import argparse
import json
import os
import time

import requests

FILLER = (
    "The archive room contains many boxes of old records. Each box is labelled "
    "with a number and a date, and the shelves are dusted every second Tuesday. "
)
# The question asks for a long, deterministic, open-ended generation on
# purpose. A question with a one-word answer makes the model stop after a
# couple of tokens, and OTPS measured over 2-3 intervals is noise. Listing
# integers has no natural stopping point, so both servers decode for the full
# max_tokens and OTPS is computed over ~100 intervals.
QUESTION = (
    "\n\nQuestion: write out the integers from 1 to 300, one per line, and "
    "nothing else."
)


def build_prompt(marker: str, reps: int) -> str:
    return (
        f"Session {marker}. Read the following and then answer.\n\n"
        + FILLER * reps
        + QUESTION
    )


def stream_once(url: str, prompt: str, max_tokens: int, timeout: float):
    """One streaming request.

    Returns (ttft_s, otps, n_tokens, prompt_tokens, text_chars). `ttft_s` is
    measured to the first delta that carries any text -- content or reasoning --
    because that is the moment the model started producing. OTPS uses only the
    gaps *after* that first token, so it is decode throughput and does not
    absorb the prefill.
    """
    t0 = time.time()
    first = None
    times = []
    n_tok = 0
    chars = 0
    prompt_tokens = None

    with requests.post(
        url,
        json={
            "model": "m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "enable_thinking": False,
            "stream": True,
            "temperature": 0,
            # Ask for the prompt token count in the stream. llama.cpp only
            # reports usage when asked; without this the comparison cannot show
            # that both servers were handed the same prompt.
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=timeout,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(d.get("usage"), dict):
                prompt_tokens = d["usage"].get("prompt_tokens", prompt_tokens)
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                text = delta.get("content") or delta.get("reasoning_content") or ""
                if not text:
                    continue
                now = time.time()
                if first is None:
                    first = now
                times.append(now)
                n_tok += 1
                chars += len(text)

    if first is None:
        raise RuntimeError("no content deltas received")
    otps = None
    if len(times) > 1 and times[-1] > first:
        otps = (len(times) - 1) / (times[-1] - first)
    return first - t0, otps, n_tok, prompt_tokens, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=172800)
    ap.add_argument("--label", default="server")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    print(f"# {args.label}  reps={args.reps} trials={args.trials} max_tokens={args.max_tokens}")
    print(f"{'trial':>5} {'prompt':>8} {'cold_ttft':>10} {'warm_ttft':>10} "
          f"{'otps_cold':>9} {'otps_warm':>9} {'tok':>4}")

    for i in range(args.trials):
        marker = f"{args.label}-{os.getpid()}-{i}-{int(time.time())}"
        prompt = build_prompt(marker, args.reps)

        cold = stream_once(url, prompt, args.max_tokens, args.timeout)
        warm = stream_once(url, prompt, args.max_tokens, args.timeout)

        c_ttft, c_otps, c_tok, c_pt, _ = cold
        w_ttft, w_otps, w_tok, w_pt, _ = warm
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        print(f"{i:>5} {c_pt if c_pt is not None else '?':>8} {c_ttft:>10.2f} "
              f"{w_ttft:>10.2f} {fmt(c_otps):>9} {fmt(w_otps):>9} {c_tok:>4}",
              flush=True)


if __name__ == "__main__":
    main()
