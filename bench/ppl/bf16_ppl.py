#!/usr/bin/env python3
"""BF16 reference perplexity on wikitext-2, matching llama.cpp's protocol.

Replicates the default (`--ppl-stride 0`) path of tools/perplexity/perplexity.cpp:

  n_ctx    = 512
  first    = n_ctx / 2                    -> 256
  n_chunk  = n_tokens / n_ctx             -> whole windows, no overlap
  per chunk, for i in 0 .. n_ctx-first-2  -> 255 predictions
      row   = first + i                   -> logits at position 256..510
      tgt   = window[first + i + 1]       -> window[257..511]
      nll  += logsumexp(row) - row[tgt]

`add_bos` is false for this checkpoint's vocab (verified: llama-tokenize emits
no 248044 for the same file), so the windows are used verbatim.

Token ids come from llama-tokenize, so the token stream is identical by
construction to the llama.cpp baseline. This is a weight-only comparison.
"""
import argparse
import json
import math
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokens", required=True, help="whitespace/py-list separated token ids")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--chunks", type=int, default=-1, help="-1 = all")
    p.add_argument("--out", default=None)
    p.add_argument("--nll-out", default=None, help="per-chunk mean nll")
    args = p.parse_args()

    with open(args.tokens) as fh:
        raw = fh.read().strip()
    # llama-tokenize --ids prints a python list, then a "Total number of
    # tokens: N" line. Keep only what is between the brackets.
    a, b = raw.find("["), raw.find("]")
    body = raw[a + 1:b] if (a >= 0 and b > a) else raw
    ids = [int(x) for x in body.replace(",", " ").split()]
    print(f"tokens: {len(ids)}", flush=True)

    n_ctx = args.ctx
    first = n_ctx // 2
    per_chunk = n_ctx - 1 - first
    n_chunk_max = len(ids) // n_ctx
    n_chunk = n_chunk_max if args.chunks < 0 else min(args.chunks, n_chunk_max)
    print(f"n_ctx={n_ctx} first={first} preds/chunk={per_chunk} chunks={n_chunk}", flush=True)

    from transformers import AutoModelForImageTextToText

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    dev = "cuda"
    nll = 0.0
    count = 0
    per_chunk_nll = []

    with torch.no_grad():
        for i in range(n_chunk):
            start = i * n_ctx
            window = ids[start:start + n_ctx]
            inp = torch.tensor([window], dtype=torch.long, device=dev)
            t1 = time.time()
            out = model(input_ids=inp, use_cache=False)
            rows = out.logits[0, first:first + per_chunk].float()
            tgt = torch.tensor(window[first + 1:first + 1 + per_chunk],
                               dtype=torch.long, device=dev)
            lse = torch.logsumexp(rows, dim=-1)
            picked = rows.gather(1, tgt[:, None]).squeeze(1)
            d = (lse - picked).double()
            c_nll = d.sum().item()
            nll += c_nll
            count += per_chunk
            per_chunk_nll.append(c_nll / per_chunk)
            if i % 10 == 0 or i == n_chunk - 1:
                print(f"[{i}] ppl={math.exp(nll/count):.4f} "
                      f"({time.time()-t1:.2f}s/pass, {count} preds)", flush=True)

    ppl = math.exp(nll / count)
    var = sum((x - nll / count) ** 2 for x in per_chunk_nll) / max(1, len(per_chunk_nll) - 1)
    print(f"Final estimate: PPL = {ppl:.4f}", flush=True)

    result = {
        "kind": "bf16-reference-ppl",
        "model": args.model_path,
        "dtype": "bfloat16",
        "n_ctx": n_ctx,
        "first": first,
        "preds_per_chunk": per_chunk,
        "chunks": n_chunk,
        "tokens": len(ids),
        "predictions": count,
        "mean_nll": nll / count,
        "ppl": ppl,
        "ppl_stderr_chunk": math.sqrt(var / len(per_chunk_nll)) if per_chunk_nll else None,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    if args.nll_out:
        with open(args.nll_out, "w") as fh:
            json.dump(per_chunk_nll, fh)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
