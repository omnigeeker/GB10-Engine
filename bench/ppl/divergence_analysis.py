#!/usr/bin/env python3
"""Classify every engine-vs-oracle divergence: near-tie or defect?

The multi-prompt check reports *where* the engine first disagrees with the
NVFP4-dequantized reference. This says *how close* that decision was, and
whether the reference is even self-consistent at that horizon.

Three things per prompt, in one model load:

  margins   top-1/top-2 logit gap at the engine's first-divergence step.
            At logit magnitude ~23 a bf16 ulp is 0.125, and the reference's
            logits land on exactly that grid, so a margin of one ulp means the
            decision is decided by the last representable bit.
  control   the same model, same prompt, same argmax, decoded incrementally
            against a KV cache instead of by recomputing the sequence. The
            oracle avoids the cache path on purpose; the engine uses it. If the
            reference disagrees with itself here, long-horizon token-exactness
            is a property of the decode strategy, not of the engine.
  oracle    recompute vs the committed fixture, as a sanity check.

    python3 bench/ppl/divergence_analysis.py --tags p02-math,p03-code,p04-chinese
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "tools")
import full_oracle as fo  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402


def greedy_recompute(model, ids, n, dev, topk=5):
    """Recompute the whole sequence each step; record the top-k each time.

    Picking uses `argmax`, exactly as `tools/full_oracle.py` does. Picking via
    `topk(...,1)` instead selects a *different* token at an exact tie, which is
    how an earlier revision of this script came to analyse a different
    trajectory than the one the engine was compared against.
    """
    out, steps = [], []
    cur = torch.tensor([ids], device=dev)
    with torch.no_grad():
        for step in range(n):
            logits = model(input_ids=cur).logits[:, -1, :].float()[0]
            picked = int(logits.argmax(-1).item())
            top = torch.topk(logits, topk)
            toks = [int(t) for t in top.indices]
            vals = [float(v) for v in top.values]
            margin = vals[0] - vals[1]
            # Every token the reference scores exactly as high as its own top-1.
            tied = [t for t, v in zip(toks, vals) if v == vals[0]]
            steps.append(
                {"step": step, "picked": picked, "margin": margin, "tied": tied,
                 "top_tokens": toks, "top_logits": vals}
            )
            if picked in fo.EOS_IDS:
                break
            out.append(picked)
            cur = torch.cat([cur, torch.tensor([[picked]], device=dev)], dim=1)
    return out, steps


def greedy_cached(model, ids, n, dev):
    """Incremental decode against a KV cache.

    `cur` must accumulate the generated tokens: after the first pass only the
    newest token is fed, so failing to append re-feeds the last prompt token
    every step and the two paths diverge at step 1 for a reason that has
    nothing to do with the cache.
    """
    out = []
    cur = torch.tensor([ids], device=dev)
    with torch.no_grad():
        past = None
        for _ in range(n):
            inp = cur if past is None else cur[:, -1:]
            o = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[:, -1, :].float().argmax(-1).item())
            if nxt in fo.EOS_IDS:
                break
            out.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=dev)], dim=1)
    return out


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, sum(1 for j in range(n) if a[j] == b[j]), n
    return None, sum(1 for j in range(n) if a[j] == b[j]), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="p02-math,p03-code,p04-chinese,p06-kyoto,p08-longctx")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--engine-json", default="bench/ppl/multi-prompt.json")
    ap.add_argument("--out", default="bench/ppl/divergence-analysis.json")
    args = ap.parse_args()

    engine = {}
    if os.path.exists(args.engine_json):
        for r in json.load(open(args.engine_json))["rows"]:
            engine[r["oracle"]] = r

    tags = [t for t in args.tags.split(",") if t]
    fixtures = {
        t: json.load(open(f"fixtures/oracle-multi/{t}/greedy_tokens.json")) for t in tags
    }

    cfg, shards = fo.build()
    model = Qwen3_5ForCausalLM._from_config(cfg, torch_dtype=torch.bfloat16)
    model.eval()
    print("loading weights ...", flush=True)
    with torch.no_grad():
        fo.load_weights(model, shards)
    model.to("cuda")
    torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(fo.MODEL_DIR)

    results = []
    for tag in tags:
        prompt = fixtures[tag]["prompt"]
        want = fixtures[tag]["tokens"]
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(rendered, add_special_tokens=False)["input_ids"]

        rec, steps = greedy_recompute(model, ids, args.n, "cuda")
        cac = greedy_cached(model, ids, args.n, "cuda")

        f_rec_fix, a_rec_fix, n_rec_fix = first_diff(rec, want)
        f_rec_cac, a_rec_cac, n_rec_cac = first_diff(rec, cac)

        eng = engine.get(tag, {})
        eng_bad = (eng.get("first_bad") or {}).get("index")
        margin_at = None
        tied_at = None
        eng_pick = (eng.get("first_bad") or {}).get("engine")
        if eng_bad is not None and eng_bad < len(steps):
            margin_at = steps[eng_bad]["margin"]
            tied_at = steps[eng_bad]["tied"]
        # Did the engine pick a token the reference scores exactly as high as
        # its own top-1? If so the "divergence" is a tie-break, not an error.
        engine_picked_a_tied_token = (
            tied_at is not None and eng_pick is not None and eng_pick in tied_at
        )

        rec_out = {
            "tag": tag,
            "prompt_tokens": len(ids),
            "engine_first_divergence": eng_bad,
            "engine_token": eng_pick,
            "oracle_token": (eng.get("first_bad") or {}).get("oracle"),
            "margin_at_engine_divergence": margin_at,
            "tied_tokens_at_divergence": tied_at,
            "engine_picked_a_tied_token": engine_picked_a_tied_token,
            "ties_over_run": sum(1 for s in steps if s["margin"] == 0.0),
            "steps_over_run": len(steps),
            "recompute_vs_fixture_first": f_rec_fix,
            "cached_vs_recompute_first": f_rec_cac,
            "cached_vs_recompute_agree": a_rec_cac,
            "cached_vs_recompute_n": n_rec_cac,
            "min_margin_over_run": min((s["margin"] for s in steps), default=None),
            "steps": steps,
        }
        results.append(rec_out)

        m = f"{margin_at:.4f}" if margin_at is not None else "n/a"
        print(
            f"{tag:<14} engine diverged @{str(eng_bad):>4} margin {m:>8} "
            f"tied={str(engine_picked_a_tied_token):<5}  "
            f"cached-vs-recompute first {str(f_rec_cac):>4} "
            f"({a_rec_cac}/{n_rec_cac})",
            flush=True,
        )

    print()
    print("at the engine's first divergence (bf16 ulp at |logit|~23 is 0.125):")
    for r in results:
        if r["margin_at_engine_divergence"] is not None:
            ulps = r["margin_at_engine_divergence"] / 0.125
            print(
                f"  {r['tag']:<14} step {r['engine_first_divergence']:>3}  "
                f"margin {r['margin_at_engine_divergence']:.4f} (~{ulps:.2f} bf16 ulp)  "
                f"engine {r['engine_token']} vs oracle {r['oracle_token']}  "
                f"engine token tied with oracle top-1: {r['engine_picked_a_tied_token']}"
            )
    tot_t = sum(r["ties_over_run"] for r in results)
    tot_s = sum(r["steps_over_run"] for r in results)
    print(
        f"\nexact ties in the reference's own logits: {tot_t}/{tot_s} steps"
        + (f" ({100*tot_t/tot_s:.1f}%)" if tot_s else "")
    )

    with open(args.out, "w") as fh:
        json.dump({"prompts": results}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
