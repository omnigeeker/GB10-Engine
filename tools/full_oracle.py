"""Full-model greedy-decode oracle.

The layer fixtures pin down each block; this pins down the *stacking* — the
embedding gather, all 64 layers wired in order with persistent state, the final
zero-centered norm, the NVFP4 `lm_head` and argmax.

The reference runs the real `Qwen3_5ForCausalLM` with every quantized weight
dequantized from this checkpoint. Weights are held in bf16 so the 27B model
fits in unified memory; the dequantized NVFP4 value (E2M1 x fp32 scale) is
rounded to bf16, which is a ~2^-9 relative perturbation per weight. That is why
`docs/TARGETS.md` specifies a two-tier gate: exact agreement against the
NVFP4 reference where available, and a high token-agreement rate against this
bf16 oracle.

Usage:
  # single prompt, legacy flat layout -> fixtures/oracle/<tag>.json
  python3 tools/full_oracle.py [--n 16] [--prompt "..."] [--tag greedy_tokens]

  # many prompts, one model load -> <out-dir>/<tag>/greedy_tokens.json
  python3 tools/full_oracle.py --n 64 --prompts-file tools/oracle_prompts.json \
      --out-dir fixtures/oracle-multi

The directory layout for `--prompts-file` is what `gb10-verify generate
--oracle <out-dir>/<tag>` expects, so each prompt can be checked without
touching the committed single-prompt fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_dequant import Shards  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)

MODEL_DIR = os.environ.get("GB10_MODEL_DIR", "models/Qwen3.8-27B-NVFP4")
OUT = "fixtures/oracle"

EOS_IDS = (248046, 248044)


def build() -> tuple[Qwen3_5TextConfig, Shards]:
    raw = json.load(open(f"{MODEL_DIR}/config.json"))
    cfg = Qwen3_5TextConfig(**raw["text_config"])
    cfg._attn_implementation = "eager"
    return cfg, Shards(MODEL_DIR)


def load_weights(model: torch.nn.Module, shards: Shards) -> None:
    """Fill `model` from the checkpoint, dequantizing quantized tensors.

    The checkpoint names the decoder stack `model.language_model.*`; the
    text-only class names it `model.*`. `lm_head` is top level in both.
    """
    sd = model.state_dict()
    src: dict[str, str] = {}
    for name in sd:
        if name.startswith("model."):
            src[name] = "model.language_model." + name[len("model.") :]
        else:
            src[name] = name

    done = 0
    for name, key in src.items():
        if key not in shards.headers:
            raise RuntimeError(f"checkpoint is missing {key}")
        dtype = shards.headers[key][0]["dtype"]
        if dtype in ("BF16", "F16", "F32"):
            arr = shards.bf16(key)
        else:
            arr = shards.dequant(key[: -len(".weight")])
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if t.shape != sd[name].shape:
            raise RuntimeError(f"{key}: {tuple(t.shape)} != {tuple(sd[name].shape)}")
        sd[name].copy_(t.to(sd[name].dtype))
        done += 1
        if done % 400 == 0:
            print(f"  loaded {done}/{len(src)}", flush=True)
    print(f"  loaded {done}/{len(src)}", flush=True)


def greedy(model, tok, dev, prompt: str, n: int):
    """Greedy decode `n` tokens, recomputing the full sequence each step.

    Slow but it avoids depending on the KV-cache path being bit-identical to a
    one-token-at-a-time decode. `model` carries no state between calls, so the
    same instance is safe to reuse for every prompt.
    """
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(rendered, add_special_tokens=False)["input_ids"]

    out: list[int] = []
    cur = torch.tensor([ids], device=dev)
    with torch.no_grad():
        for _ in range(n):
            logits = model(input_ids=cur).logits[:, -1, :].float()
            nxt = int(logits.argmax(-1).item())
            # Stop *before* emitting EOS, which is what `gb10-verify generate`
            # does (`if tok.is_eos(next) { break; }` precedes the push). Emitting
            # it here instead would leave the oracle one token longer than the
            # engine and make an agreeing run look like a length mismatch.
            if nxt in EOS_IDS:
                break
            out.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=dev)], dim=1)
    return rendered, ids, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--tag", default="greedy_tokens")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument(
        "--prompts-file",
        default=None,
        help="JSON list of {tag, prompt}; writes <out-dir>/<tag>/greedy_tokens.json",
    )
    args = ap.parse_args()

    if args.prompts_file:
        jobs = [(j["tag"], j["prompt"]) for j in json.load(open(args.prompts_file))]
        if not jobs:
            raise SystemExit(f"{args.prompts_file}: no prompts")
    else:
        jobs = [(args.tag, args.prompt)]

    cfg, shards = build()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("instantiating model in bf16 ...", flush=True)
    model = Qwen3_5ForCausalLM._from_config(cfg, torch_dtype=torch.bfloat16)
    model.eval()
    print("loading weights ...", flush=True)
    with torch.no_grad():
        load_weights(model, shards)
    model.to(dev)
    torch.cuda.empty_cache()

    n_exact = 0
    for i, (tag, prompt) in enumerate(jobs, 1):
        t0 = time.time()
        rendered, ids, out = greedy(model, tok, dev, prompt, args.n)
        if args.prompts_file:
            path = os.path.join(args.out_dir, tag, "greedy_tokens.json")
        else:
            path = os.path.join(args.out_dir, f"{tag}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "prompt": prompt,
                    "rendered": rendered,
                    "prompt_ids": ids,
                    "n_new": args.n,
                    "tokens": out,
                    "reference": "Qwen3_5ForCausalLM, weights dequantized from NVFP4 to bf16",
                    "precision": "bf16",
                },
                f,
                indent=2,
            )
        hit_eos = bool(out) and out[-1] in EOS_IDS
        print(
            f"[{i}/{len(jobs)}] {tag:<14} prompt {len(ids):>3} tok  "
            f"out {len(out):>3} tok{' (eos)' if hit_eos else '     '}  "
            f"{time.time() - t0:5.1f}s  -> {path}",
            flush=True,
        )
        print(f"    ids:  {out[:24]}", flush=True)
        if len(out) == args.n:
            n_exact += 1

    print(f"wrote {len(jobs)} oracle(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
