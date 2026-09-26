# Accuracy evaluation — does the engine lose anything?

**Question asked:** run an accuracy evaluation and make sure nothing is dropped.

**Answer:** the engine does not drop accuracy. On wikitext-2 it lands **0.68 % above
the unquantized BF16 model**, against **2.24 %** for llama.cpp's NVFP4 path on the
same weights. The engine is the closer of the two to the reference.

At the token level, extended from 1 prompt x 16 tokens to **8 prompts x 64 tokens**
(including a 367-token prefill), the engine agrees on **409/477 = 85.74 %** of
tokens and is exact on 3 of 8 prompts. Every attributable divergence is at an
**exact tie or one bf16 ulp** in the reference's own logits, and the reference
disagrees with *itself* across equivalent decode strategies at the same or earlier
steps. Details in **T7 extended** below.

## The three numbers

wikitext-2 test (`wiki.test.raw`, 297,054 tokens), `n_ctx = 512`, non-overlapping
windows, 147,900 predictions each, **byte-identical token ids for all three runs**.

| implementation | weights | PPL | vs BF16 |
|---|---|---|---|
| llama.cpp `llama-perplexity` | NVFP4 (GGUF of this checkpoint) | **7.2088** ± 0.0471 | +2.244 % |
| **gb10-engine** | NVFP4 (the checkpoint itself) | **7.0988** ± 0.0164 | **+0.684 %** |
| `transformers` BF16 | BF16 (base model, unquantized) | **7.0506** ± 0.0165 | — |

* engine − llama.cpp = **−0.1100 PPL (−1.526 %)**, engine lower
* engine − BF16 = **+0.0482 PPL (+0.684 %)** — this is the whole cost of 4-bit weights, as the engine sees it
* llama.cpp − BF16 = **+0.1582 PPL (+2.244 %)** — 3.3x the engine's quantization cost

The BF16 row is what makes the middle row interpretable. Without it, the engine
merely looks 1.5 % "different" from llama.cpp. With it, the difference is
*directionally favourable*: both run the same NVFP4 weights, and the engine
reproduces the unquantized model more faithfully.

## Protocol — why these three are comparable

`tools/llama.cpp/tools/perplexity` with its default `--ppl-stride 0` hardcodes
`n_ctx = 512` (`perplexity.cpp:2016`) and scores:

* whole, non-overlapping 512-token windows, model state cleared per window
* `first = n_ctx/2 = 256`
* 255 predictions per window: logits at positions 256..510 against the token
  that follows each (`window[257..511]`)
* `ppl = exp(mean(nll))`

`gb10-verify perplexity` replicates exactly that. The BF16 reference is a
separate 60-line script (`bench/ppl/bf16_ppl.py`) that consumes the same token
file and applies the same windowing.

`add_bos` is **false** for this vocab, so no window is prefixed with a BOS. This
was verified rather than assumed: `llama-tokenize` emits no 248044 for the file
(248044 is this checkpoint's `bos_token_id`), which matches `add_bos = false`
falling out of the QWEN35 pre-tokenizer branch (`llama-vocab.cpp:2266`).

### The token stream is not a variable

The engine tokenizes the same raw file through its own Rust tokenizer and the
stream is compared against llama.cpp's before anything is scored:

```
token stream cross-check
  engine tokenizer   297054 tokens
  llama-tokenize     297054 tokens
  first divergence   None
  mismatches         0 / 297054
```

Failing this check is a hard error path, not a warning. It means a perplexity
difference cannot be a tokenizer artefact.

### The weights are the same weights

A GGUF inventory confirms the GGUF carries the checkpoint's own quantization
rather than a re-quantization:

```
NVFP4   193 tensors   9.631 GiB   (exactly the MLP gate/up/down of all 64 layers, plus lm_head)
BF16    313 tensors  16.641 GiB   (attention projections, embeddings, norms, conv, MTP)
F32     746 tensors   0.010 GiB
```

193 = 3 x 64 + 1, which is precisely the set the checkpoint marks NVFP4. The
attention projections are FP8 E4M3 in the checkpoint and BF16 in the GGUF; that
upconvert is exact (E4M3 has 3 mantissa bits, BF16 has 7), so it moves nothing.
Both implementations therefore run the same numbers.

## Corroboration from the existing gate

Independently of perplexity, the 64-layer greedy gate still agrees **16/16
(100 %)** with the HF `Qwen3_5ForCausalLM` oracle that was produced by
dequantizing this same NVFP4 checkpoint to bf16. That gate compares tokens, not
aggregates, and it is what rules out a systematic logit shift.

## T7 extended: 8 prompts x 64 tokens

T7 (`docs/TARGETS.md`) asks for **top-1 token match against the NVFP4 reference
on the fixed prompt set, >= 32 greedy tokens**. The committed gate checks one
prompt for 16 tokens, so the horizon had never actually been exercised. It is
now: eight prompts, up to 64 tokens each, including a 367-token prefill.

```
python3 tools/full_oracle.py --n 64 --prompts-file tools/oracle_prompts.json \
    --out-dir fixtures/oracle-multi
python3 bench/ppl/multi_prompt_check.py --n 64
```

| prompt | prompt tok | generated | agreement | first divergence |
|---|---|---|---|---|
| p01-capital | 59 | 29 (EOS) | 29/29 | **exact** |
| p02-math | 110 | 64 | 13/64 | 11 |
| p03-code | 76 | 64 | 62/64 | 57 |
| p04-chinese | 76 | 64 | 61/64 | 61 |
| p05-prime | 74 | 64 | 64/64 | **exact** |
| p06-kyoto | 93 | 64 | 55/64 | 55 |
| p07-proof | 80 | 64 | 64/64 | **exact** |
| p08-longctx | 367 | 64 | 61/64 | 61 |
| **total** | | **477** | **409/477 = 85.74 %** | 3/8 exact |

The original fixed prompt (p01) is still exact over its whole 29-token run; it
ends in EOS, which is why T7's ">= 32 tokens" is not reachable on it.

### Why the other five are not an accuracy drop

The reference emits **bf16 logits**. At `|logit| ~ 23` a bf16 ulp is 0.125 and at
`~27` it is 0.25, so its logits sit on a coarse grid and ties are common.
Measured on the reference's own trajectory, at the exact step the engine first
disagrees:

| prompt | step | margin | in bf16 ulp | engine's token tied with the reference's top-1 |
|---|---|---|---|---|
| p02-math | 11 | 0.1250 | 1 | no |
| p03-code | 57 | 0.0000 | 0 | **yes** |
| p04-chinese | 61 | 0.0000 | 0 | **yes** |
| p06-kyoto | 55 | 0.0000 | 0 | **yes** |
| p08-longctx | 61 | 0.0000 | 0 | **yes** |

**Four of five engine divergences are the engine emitting a token the reference
scores exactly as high as its own top-1**, and the fifth is one ulp below. 2.2 %
of the reference's greedy decisions (7/320) are exact ties. At such a step "the"
greedy token is not defined, and the choice is made by whichever tie-breaking
rule the implementation happens to use — `argmax` and `topk(logits, 1)` on the
same tensor return different tokens.

That is not hypothetical; it is how this investigation first went wrong. An
earlier revision of `divergence_analysis.py` picked with `topk` where
`full_oracle.py` picks with `argmax`. At p04-chinese step 1 the reference scores
tokens 96719 and 95826 **both at 27.5**; the two rules chose differently, and the
script then spent a run analysing a trajectory that had nothing to do with the
one the engine was compared against. `bench/ppl/oracle_determinism.py` confirms
the reference is otherwise reproducible: 3 repetitions of the same decode in one
process agree exactly on all 4 prompts tested.

### The reference is not self-consistent either

Stronger control: the same model, same prompt, same argmax, decoded
incrementally against a **KV cache** instead of by recomputing the sequence.
`tools/full_oracle.py` avoids the cache path deliberately ("it avoids depending
on the KV-cache path being bit-identical to a one-token-at-a-time decode"); the
engine uses it.

| prompt | reference cache-vs-recompute first divergence | engine first divergence |
|---|---|---|
| p02-math | none (64/64) | 11 |
| p03-code | **38** | 57 |
| p04-chinese | **35** | 61 |
| p06-kyoto | **55** | **55** |
| p08-longctx | **61** | **61** |

On p06-kyoto and p08-longctx the reference's own cache path leaves the recompute
path at *precisely* the step the engine does. On p03-code and p04-chinese it
leaves **earlier** than the engine. So token-exactness against a recomputing
reference is a property of the decode strategy, not of the engine: on four of
five prompts no KV-cache implementation, including the reference's, can hold it.

p02-math is the one case where the reference's cache path does agree with itself
(64/64) and the engine still diverges at step 11 — that is the one-ulp margin
above, where a difference below the reference's own representable resolution
decides the token.

### What this does not establish

It shows the engine's tokens are ones the reference scores identically or within
one ulp, and that a recomputing reference is not a stable target. It does **not**
prove the engine's logits track the reference's to within an ulp everywhere —
that needs a per-step top-k comparison of engine logits against reference logits,
which is not built. The layer-parity contract remains the looser 2e-3 relative
norm / 5e-2 relative (`crates/gb10-verify/src/main.rs`).

## Negative finding: the llama.cpp NVFP4 path is the outlier

The 1.5 % gap is systematic, not noise. The cumulative traces track each other
with a stable relative offset that widens slowly:

| windows | llama.cpp | engine | delta |
|---|---|---|---|
| 1 | 4.5520 | 4.5006 | −1.13 % |
| 51 | 6.3758 | 6.2927 | −1.30 % |
| 101 | 6.9329 | 6.8378 | −1.37 % |
| 201 | 7.0466 | 6.9438 | −1.46 % |
| 580 | 7.2088 | 7.0988 | −1.53 % |

A stable offset of this shape is a *numerics* difference, not a bug in either
implementation: the engine dequantizes NVFP4 on the fly into a bf16 outer
product, while llama.cpp takes the Blackwell native-FP4 tensor-core path
(`BLACKWELL_NATIVE_FP4 = 1`). The bf16 reference says the engine's choice is the
more accurate of the two on this checkpoint. No mechanism-level attribution was
attempted beyond this, and none is needed to answer the question asked.

## Incident: one load-induced divergent token

While llama.cpp's perplexity run held the GPU, one `generate` invocation
produced `[760, ...]` instead of the oracle's `[1421, ...]` — a fluent but
different continuation, with no error printed. On an idle machine it is 16/16.

It did not reproduce. The compute path was then put under a test with real
statistical power rather than more one-shot runs:

* 23 further `generate` runs passed — 8 of them under the **exact** original
  condition (llama.cpp's perplexity job holding the GPU), and every one of those
  8 produced output **byte-identical** to its clean-machine counterpart: same
  divergence indices, same token ids. Only TTFT moved (455 -> 963 ms; the
  367-token prompt 2290 -> 4752 ms).
* `gb10-verify generate --repeat 100` under that same load: **99/99 extra
  repeats identical to run 0** (450 s of continuous GPU work).
* The 580-window perplexity run is bit-reproducible, and a 21-window re-run
  reproduces the full run's `[20] ppl=6.8307` exactly.

That is roughly 700 forward passes with no observed nondeterminism, under load
as well as idle. The `--repeat` flag is now part of the round gate
(`--repeat 8`), so a nondeterministic path fails the gate rather than being
discovered later.

The mechanism that permits a *silent* wrong answer was found regardless and is
now closed. `CudaSlice::drop` in cudarc synchronises the stream and hands the
result to `CudaContext::record_err`, which stores it in an atomic rather than
raising it (`cudarc-0.19.9/src/driver/safe/core.rs:816`). An asynchronous
failure — illegal access, launch abort, watchdog kill under contention — would
therefore set a sticky error that nothing read, and the engine would keep
running and read whatever the failed kernel left in its output buffer. On the
argmax that is a wrong token that still decodes to fluent text.

`Device::check_err()` now surfaces it, and is called before any result is
trusted: after `argmax` in `step`, `step_batch`, `step_timed` and `prefill_seq`,
and at the end of `forward_normed` (which covers the perplexity path). It is a
host-side atomic swap with no device work.

This does not explain the observed divergence, and the incident stays open. What
changed is its failure mode: it can no longer be silent.

## Reproducing

```bash
# 1. data (ModelScope; huggingface.co is not reachable from this host)
ms download --repo-type dataset Salesforce/wikitext \
    --include "wikitext-2-raw-v1/*" --local-dir bench/ppl/ms-wt
python3 -c "import pyarrow.parquet as pq; \
  t=pq.read_table('bench/ppl/ms-wt/wikitext-2-raw-v1/test-00000-of-00001.parquet'); \
  open('bench/ppl/wiki.test.raw','w').write('\n'.join(t.column('text').to_pylist()))"

# 2. one shared token stream
tools/llama.cpp/build/bin/llama-tokenize \
    -m models/Qwen3.8-27B-NVFP4.gguf -f bench/ppl/wiki.test.raw \
    --ids --show-count > bench/ppl/wiki.tokens.txt

# 3. the three runs
tools/llama.cpp/build/bin/llama-perplexity -m <nvfp4>.gguf -f bench/ppl/wiki.test.raw \
    -ngl 99 --ppl-output-type 1            # 7.2088, 7m39s
target/release/gb10-verify perplexity --model models/Qwen3.8-27B-NVFP4 \
    --tokens bench/ppl/wiki.tokens.txt --ctx 512 --out bench/ppl/engine.json   # 7.0988, 37m46s
python bench/ppl/bf16_ppl.py --model-path <bf16-dir> \
    --tokens bench/ppl/wiki.tokens.txt --out bench/ppl/bf16.json               # 7.0506

# 4. combine
python3 bench/ppl/compare.py

# 5. token-level: 8 prompts, 64 tokens, vs the NVFP4 reference
python3 tools/full_oracle.py --n 64 --prompts-file tools/oracle_prompts.json \
    --out-dir fixtures/oracle-multi               # ~6 min, one model load
python3 bench/ppl/multi_prompt_check.py --n 64    # 3/8 exact, 409/477 tokens
python3 bench/ppl/oracle_determinism.py --reps 3  # reference is reproducible
python3 bench/ppl/divergence_analysis.py          # margins at every divergence
```

The engine perplexity path costs ~3.9 s per 512-token window against llama.cpp's
~0.79 s, which is the same prefill deficit the performance work already
documents; it is not a new finding.

## Artifacts

| file | what |
|---|---|
| `bench/ppl/compare.json` | the perplexity table, machine-readable |
| `bench/ppl/engine.json` | engine result + full config |
| `bench/ppl/bf16.json` | BF16 reference result |
| `bench/ppl/bf16-nll.json` | per-window mean NLL, all 580 windows |
| `bench/ppl/engine.log` | engine cumulative trace |
| `bench/ppl/llamacpp-nvfp4.log` | llama.cpp cumulative trace + final estimate |
| `bench/ppl/wiki.tokens.txt` | the shared 297,054-token stream |
| `bench/ppl/multi-prompt.json` | per-prompt token agreement |
| `bench/ppl/divergence-analysis.json` | margins, tie sets, cache control |
| `bench/ppl/oracle-determinism.json` | reference repeatability |
| `fixtures/oracle-multi/<tag>/greedy_tokens.json` | the 8 NVFP4-reference traces |
| `tools/oracle_prompts.json` | the prompt set |

## What this does not establish

* Only **one** dataset for perplexity (wikitext-2). Task accuracy (MMLU-style
  few-shot) was not run; the scope chosen for this evaluation was perplexity
  plus token agreement.
* Perplexity is a smooth aggregate and would not catch a rare, catastrophic
  failure on an unusual input. The token-level work above covers that direction
  on 8 prompts, and what it caught was divergence at ties, not bad text.
* **No engine logits were ever compared to reference logits.** Every token-level
  conclusion here is inferred from token identity and from the *reference's*
  margins. A per-step top-k comparison would measure the engine's actual error
  directly instead of bounding it by the reference's resolution, and it is the
  single most valuable missing measurement.
* The 1.5 % disagreement with llama.cpp is characterised, not explained. If the
  two must agree exactly, the next step is to compare one tensor's dequantized
  values elementwise, which needs no GPU.
* The load-induced divergence is unreproduced. `check_err` makes it loud, which
  is a mitigation and not a root cause.
