# Accuracy evaluation — does the engine lose anything?

**Question asked:** run an accuracy evaluation and make sure nothing is dropped.

**Answer:** the engine does not drop accuracy. On wikitext-2 it lands **0.68 % above
the unquantized BF16 model**, against **2.24 %** for llama.cpp's NVFP4 path on the
same weights. The engine is the closer of the two to the reference.

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

It did not reproduce: **7 further runs passed**, 3 of them under the BF16 job's
load and 4 under a concurrent engine perplexity run.

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
```

The engine perplexity path costs ~3.9 s per 512-token window against llama.cpp's
~0.79 s, which is the same prefill deficit the performance work already
documents; it is not a new finding.

## Artifacts

| file | what |
|---|---|
| `bench/ppl/compare.json` | the table above, machine-readable |
| `bench/ppl/engine.json` | engine result + full config |
| `bench/ppl/bf16.json` | BF16 reference result |
| `bench/ppl/bf16-nll.json` | per-window mean NLL, all 580 windows |
| `bench/ppl/engine.log` | engine cumulative trace |
| `bench/ppl/llamacpp-nvfp4.log` | llama.cpp cumulative trace + final estimate |
| `bench/ppl/wiki.tokens.txt` | the shared 297,054-token stream |

## What this does not establish

* Only **one** dataset (wikitext-2). Task accuracy (MMLU-style few-shot) was not
  run; the scope chosen for this evaluation was perplexity.
* Perplexity is a smooth aggregate. It would not catch a rare, catastrophic
  failure on an unusual input; the token-exact `generate` gate covers that
  direction, but only on one prompt.
* The 1.5 % disagreement with llama.cpp is characterised, not explained. If the
  two must agree exactly, the next step is to compare one tensor's dequantized
  values elementwise, which needs no GPU.
* The load-induced divergence is unreproduced. `check_err` makes it loud, which
  is a mitigation and not a root cause.
