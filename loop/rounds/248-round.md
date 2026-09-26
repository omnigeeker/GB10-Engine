# Round 248 — 20260926T083832Z

**Milestone: task accuracy measured, and the engine does not drop points.**
No model behaviour changed. `gb10-verify` gained a `choice` subcommand that
scores multiple-choice questions by answer-letter log-probability, and the engine
was measured against three references on 3,240 MMLU questions.

## MMLU: 3,240 questions, 14 subjects, zero-shot

| implementation | weights | accuracy | vs BF16 | McNemar p |
|---|---|---|---|---|
| `transformers` | BF16 base | **80.74 %** | — | — |
| `transformers` | NVFP4 dequantized — *the engine's own weights* | 80.46 % | −0.28 % | 0.45 |
| **gb10-engine** | NVFP4 | **80.34 %** | **−0.40 %** | **0.25** |
| llama.cpp | NVFP4 GGUF | 79.88 % | −0.86 % | 0.011 |

* Against a reference holding **exactly the same weights**, the engine agrees on
  **3216/3240 = 99.26 %** of questions (net −0.12 %, p = 0.48). The engine's own
  arithmetic costs essentially nothing.
* The 0.40-point gap to BF16 is 13 questions in 3,240 and is **not significant**.
  The dequantized reference pays a similar −0.28 %, so what gap exists is the
  price of 4-bit weights.
* llama.cpp's NVFP4 path is again the outlier, and the only significant gap —
  the same ordering the perplexity result found, on a task metric.
* Prompt is defined once (`bench/mmlu/render.py`, mirrored in the engine) and
  both `transformers` references report **0 prompt-token mismatches / 3,240**.

Two traps caught by assertions rather than assumed away: llama.cpp's server
returns OpenAI-style `logprob`/`top_logprobs` keys (not `prob`/`probs`), and its
candidate list at `n_probs = 100` omitted an answer letter on 130 questions.
Sorting the letter ids would have silently permuted A/B/C/D — the engine's order
is 357/417/351/414, not sorted. Re-running at `n_probs = 1000` reduced omissions
to 3 and changed **0 of 3,240 picks**, so the number was already correct.

Full write-up in `docs/ACCURACY.md` → "Task accuracy: MMLU".

## Gates

| gate | result |
|---|---|
| build | pass |
| test | pass |
| correctness (layers) | pass |
| correctness (64-layer) | pass |
| benchmark | pass |

**status: PASS**

## Log

```
  seq  2 (prompt   13 tok): match
  seq  3 (prompt   17 tok): match
  seq  4 (prompt   21 tok): match
  seq  5 (prompt   25 tok): match
  seq  6 (prompt   29 tok): match
  seq  7 (prompt   33 tok): match
  seq  8 (prompt   37 tok): match
  seq  9 (prompt   41 tok): match
  seq 10 (prompt   45 tok): match
  seq 11 (prompt   49 tok): match
  seq 12 (prompt   53 tok): match
  seq 13 (prompt   57 tok): match
  seq 14 (prompt   61 tok): match
  seq 15 (prompt   65 tok): match
batch parity: 16/16 sequences exact over 16 tokens

batch-parity: OK
[round 248] batch-parity OK
[round 248] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 39.8s
interleave: false
  iter  0: 90.52 ms  193.9 GB/s
  iter  5: 89.80 ms  195.5 GB/s
  iter 10: 91.41 ms  192.1 GB/s
  iter 15: 83.26 ms  210.9 GB/s
  iter 20: 81.68 ms  214.9 GB/s
  iter 25: 85.33 ms  205.7 GB/s
  iter 29: 89.40 ms  196.4 GB/s

per-token weight traffic : 17.555 GB
time per token           : 87.79 ms
achieved bandwidth       : 200.0 GB/s
projected decode         : 11.39 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 87.7% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260926T083832Z.json
[round 248] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260926T083832Z.json
```
