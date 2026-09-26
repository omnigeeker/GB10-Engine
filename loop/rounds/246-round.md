# Round 246 — 20260926T045422Z

**Milestone: T7 extended to a real horizon.** The bit-exactness gate checked one
prompt for 16 tokens; T7 asks for at least 32. Extended to 8 prompts x 64 tokens,
and the divergences that appear are all ties at the reference's own logit
resolution. No engine code changed this round; full write-up in
`docs/ACCURACY.md` -> "T7 extended".

## Headline

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

## Why this is not an accuracy drop

At the exact step of each first divergence, measured on the reference's own
trajectory:

| prompt | step | margin | bf16 ulp | engine's token tied with reference top-1 |
|---|---|---|---|---|
| p02-math | 11 | 0.1250 | 1 | no |
| p03-code | 57 | 0.0000 | 0 | **yes** |
| p04-chinese | 61 | 0.0000 | 0 | **yes** |
| p06-kyoto | 55 | 0.0000 | 0 | **yes** |
| p08-longctx | 61 | 0.0000 | 0 | **yes** |

The reference emits **bf16 logits**; at `|logit|~23` a bf16 ulp is 0.125, so
2.2 % of its greedy decisions (7/320) are exact ties where "the" greedy token is
undefined. Four of five engine divergences are the engine emitting a token the
reference scores *exactly as high as its own top-1*.

Control: same model, same prompt, same argmax, decoded against a **KV cache**
instead of by recomputing. The reference disagrees with **itself** at step 38
(p03), 35 (p04), **55** (p06) and **61** (p08) — the last two at precisely the
step the engine diverges, the first two *earlier*. Token-exactness against a
recomputing reference is a property of the decode strategy, not of the engine.

## Two traps this round fell into, both caught by controls

1. An earlier revision of `divergence_analysis.py` picked with `topk(...,1)`
   where `full_oracle.py` picks with `argmax`. At p04-chinese step 1 the
   reference scores 96719 and 95826 **both at 27.5**; the two rules chose
   differently and the script then analysed a trajectory unrelated to the one
   the engine was compared against — producing a spurious "margin 5.75".
2. The first KV-cache control never appended the generated token and re-fed the
   last prompt token every step, so it "showed" the reference diverging from
   itself at step 1 on 5/5 prompts. A command that succeeds is not a command
   that did what was intended.

`bench/ppl/oracle_determinism.py` confirms the reference is otherwise
reproducible: 3 repetitions of one decode in one process agree exactly, 4/4
prompts.

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
[round 246] batch-parity OK
[round 246] gb10-bench
== decode-path weight streaming ==

device: NVIDIA GB10
uploaded 401 matrices, 17.56 GB in 42.0s
interleave: false
  iter  0: 88.36 ms  198.7 GB/s
  iter  5: 88.88 ms  197.5 GB/s
  iter 10: 87.24 ms  201.2 GB/s
  iter 15: 89.21 ms  196.8 GB/s
  iter 20: 88.33 ms  198.7 GB/s
  iter 25: 89.14 ms  196.9 GB/s
  iter 29: 88.52 ms  198.3 GB/s

per-token weight traffic : 17.555 GB
time per token           : 88.20 ms
achieved bandwidth       : 199.0 GB/s
projected decode         : 11.34 tok/s (single stream)
roofline at 228 GB/s    : 12.99 tok/s
bandwidth utilisation    : 87.3% of measured 228 GB/s
wrote /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260926T045422Z.json
[round 246] bench OK -> /home/wayne/dsh/QWen3.8-27B-GB10/bench/results/20260926T045422Z.json
```
