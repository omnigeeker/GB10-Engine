# Same-session cold-TTFT A/B

> **WARNING — the `gb10-fa2off` column below is a cold-cache artifact and must
> not be quoted.** `gb10-fa2off` was the first engine started in this run, so it
> paid first-touch cost on the 21.9 GB model file. Re-measured warm it is 15-23%
> faster — **9.67 / 52.14 / 441.59 s, not 12.57 / 65.96 / 522.49 s** — while
> `gb10-fa2on` and `llama` reproduce across sessions to ~1%. Therefore the
> ratios derived from the `gb10-fa2off` column (1.202 / 1.514 / 2.302 / 2.512)
> and the pipeline-speedup figures (1.33x / 1.63x / 2.23x / 2.14x) are **void**.
> See `bench/longctx/TTFT_PROOF_128K.md` for the warm column, and the
> 2026-09-30 section at the end of `comparison.md` for the full withdrawal.
> **The `gb10-fa2on` vs `llama` column below is unaffected and remains valid** —
> both arms ran warm and both reproduce.

- started 2026-09-30 14:34:17
- contexts [8192, 32768, 131072, 262144]
- engines ['gb10-fa2off', 'gb10-fa2on', 'llama']
- trials 2 (minimum reported), max_tokens 32
- host wayneGB10


## Cold TTFT (s)

| context | gb10-fa2off | gb10-fa2on | llama |
|---|---|---|---|
| 8192 | 12.57 | 9.29 | 10.46 |
| 32768 | 65.96 | 40.41 | 43.57 |
| 131072 | 522.49 | 233.46 | 226.93 |
| 262144 | 1507.08 | 709.52 | 600.03 |

## Ratio vs llama.cpp (< 1.00 means gb10 is faster)

| context | gb10-fa2off | gb10-fa2on |
|---|---|---|
| 8192 | 1.202 | 0.888 |
| 32768 | 1.514 | 0.928 |
| 131072 | 2.302 | 1.029 |
| 262144 | 2.512 | 1.182 |

## Prompt tokens actually seen (proof both engines got the same prompt)

| context | gb10-fa2off | gb10-fa2on | llama |
|---|---|---|---|
| 8192 | 8193 | 8193 | 8196 |
| 32768 | 32746 | 32746 | 32749 |
| 131072 | 131017 | 131017 | 130989 |
| 262144 | 261992 | 261992 | 261995 |

- finished 2026-09-30 16:52:52
