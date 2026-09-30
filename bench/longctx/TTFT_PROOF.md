# Same-session cold-TTFT A/B

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
