# Same-session cold-TTFT A/B

- started 2026-09-30 18:01:19
- contexts [8192, 32768, 131072]
- engines ['gb10-fa2off', 'gb10-fa2on', 'llama']
- trials 2 (minimum reported), max_tokens 32
- host wayneGB10


## Cold TTFT (s)

| context | gb10-fa2off | gb10-fa2on | llama |
|---|---|---|---|
| 8192 | 9.67 | 9.25 | 10.61 |
| 32768 | 52.14 | 39.78 | 43.72 |
| 131072 | 441.59 | 231.71 | 226.76 |

## Ratio vs llama.cpp (< 1.00 means gb10 is faster)

| context | gb10-fa2off | gb10-fa2on |
|---|---|---|
| 8192 | 0.911 | 0.872 |
| 32768 | 1.193 | 0.910 |
| 131072 | 1.947 | 1.022 |

## Prompt tokens actually seen (proof both engines got the same prompt)

| context | gb10-fa2off | gb10-fa2on | llama |
|---|---|---|---|
| 8192 | 8193 | 8193 | 8196 |
| 32768 | 32746 | 32746 | 32749 |
| 131072 | 131017 | 131017 | 130989 |

- finished 2026-09-30 18:40:33
