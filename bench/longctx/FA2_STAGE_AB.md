# FA2 staging-loop A/B -- same binary, PTX swap, one session

- started 2026-09-30 19:45:22
- passes 2 per arm per context, interleaved, minimum reported
- instrument: `gb10-verify prefill-shape`, GB10_FA2=1 GB10_PREFILL_NSEQ=1 GB10_ATTN_EVENTS=1, chunk 8192 (the server's PREFILL_CHUNK)
- contexts [8192, 32768]

| arm | ptx sha256[:12] | file |
|---|---|---|
| A_control | `bdbff89850cb` | armA_elementwise.ptx |
| B_sr | `069fa019aad5` | armB_build.ptx |
| C_sr_unroll2 | `542b039bec81` | armC_u2.ptx |

## attn kernel (ms, minimum of passes)

| context | A_control | B_sr | C_sr_unroll2 |
|---|---|---|---|
| 8192 | 344 | 349 | 344 |
| 32768 | 5500 | 5801 | 5638 |

## attn kernel speedup vs the first arm

| context | B_sr | C_sr_unroll2 |
|---|---|---|
| 8192 | 0.9857 | 1.0000 |
| 32768 | 0.9481 | 0.9755 |

## all passes (raw)

| context | arm | attn kernel ms | total s |
|---|---|---|---|
| 8192 | A_control | 344 | 8.73 |
| 8192 | A_control | 344 | 8.83 |
| 8192 | B_sr | 357 | 8.75 |
| 8192 | B_sr | 349 | 8.6 |
| 8192 | C_sr_unroll2 | 344 | 8.77 |
| 8192 | C_sr_unroll2 | 351 | 8.71 |
| 32768 | A_control | 5500 | 37.94 |
| 32768 | A_control | 5586 | 39.14 |
| 32768 | B_sr | 5857 | 38.89 |
| 32768 | B_sr | 5801 | 40.02 |
| 32768 | C_sr_unroll2 | 5706 | 38.79 |
| 32768 | C_sr_unroll2 | 5638 | 38.15 |

