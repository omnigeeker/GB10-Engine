# FA2 staging-loop A/B -- same binary, PTX swap, one session

- started 2026-09-30 20:32:19
- passes 2 per arm per context, interleaved, minimum reported
- instrument: `gb10-verify prefill-shape`, GB10_FA2=1 GB10_PREFILL_NSEQ=1 GB10_ATTN_EVENTS=1, chunk 8192 (the server's PREFILL_CHUNK)
- contexts [8192, 32768]

| arm | ptx sha256[:12] | file | extra env |
|---|---|---|---|
| bc32 | `bdbff89850cb` | armA_elementwise.ptx | `GB10_FA2_BC=32` |
| bc16 | `e00754e5f1c1` | armBC16.ptx | `GB10_FA2_BC=16` |

## attn kernel (ms, minimum of passes)

| context | bc32 | bc16 |
|---|---|---|
| 8192 | 335 | 234 |
| 32768 | 5551 | 3782 |

## attn kernel speedup vs the first arm

| context | bc16 |
|---|---|
| 8192 | 1.4316 |
| 32768 | 1.4677 |

## all passes (raw)

| context | arm | attn kernel ms | total s |
|---|---|---|---|
| 8192 | bc32 | 344 | 8.66 |
| 8192 | bc32 | 335 | 8.58 |
| 8192 | bc16 | 234 | 8.47 |
| 8192 | bc16 | 239 | 8.6 |
| 32768 | bc32 | 5551 | 38.03 |
| 32768 | bc32 | 5675 | 39.25 |
| 32768 | bc16 | 3801 | 37.12 |
| 32768 | bc16 | 3782 | 37.42 |

