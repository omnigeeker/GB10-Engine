# FA2 staging-loop A/B -- same binary, PTX swap, one session

- started 2026-09-30 20:41:05
- passes 2 per arm per context, interleaved, minimum reported
- instrument: `gb10-verify prefill-shape`, GB10_FA2=1 GB10_PREFILL_NSEQ=1 GB10_ATTN_EVENTS=1, chunk 8192 (the server's PREFILL_CHUNK)
- contexts [131072]

| arm | ptx sha256[:12] | file | extra env |
|---|---|---|---|
| bc32 | `bdbff89850cb` | armA_elementwise.ptx | `GB10_FA2_BC=32` |
| bc16 | `e00754e5f1c1` | armBC16.ptx | `GB10_FA2_BC=16` |

## attn kernel (ms, minimum of passes)

| context | bc32 | bc16 |
|---|---|---|
| 131072 | 92386 | 65632 |

## attn kernel speedup vs the first arm

| context | bc16 |
|---|---|
| 131072 | 1.4076 |

## all passes (raw)

| context | arm | attn kernel ms | total s |
|---|---|---|---|
| 131072 | bc32 | 92386 | 227.71 |
| 131072 | bc32 | 93204 | 225.54 |
| 131072 | bc16 | 65642 | 199.33 |
| 131072 | bc16 | 65632 | 200.3 |

