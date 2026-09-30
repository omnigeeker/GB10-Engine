# FA2 staging-loop A/B -- same binary, PTX swap, one session

- started 2026-09-30 21:32:58
- passes 2 per arm per context, interleaved, minimum reported
- instrument: `gb10-verify prefill-shape`, GB10_FA2=1 GB10_PREFILL_NSEQ=1 GB10_ATTN_EVENTS=1, chunk 8192 (the server's PREFILL_CHUNK)
- contexts [262144]

| arm | ptx sha256[:12] | file | extra env |
|---|---|---|---|
| bc32 | `bdbff89850cb` | armA_elementwise.ptx | `GB10_FA2_BC=32` |
| bc16 | `e00754e5f1c1` | armBC16.ptx | `GB10_FA2_BC=16` |

## attn kernel (ms, minimum of passes)

| context | bc32 | bc16 |
|---|---|---|
| 262144 | 438623 | 330455 |

## attn kernel speedup vs the first arm

| context | bc16 |
|---|---|
| 262144 | 1.3273 |

## all passes (raw)

| context | arm | attn kernel ms | total s |
|---|---|---|---|
| 262144 | bc32 | 438623 | 704.85 |
| 262144 | bc32 | 459865 | 726.59 |
| 262144 | bc16 | 371698 | 642.42 |
| 262144 | bc16 | 330455 | 610.18 |

