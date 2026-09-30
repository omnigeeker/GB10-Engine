# FA2 staging-loop A/B -- same binary, PTX swap, one session

## VERDICT: BQ 8 -> 16 wins 256K, and it wins on EVERY pairing

| 256K | attn kernel ms | total s |
|---|---|---|
| **bq8** (shipped, `e00754e5f1c1`) | 348,447 | 617.39 |
| **bq16** (`a5ab008e2037`) | **268,401** | **540.54** |
| **ratio** | **1.298x** | **1.142x** |

Against llama.cpp's same-session 256K of **601.45 s** (from `TTFT_PROOF_FINAL.md`), BQ=16's
**540.54 s wins by 1.113x**.

**This is robust, not a noise artefact.** The decisive fact is that the arms do not overlap:

* bq8 passes: **649.61 / 617.39 s** (5.2% spread) — min **617.39**
* bq16 passes: **540.54 / 562.20 s** (4.0% spread) — min **540.54**

**bq16's WORST pass (562.20 s) still beats bq8's BEST pass (617.39 s) by 1.098x, and still beats
llama's 601.45 s by 1.070x.** There is no pairing of measured passes in which bq8 — or llama — wins.
That is what makes this a flip rather than a marginal read, and it is the opposite situation from the
BC=16-only 256K result, where the deficit was inside the noise band.

**Why it works:** BQ=16 halves the number of query-row blocks, so each K/V tile fetched is reused
across 16 query rows instead of 8 — halving the K/V *request* traffic (274.9 TB -> ~137 TB at 256K)
at **unchanged total mma work** and **unchanged warp occupancy** (12 warps/SM either way; regs 166
vs 168 against a budget of 170.7). The attention phase speeds up 1.298x while the total speeds up
1.142x, the difference being the ~266-280 s of non-attention work the kernel cannot touch.

**It is also numerically free** — unlike BC=16. `generate` is exact 16/16, `attn-tile` output is
byte-identical to BQ=8, and `ppl512` mean nll is **bit-identical** at 1.875132. The extra key tiles
BQ=16 scans are fully masked, and a masked key contributes `exp(-inf) = 0` to the rowsum with a
rescale factor of `exp(0) = 1`, so it is an exact no-op. This is a *provably free* change, which is
not the same claim as BC=16's *accepted characterised cost* (+6.5e-5 mean NLL).

The four-context same-session cold-TTFT proof is being re-run on this kernel; cold TTFT is the
instrument the objective specifies, and `prefill-shape` totals are not interchangeable with it in
general (at 32K they differ by ~37%; at 256K they happen to agree within ~1.3%, which is why this
result is expected to carry over, but "expected to" is not "measured").

---

- started 2026-10-01 02:09:55
- passes 2 per arm per context, interleaved, minimum reported
- instrument: `gb10-verify prefill-shape`, GB10_FA2=1 GB10_PREFILL_NSEQ=1 GB10_ATTN_EVENTS=1, chunk 8192 (the server's PREFILL_CHUNK)
- contexts [262144]

| arm | ptx sha256[:12] | file | extra env |
|---|---|---|---|
| bq8 | `e00754e5f1c1` | armBC16.ptx | `GB10_FA2_BQ=8` |
| bq16 | `a5ab008e2037` | armBQ16.ptx | `GB10_FA2_BQ=16` |

## attn kernel (ms, minimum of passes)

| context | bq8 | bq16 |
|---|---|---|
| 262144 | 348447 | 268401 |

## attn kernel speedup vs the first arm

| context | bq16 |
|---|---|
| 262144 | 1.2982 |

## all passes (raw)

| context | arm | attn kernel ms | total s |
|---|---|---|---|
| 262144 | bq8 | 382347 | 649.61 |
| 262144 | bq8 | 348447 | 617.39 |
| 262144 | bq16 | 268401 | 540.54 |
| 262144 | bq16 | 283301 | 562.2 |

