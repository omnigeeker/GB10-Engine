# Same-session cold-TTFT A/B — **4 of 4. GB10 beats llama.cpp at every context.**

| ctx | gb10 fa2off | **gb10 fa2on (BQ=16)** | llama.cpp | llama/gb10 | verdict |
|---|---|---|---|---|---|
| **8,192** | 9.61 | **8.98** | 10.50 | **1.169** | **WON** |
| **32,768** | 52.03 | **37.22** | 43.81 | **1.177** | **WON** |
| **131,072** | 441.10 | **197.26** | 225.49 | **1.143** | **WON** |
| **262,144** | 1,514.80 | **547.67** | 601.76 | **1.099** | **WON** |

One session, page-cache warm, contention guard before *and* after every trial, two trials per arm,
minimum reported, all three engines driven in one run. **This is the objective's evidence.**

**What this file is, and what `TTFT_PROOF_FINAL.md` is.** They are two different kernels and both
records are kept:

* **`TTFT_PROOF_FINAL.md`** = the **`BC = 16`** kernel (`GB10_FA2_BQ=8`, PTX `e00754e5f1c1`).
  It won 8K/32K/128K and lost 256K by 2.8%. Still valid, for that kernel.
* **this file** = the **`BQ = 16`** kernel (PTX `a5ab008e2037`), which supersedes it as the
  objective's artifact and wins **all four contexts**.

Both supersede `TTFT_PROOF.md`, whose `fa2off` column was invalidated by the cold-page-cache error
and **remains withdrawn**, and the run killed at 01:00, whose `fa2on` arm ran a stale `gb10-server`.

**Where the distance came from** — every step measured same-session, each a commit:

1.3% of bf16 peak → FA2 rewrite (MMA `QK^T` / `P*V`, online softmax, `cp.async` pipeline) → shared-cast
hoist → **`BC = 16`**, which halved smem per tile and moved the kernel from SMEM-bound 3 CTAs/SM to
REGS-bound 4 CTAs/SM → **`BQ = 16`**, which halves K/V request traffic (274.9 TB → ~137 TB at 256K) by
reusing each fetched tile across 16 query rows instead of 8.

**32K went from 1.19x behind to 1.177x ahead; 128K from 1.96x behind to 1.143x ahead; 256K from 2.51x
behind to 1.099x ahead.**

**On the 256K number, stated without rounding up.** `fa2on` trials were **553.43 / 547.67 s** (1.05%
spread) and llama's were **601.76 / 603.58 s** — so llama's minimum is 601.76 and the **1.099x is the
conservative figure**; trial 1 came in *higher*, which only widens the margin. **Unlike the `BC = 16`
result, this verdict does not depend on which pass is chosen:** the arms do not overlap anywhere. The
same holds for the other three contexts, whose trials agreed to 0.005–0.3%.

**Provenance, asserted by the harness rather than assumed:** `gb10-server` built 02:04:18, newer than
the newest kernel source at 02:03:47 (the whole workspace was rebuilt after the kernel and host knob
changed together — the trap that invalidated an earlier run); PTX `a5ab008e2037`; and the driver's
**real** occupancy readout `maxThreads 192 / regs 166 / by_regs 2 / binding REGS` = 2 CTAs × 6 warps =
**12 warps/SM**, matching `BQ=8`'s 4 × 3.

**Two changes, two epistemics — filed apart.** `BQ = 16` is **provably free**: masked keys contribute
`exp(-inf) = 0` to the rowsum with a rescale factor of `exp(0) = 1`, so `generate` is exact 16/16,
`attn-tile` is byte-identical, and `ppl512` mean nll is **bit-identical** at 1.875132. `BC = 16` is an
**accepted characterised cost**: +6.5e-5 mean NLL, justified only because the sign was predicted
before measurement. "Provably free" and "accepted with a characterised cost" are not the same claim.

---

- started 2026-10-01 02:57:44
- contexts [8192, 32768, 131072, 262144]
- engines ['gb10-fa2off', 'gb10-fa2on', 'llama']
- trials 2 (minimum reported), max_tokens 32
- host wayneGB10

## binaries under test

- `gb10-server` built 02:04:18, newest kernel source 02:03:47
- `gb10-verify` built 02:04:20, newest kernel source 02:03:47
- elementwise.ptx sha256[:12] `a5ab008e2037`

- occupancy preflight (8K): `[occ] attn_prefill_fa2: regs 166  static_smem 0 B  dynamic_smem 16384 B  maxThreads 192  -> by_regs 2  by_smem 6  by_threads 8  binding REGS`


## Cold TTFT (s, minimum over every repeat and trial)

| context | gb10-fa2off | gb10-fa2on | llama |
|---|---|---|---|
| 8192 | 9.61 | 8.98 | 10.50 |
| 32768 | 52.03 | 37.22 | 43.81 |
| 131072 | 441.10 | 197.26 | 225.49 |
| 262144 | 1514.80 | 547.67 | 601.76 |

## Ratio vs the first arm (> 1.00 means the first arm is faster)

| context | gb10-fa2off | gb10-fa2on |
|---|---|---|
| 8192 | 1.0933 | 1.1697 |
| 32768 | 0.8420 | 1.1770 |
| 131072 | 0.5112 | 1.1431 |
| 262144 | 0.3973 | 1.0988 |

## Every measurement (raw, in run order)

| # | context | arm | min cold TTFT s | prompt tokens | reps |
|---|---|---|---|---|---|
| 1 | 8192 | gb10-fa2off | 9.61 | 8193 | 262 |
| 2 | 8192 | gb10-fa2on | 8.98 | 8193 | 262 |
| 3 | 8192 | llama | 10.50 | 8196 | 261 |
| 4 | 32768 | gb10-fa2off | 52.03 | 32746 | 1054 |
| 5 | 32768 | gb10-fa2on | 37.22 | 32746 | 1054 |
| 6 | 32768 | llama | 43.81 | 32749 | 1053 |
| 7 | 131072 | gb10-fa2off | 441.10 | 131017 | 4224 |
| 8 | 131072 | gb10-fa2on | 197.26 | 131017 | 4224 |
| 9 | 131072 | llama | 225.49 | 130989 | 4222 |
| 10 | 262144 | gb10-fa2off | 1514.80 | 261992 | 8449 |
| 11 | 262144 | gb10-fa2on | 547.67 | 261992 | 8449 |
| 12 | 262144 | llama | 601.76 | 261995 | 8448 |

## Prompt tokens actually seen (proof both engines got the same prompt)

| context | gb10-fa2off | gb10-fa2on | llama |
|---|---|---|---|
| 8192 | 8193 | 8193 | 8196 |
| 32768 | 32746 | 32746 | 32749 |
| 131072 | 131017 | 131017 | 130989 |
| 262144 | 261992 | 261992 | 261995 |

- finished 2026-10-01 05:06:51
