# Same-session cold-TTFT A/B — THE FOUR-CONTEXT THREE-ENGINE PROOF

**VERDICT: 3 of 4 contexts won (8K, 32K, 128K). 256K is lost by 2.8%.**

One session, page-cache warm, contention guard before *and* after every trial, two trials per
arm, minimum reported, all three engines driven in one run. This is the reproducible same-session
comparison the objective requires.

**This file supersedes two earlier artifacts, and neither may be quoted:**

* **`TTFT_PROOF.md`** — its `gb10-fa2off` column was invalidated by the cold-page-cache error (the
  first engine started paid first-touch cost on the 21.9 GB model) and **remains withdrawn**.
* **the run killed at 01:00 on 2026-10-01** — its `gb10-fa2on` arm ran a **stale `gb10-server`**,
  built before the `GB10_FA2_BC` default changed 32 -> 16. That binary requested 32 KB of dynamic
  shared memory for a 16 KB tile, silently dropping occupancy from **4 CTAs/SM to 3** and losing the
  whole BC=16 win, with every correctness gate still green. It read 252.78 s at 128K against the
  200.11 s the kernel actually delivers — *worse than plain BC=32* (231.71 s). **Zero `fa2on`
  numbers from that run are usable at any context**, including 8K and 32K.

| ctx | gb10 fa2off | **gb10 fa2on (BC=16)** | llama.cpp | llama/gb10 | verdict |
|---|---|---|---|---|---|
| **8,192** | 9.80 | **8.96** | 10.60 | **1.183** | **WON** |
| **32,768** | 52.31 | **37.18** | 43.31 | **1.165** | **WON** |
| **131,072** | 446.71 | **200.11** | 228.02 | **1.139** | **WON** |
| 262,144 | 1521.94 | 618.22 | **601.45** | 0.973 | **LOST 2.8%** |

**The 256K number carries real uncertainty and must not be quoted as a single figure.** Its two
`gb10-fa2on` trials were **636.91 / 618.22 s — a 3.0% spread**, where llama's two trials were
603.05 / 601.45 s (0.27%) and every other arm agreed to 0.1–0.3%. So the true 256K deficit is
**~2.8%, plausibly a little smaller or larger**. This is the same instrument-resolution problem
noted in `FA2_BC16_256K_BOUND.md`: a single 256K pass is not evidence, and the interleaved-minimum
design is load-bearing at that length.

**Session soundness:** llama.cpp reproduces its previous-session figures (228.02 vs 226.76 s at
128K, 601.45 vs 600.03 s at 256K), and `gb10-fa2off` reproduces its warm values at all four
contexts (9.80 / 52.31 / 446.71 / 1521.94), which independently confirms the page-cache fix is
working.

**Provenance, asserted by the harness rather than assumed** (see "binaries under test" and the
occupancy preflight below): `gb10-server` built 23:44:59, newer than the newest kernel source at
21:16:13; PTX sha256[:12] `e00754e5f1c1`; and the driver's *real* occupancy readout
`binding REGS`, `by_regs 4`, `dynamic_smem 16384` — which is the check that cannot be faked, since
mtimes can lie while the driver's occupancy cannot.

**Gate standard.** BC=16 is not bit-identical to BC=32 and that is accepted, not hidden: it costs
+6.5e-5 mean NLL at ctx 512 and +9.2e-5 at ctx 4096, because the online softmax runs over key tiles
and the fp16 `P*V` accumulator is rescaled twice as often at half the tile width. **The sign was
predicted before measurement** — that is what makes it a characterised cost rather than an
unexplained change, and it is why a 1.14x win shipped instead of being discarded. An fp32 `P*V`
accumulator would buy the 5e-5 back and cost the register budget holding the 4th CTA, i.e. the
entire win.

**Where 256K stands:** from **2.51x behind** (the contaminated run's premise) to **2.8% behind**,
with the residual quantified as an **L2-service dependency on 274.9 TB of K/V requests over a
17.18 GB footprint** — not a DRAM-byte wall. See `FA2_BC16_256K_BOUND.md`.

---

- started 2026-09-30 23:49:59
- contexts [8192, 32768, 131072, 262144]
- engines ['gb10-fa2off', 'gb10-fa2on', 'llama']
- trials 2 (minimum reported), max_tokens 32
- host wayneGB10

## binaries under test

- `gb10-server` built 23:44:59, newest kernel source 21:16:13
- `gb10-verify` built 21:21:24, newest kernel source 21:16:13
- elementwise.ptx sha256[:12] `e00754e5f1c1`

- occupancy preflight (8K): `[occ] attn_prefill_fa2: regs 168  static_smem 0 B  dynamic_smem 16384 B  maxThreads 96  -> by_regs 4  by_smem 6  by_threads 16  binding REGS`


## Cold TTFT (s, minimum over every repeat and trial)

| context | gb10-fa2off | gb10-fa2on | llama |
|---|---|---|---|
| 8192 | 9.80 | 8.96 | 10.60 |
| 32768 | 52.31 | 37.18 | 43.31 |
| 131072 | 446.71 | 200.11 | 228.02 |
| 262144 | 1521.94 | 618.22 | 601.45 |

## Ratio vs the first arm (> 1.00 means the first arm is faster)

| context | gb10-fa2off | gb10-fa2on |
|---|---|---|
| 8192 | 1.0813 | 1.1828 |
| 32768 | 0.8279 | 1.1650 |
| 131072 | 0.5104 | 1.1395 |
| 262144 | 0.3952 | 0.9729 |

## Every measurement (raw, in run order)

| # | context | arm | min cold TTFT s | prompt tokens | reps |
|---|---|---|---|---|---|
| 1 | 8192 | gb10-fa2off | 9.80 | 8193 | 262 |
| 2 | 8192 | gb10-fa2on | 8.96 | 8193 | 262 |
| 3 | 8192 | llama | 10.60 | 8196 | 261 |
| 4 | 32768 | gb10-fa2off | 52.31 | 32746 | 1054 |
| 5 | 32768 | gb10-fa2on | 37.18 | 32746 | 1054 |
| 6 | 32768 | llama | 43.31 | 32749 | 1053 |
| 7 | 131072 | gb10-fa2off | 446.71 | 131017 | 4224 |
| 8 | 131072 | gb10-fa2on | 200.11 | 131017 | 4224 |
| 9 | 131072 | llama | 228.02 | 130989 | 4222 |
| 10 | 262144 | gb10-fa2off | 1521.94 | 261992 | 8449 |
| 11 | 262144 | gb10-fa2on | 618.22 | 261992 | 8449 |
| 12 | 262144 | llama | 601.45 | 261995 | 8448 |

## Prompt tokens actually seen (proof both engines got the same prompt)

| context | gb10-fa2off | gb10-fa2on | llama |
|---|---|---|---|
| 8192 | 8193 | 8193 | 8196 |
| 32768 | 32746 | 32746 | 32749 |
| 131072 | 131017 | 131017 | 130989 |
| 262144 | 261992 | 261992 | 261995 |

- finished 2026-10-01 02:02:15
