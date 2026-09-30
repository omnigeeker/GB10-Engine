# 256K on the BC=16 kernel: measured ratio, and what the 256K attention is actually bound by

**Status: 256K does NOT flip.** 3 of 4 contexts won (8K, 32K, 128K). This document is the
measured ratio plus the arithmetic that bounds the residual. The measurement is in
`bench/longctx/FA2_BC16_256K_AB.md`; this is the analysis of it.

## 1. The measurement (same session, same binary, PTX swap, interleaved, min of 2)

| ctx | bc32 (3 CTAs/SM) | **bc16 (4 CTAs/SM)** | speedup |
|---|---|---|---|
| 262144 | 438,623 ms | **330,455 ms** | **1.3273** |

Raw passes:

| ctx | arm | attn kernel ms | total s | implied non-attention |
|---|---|---|---|---|
| 262144 | bc32 | 438,623 | 704.85 | 266.23 s |
| 262144 | bc32 | 459,865 | 726.59 | 266.73 s |
| 262144 | bc16 | 371,698 | 642.42 | 270.72 s |
| 262144 | bc16 | 330,455 | 610.18 | 279.73 s |

* **bc32 reproduces the recorded 443,018 ms to within 1.0%** (438,623), so instrument and
  recorded baseline agree and the comparison is on the same scale.
* **Every bc16 pass beats every bc32 pass.** The worst bc16 (371,698) is 15.3% below the best
  bc32 (438,623). The direction is unambiguous even at worst-case pairing.
* **Pass-to-pass spread at 256K is large**: 4.8% for bc32, 12.5% for bc16. This is the honest
  resolution of the instrument at this length, and it is larger than the ~1.5% seen at 32K.
  A same-session internal check confirms it: **non-attention work is identical code in both
  arms, and it reads 266.23 / 266.73 / 270.72 / 279.73 s — a 5.1% spread.** That is the
  session's own noise floor, measured on a quantity the kernel change cannot touch.
* So the reportable ratio is a **range 1.180 (worst pairing) to 1.327 (min of 2)**. The
  protocol figure is **1.3273**.

Attention rate at 256K (`13,510.9 TFLOP` — the recorded convention, see §4):

| ctx | arm | attn s | TFLOP/s | % of the 115 TFLOP/s `mma.sync` ceiling |
|---|---|---|---|---|
| 262144 | bc32 | 438.623 | 30.80 | 26.8% |
| 262144 | **bc16** | **330.455** | **40.89** | **35.6%** |

## 2. Verdict: 256K does not flip, and the reason is *not* a hard bandwidth wall

**The decisive negative test came back negative for the bandwidth hypothesis.** The handoff
stated the test in advance: *"If the ratio comes in near 1.0, the honest reading is BC=16 is an
occupancy win and 256K is bandwidth-bound."* The ratio is **1.327, not near 1.0.** Occupancy
still buys 33% at 256K. Therefore:

> **256K is still predominantly occupancy/latency-hiding limited.** A pure KV-bandwidth-bound
> kernel would be insensitive to resident blocks per SM; this one is not.

What *is* true is that the occupancy gain shrinks with context: **1.408 at 128K → 1.327 at
256K** (~6% relative). So a second constraint grows with context, but it does not dominate at
256K, and it is not a wall.

## 3. How close 256K is — and why the cross-session number cannot settle it

Measured this session, bc16: **total prefill 610.18 s.**
Recorded llama.cpp (previous session, trial 0): **600.03 s.**

Nominal deficit **1.7%**, down from **18.2%**. But this comparison is **not valid evidence**:

* the two numbers are from **different sessions**, and this box has drifted up to 23%;
* the session's own noise floor on the total is **5.1%** (§1), which is 3x the nominal 1.7%;
* the arithmetic is boundary-sensitive. `total = attention + non_attention`, and non-attention
  was measured at both 266.23 s and 279.73 s in this very session:
  * with non-attention = 279.73 s → attention must be < 320.30 s → **needs a further 1.032x**;
  * with non-attention = 266.23 s → attention must be < 333.80 s → **330.455 s already wins**.

**So 256K is a coin-flip against the recorded number, decided by which side of a 5% noise floor
it lands on. Only `ab_all.py` (three engines, one session) can decide it.** That is task 2, and
it is not a formality here — it is the measurement that resolves 256K.

## 4. FLOP convention (recorded, verified here so the rates are not misread)

`comparison.md:4019`: `causal prefill attention FLOPs = 2 * T^2 * n_heads * head_dim * n_layers`
with `n_heads = 24`, `head_dim = 256`, `n_layers = 16` full-attention layers. At T = 262144 that
is **13,510.9 TFLOP**, matching the recorded table exactly. I re-derived it independently from
the kernel's own `mma` issue count (per key-tile: QK `2*48*BC*256` + PV `2*48*256*BC` = 786,432
FLOP; summed over the `s0 <= win_max` loop and the block grid, × 16 layers) and got 211.2 TFLOP
at 32K against the recorded 211.1 — so the recorded rates are **real, causal, 16-layer** FLOPs,
not inflated. Attention at 256K/bc16 is **40.89 TFLOP/s = 35.6%** of the 115 TFLOP/s ceiling.

## 5. The byte arithmetic: this is a *request* problem, not a DRAM-bytes problem

Traffic is counted from the kernel, not assumed. The kernel is `kh = blockIdx.x` (**one block per
KV head**) and the block serves all 6 GQA heads (`FA2_GQA = 6`, `FA2_NCOLS = 48`), so K/V is
fetched **once per KV head per query tile** — GQA sharing is already exploited, and there is no
6x waste to recover.

Per key position the block fetches a K row (512 B) + a V row (512 B) = **1 KB**
(`kernels/elementwise.cu:746-760`, 16-byte `cp.async` per 512-byte row per warp).

Keys scanned per KV head per layer, summed over the chunked prefill (32 chunks of 8192, causal
`win_max = start + t0 + rows - 1`, `elementwise.cu:825`):

```
sum over chunks c of [ 1024*start_c + sum_{j<1024} (8j + 8) ]
  = 8,388,608 * (0+1+...+31) + 32 * 4,198,400
  = 4,160,749,568 + 134,348,800 = 4.295e9   key-positions per KV head per layer
```

| quantity | value |
|---|---|
| key-positions per layer (× 4 KV heads) | 1.718e10 |
| K/V bytes requested per layer | **17.18 TB** |
| K/V bytes requested, all 16 layers | **274.9 TB** |
| unique K/V footprint, all 16 layers | **17.18 GB** |
| required reuse factor (requests ÷ unique) | **16,000x** |
| unique footprint at 228 GB/s | **75 ms** |

Request throughput the kernel actually sustains:

| arm | attn s | request throughput | vs 228 GB/s DRAM |
|---|---|---|---|
| bc32 | 438.623 | 627 GB/s | 2.75x |
| **bc16** | **330.455** | **832 GB/s** | **3.65x** |

**Consequences — this is what the 256K verdict rests on:**

1. **Unique DRAM bytes are irrelevant.** The whole 17.18 GB of KV is 75 ms at the measured
   228 GB/s read bandwidth. It cannot explain 330 s.
2. **DRAM can absorb at most 27% of the requests.** If DRAM were 100% busy for the entire
   attention pass it would move `330.455 s * 228 GB/s = 75.3 TB` — only 27% of the 274.9 TB of
   requests. So the kernel **depends on ≥73% of K/V requests being served by L2**, and *that*
   dependency is the thing that grows with context. This is the correct, defensible form of the
   "KV bandwidth" story: it is an **L2 hit-rate / latency** dependency, not a DRAM-byte wall.
3. **L2 *bandwidth* alone is not the bound either.** 274.9 TB over 330 s is 832 GB/s, far below
   any plausible L2 bandwidth. What is expensive is not the bytes but the **latency of the
   fraction that misses**.
4. **This is corroborated by a committed negative result.** `2673c74` measured that
   strength-reducing the staging-loop address arithmetic *shortened the prefetch distance and made
   the kernel slower* (-23% instructions, 0.948x at 32K), with occupancy unchanged. A kernel
   limited by instruction issue would have got faster. A kernel limited by **hiding memory
   latency** gets slower when its prefetch distance shrinks. That is the same signature seen here
   at 256K, where L2 misses are more frequent, and it is why more resident CTAs still help
   (bc16 > bc32) while the *incremental* gain falls.

**Honest limits of this analysis:** there are **no hardware counters** on this box (`ncu` blocked
by `RmProfilingAdminOnly: 1`, no passwordless sudo — re-verified in this project). Every number
above is either derived from the kernel's own addressing or measured wall-clock. The L2 hit-rate
implication in (2) is **arithmetic, not a measurement**; no counter confirms the 73% figure. The
attribution to *latency* rather than *bandwidth* rests on the occupancy response (§2) and on
`2673c74`, and is a **well-supported hypothesis, not a proven mechanism**.

## 6. Priced levers

### Lever A — BQ 8 → 16 query rows per block (halves K/V request traffic at equal warp occupancy)

**This is the lever that targets the actual constraint, and it is the one worth building.**

Because traffic is per *key fetch per query tile*, doubling the query rows each block covers
halves the total request traffic **and** halves the number of blocks, with total `mma` work
unchanged.

The non-obvious part is that it need not cost warp occupancy:

| config | threads/CTA | CTAs/SM by regs (168 regs) | warps/SM | K/V traffic |
|---|---|---|---|---|
| BQ=8, BC=16 (shipped) | 96 | `65536/(96*168)` = 4.06 → **4** | 12 | 1.0x |
| **BQ=16, BC=16** | 192 | `65536/(192*168)` = 2.03 → **2** | **12** | **0.5x** |

**2 CTAs × 192 threads = 4 CTAs × 96 threads = 384 threads = 12 warps per SM.** Per-thread state
is unchanged: each warp still owns 16 qcols (8 rows × 2 heads), so the Q fragment stays 64 regs
and `VKQ_C` stays 64 regs. Smem is unchanged (K + V = 2 × BC × 512 B = 16 KB per CTA; 2 × 16 KB =
32 KB, far under 101,376 B), so `by_smem` is not binding. The block simply becomes 6 warps
covering 16 rows × 6 heads = 96 qcols, with warp `w` taking rows `(w/3)*8` and heads
`2*(w%3), 2*(w%3)+1`.

* **Predicted payoff:** halves K/V requests. If the ~20% efficiency gap from 128K to 256K is
  request/latency-driven, this recovers a large part of it. A further **1.032x on attention flips
  256K** (§3) — this lever is priced well above that.
* **Predicted sign: positive.** The evidence is that occupancy pays (bc16 > bc32) and that the
  binding cost is request service, which this halves.
* **Risk, stated plainly:** the register budget is `65536/(192*2) = 170.7` regs/thread, and the
  kernel already sits at **168** — a **2-register margin**, exactly the margin the BC=16 landing
  ran on. If 6 warps change allocation and it spills, occupancy drops to 1 CTA/SM (6 warps) and
  the whole gain is lost. **Check `GB10_ATTN_OCCUPANCY=1` first; that readout is the go/no-go.**
* **Do not build it without that check**, and measure it with the same PTX-swap A/B
  (`bench/longctx/fa2_stage_ab.py`) so the ratio is same-session.

### Lever B — deepen the `cp.async` pipeline (rejected on price)

More in-flight K tiles means double-buffering K: K 32 KB + V 16 KB = 48 KB per CTA → `by_smem`
`102400/49152 = 2` CTAs/SM, down from 4. The old kernel's own measurement prices 3→2 CTAs/SM at
**+22.4%**, and bc16 > bc32 shows occupancy is worth more than prefetch depth here. **Predicted
negative; do not build.**

### Lever C — key-blocked / persistent-CTA scheduling (the only lever aimed at L2 reuse)

Resident CTAs are near-synchronised in `s0`, so intra-wave reuse across the ~48 concurrent query
tiles should be good; but a wave's CTAs finish at different `win_max` and new CTAs restart at
`s0 = 0`, so reuse degrades as the prefix grows. A persistent-CTA schedule that walks the key
dimension in a coordinated way would tighten it. **Cost: a significant rewrite of the launch and
loop structure, unverifiable without counters.** Only worth it if Lever A fails.

### Explicitly NOT a lever

* **GQA sharing** — already fully exploited (`kh = blockIdx.x`, 6 heads per block). The
  "6 blocks re-read the same K/V" diagnosis in `HANDOFF.md` described the **old** kernel; it does
  not apply to the FA2 kernel.
* **Reducing K or V bytes** — both are fp16 and both are needed; the only reduction available is
  reuse across query rows, which is Lever A.
* **Non-attention** — 266–280 s and already at the measured cuBLAS bf16 floor (<3% recoverable,
  committed). It cannot supply the 3.2% needed at 256K in any case.

## 7. Summary

| question | answer |
|---|---|
| Does 256K flip? | **No.** 3 of 4 contexts won. |
| Measured 256K attention ratio (bc32 → bc16) | **1.3273** (min of 2; worst-pairing 1.180) |
| Is 256K bandwidth-bound? | **No, not in the pure sense** — occupancy still buys 33% (ratio ≫ 1.0). |
| What *is* the constraint? | **Request service / memory-latency hiding**, with ≥73% of the 274.9 TB of K/V requests needing L2 (arithmetic, no counters). Unique DRAM bytes (17.18 GB = 75 ms) are irrelevant. |
| How far from flipping? | **1.032x more on attention** against the recorded llama.cpp total — but that is inside a 5.1% session noise floor, so only `ab_all.py` can decide it. |
| Best lever | **BQ 8 → 16**: halves K/V requests at equal warp occupancy (12 warps/SM either way), gated on a 2-register margin. |
