# FA2 attention: occupancy is the dominant lever, but 4 CTAs/SM is not free

Measured 2026-09-30 on `sm_121a`, same session, `GB10_FA2=1`,
`GB10_PREFILL_NSEQ=1`, `prefill-shape --limit 32768` (attention-kernel ms from
`GB10_ATTN_EVENTS`). **This is a partial result: the probe is complete, the
follow-on restructuring is priced but was NOT implemented or measured.**

## The real occupancy, from the driver (not a proxy)

`crates/gb10-cuda/src/ops.rs` computes `by_smem` from a hardcoded 102400 B, and
this file exists because that constant had been treated as an assumption. It is
correct for this device. Queried directly via `cuDeviceGetAttribute`:

| attribute | value |
|---|---|
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | **102400 B (100 KB)** |
| `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` | 101376 B |
| `MAX_REGISTERS_PER_MULTIPROCESSOR` | 65536 |
| `MAX_THREADS_PER_MULTIPROCESSOR` | 1536 |
| `MAX_BLOCKS_PER_MULTIPROCESSOR` | 24 |

And `attn_prefill_fa2_kernel` as built:

```
[occ] attn_prefill_fa2: regs 168  static_smem 0 B  dynamic_smem 32768 B
      maxThreads 96  -> by_regs 4  by_smem 3  by_threads 16  binding SMEM
```

So the kernel runs **3 CTAs/SM, bound by shared memory, not registers**:
`4 x 32768 = 131072 > 102400`, while registers would allow 4
(`65536 / (168 x 96) = 4.06`). Note 3 CTAs use only 98304 of the 102400 B
available — 4096 B of the SM's shared memory is unreachable at this tile size.

**To reach 4 CTAs/SM, dynamic smem must drop to <= 25600 B (`102400 / 4`), i.e.
21.9% below the current 32768 B.**

## The probe: occupancy is worth ~22% per block

`GB10_FA2_SMEM_PROBE=<bytes>` (added in `ops.rs`) requests more dynamic shared
memory than the kernel uses, lowering blocks/SM **without changing a single
instruction**. That isolates "does this kernel want more resident blocks?" from
every other effect.

| CTAs/SM | smem request | attention, p1 / p2 (ms) |
|---|---|---|
| **3** | 32768 (probe off) | **5576 / 5558** |
| **2** | 49152 | **6800 / 6831** |

**3 -> 2 costs +22.3%.** That is the same figure the older, non-FA2 attention
kernel recorded for 3 -> 2 (+22.4%), so occupancy has stayed the dominant term
across the rewrite. A two-point fit of `T(n) = a + b/n` gives `a = 3074`,
`b = 7452`, which predicts:

```
T(3) = 5558 ms   (measured)
T(4) = 4937 ms   (-11.2% on the attention kernel)
```

**Caveat on this pair:** a sibling agent began a `prefill-shape --limit 8192`
run at roughly the moment the `2 CTA` p2 pass was finishing, so that one pass
may carry a small contention penalty. p1 and p2 agree to 0.5% and p1 was
uncontended, so the 22.3% stands, but a clean re-run on a quiet GPU would be
worth having.

## Why this was not carried through to a 4-CTA kernel

Getting smem under 25600 B means shrinking the key tile `FA2_BC` from 32, and
both candidate sizes cost instructions:

* **`FA2_BC = 24`** -> `2 x 24 x 512 = 24576 B`, fits. But 24 keys is 3 mma
  n-tiles, and `P*V` steps the k-dimension in units of 16
  (`mma.m16n8k16`), so the second k-step would carry only 8 real keys. `QK^T`
  drops to 3 mma per k-step but `P*V` does not shrink: ~112 mma per 24 keys
  versus 128 per 32, i.e. **+16.7% mma per key**.
* **`FA2_BC = 16`** -> `2 x 16 x 512 = 16384 B`, fits with room, and 16 keys is
  exactly 2 n-tiles and one 16-key k-step, so **there is no mma or ldmatrix
  waste at all** (64 mma + 32 ldmatrix per 16 keys = the same per-key rate as
  `BC = 32`). The cost is that every per-tile fixed cost doubles per key: the
  online-softmax `VKQ_C` rescale (32 `__hmul2` per tile), the softmax shuffle
  reductions, the mask, the P packing, and **2 extra `__syncthreads()` per 32
  keys**. That is roughly **+14% issued instructions**.

So the trade is roughly `-11%` from occupancy against `+14%` from doubled
per-tile overhead — **not an obvious win, and it was not measured.** It may
still be worth trying, because the two effects are not additive in an obvious
direction: the occupancy gain applies to the whole kernel while the extra
instructions are concentrated in the softmax/rescale path, and `BC = 16` also
halves the `KQ_C` fragment (4x4 -> 2x4 floats), which frees registers.

**Unresolved and explicitly not claimed:** whether a 4-CTA `BC = 16` kernel is
net faster than the current 3-CTA `BC = 32` kernel. Pricing says it is close.
This needs a build and a same-session A/B that was not run.

## Handover note

The `cp.async` staging loop in `fa2_stage_async` was being reworked concurrently
by a sibling agent (staging is ~35 SASS instructions per 16-byte copy, 31 of
them address/control). That change and this one touch the same file. If both are
pursued, the occupancy route should be re-priced against the *new* staging cost,
because a cheaper staging loop shifts the balance toward the occupancy win.
