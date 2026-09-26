# GB10 vs GB300 — ISA targets, tensor-core paths, and kernel portability

**Scope.** This answers a question the rest of `docs/` does not: *GB10 and GB300
are both "Blackwell" — how far can a kernel tuned on one be reused on the
other?* It is the target/portability companion to [`PHYSICS.md`](PHYSICS.md)
(the bandwidth roofline) and [`ARCHITECTURE.md`](ARCHITECTURE.md) (model
semantics).

**Evidence level — read this before quoting anything below.** `PHYSICS.md` is
measurement-first. This document is **not**: nothing here has been run on a
GB300, and the GB300 column is documentation-derived. Every claim is tagged:

| Tag | Meaning |
|---|---|
| `[measured]` | produced by a command run on this Spark |
| `[vendor]` | quoted from NVIDIA documentation (CUDA 13.0, PTX ISA 9.4) |
| `[report]` | third-party bug report — evidence of real-world breakage, not spec |

---

## 1. They are in different compute-capability families

`[vendor]` NVIDIA's compute-capability table:

| CC | Data Center | Workstation / Consumer | Jetson |
|---|---|---|---|
| **12.1** | — | — | **NVIDIA GB10 (DGX Spark)** |
| 12.0 | RTX PRO 6000 Blackwell SE | RTX PRO Blackwell, GeForce RTX 50 | — |
| 11.0 | — | — | Jetson T5000 / T4000 |
| **10.3** | **NVIDIA GB300, NVIDIA B300** | NVIDIA GB300 (DGX Station) | — |
| 10.0 | NVIDIA GB200, NVIDIA B200 | — | — |

Two things to take from this table. First, "both are Blackwell" is a marketing
statement, not an ISA statement: **GB10 is CC 12.1, GB300 is CC 10.3**, and they
do not sit in the same family. Second, note the column GB10 appears in — NVIDIA
classifies it with Jetson (edge/embedded), not with the data-centre parts.

`[measured]` One toolchain covers both. CUDA 13.0 `nvcc` on this machine:

```
$ nvcc --list-gpu-arch
compute_75  compute_80  compute_86  compute_87  compute_88  compute_89
compute_90  compute_100 compute_110 compute_103 compute_120 compute_121
$ nvcc --list-gpu-code
sm_75 … sm_90  sm_100  sm_110  sm_103  sm_120  sm_121
```

Same compiler, same language, same CUDA version. The divergence is in **which
features each target unlocks**, and that is where portability dies.

---

## 2. CUDA 13 targets come in three flavours

`[vendor]` nvcc documentation, verbatim:

> PTX for `.target sm_XY` can be compiled to all GPU targets `sm_MN`, `sm_MNa`,
> `SM_MNf` where `MN >= XY`. PTX for `.target sm_XYf` can be compiled to GPU
> targets `sm_XZ`, `sm_XZf`, `sm_XZa` where `Z >= Y` and `sm_XY` and `sm_XZ`
> belong in same family. PTX with `.target sm_XYa` can only be compiled to GPU
> target `sm_XYa`.

In plain terms:

| flavour | example | reach |
|---|---|---|
| **baseline** | `sm_121` | any target with `MN >= 121`; carries no arch-specific instructions |
| **family** | `sm_121f` | `sm_XZ{,f,a}` for `Z >= Y` **inside the same family** |
| **arch-specific** | `sm_121a` | exactly `sm_121a`, nothing else, ever |

Applied to our two machines:

| artifact | runs on GB10 (12.1) | runs on GB300 (10.3) |
|---|---|---|
| `sm_121` cubin | yes | **no** — `no kernel image is available` |
| `compute_121` PTX | yes | **no** — PTX is forward-compatible only to *higher* CC, and 10.3 < 12.1 |
| `sm_121a` / `sm_121f` PTX | yes | **no** — `a` is pinned to one target; `f` cannot leave the 12.x family |
| `sm_120f` PTX | yes | **no** — same family as 121, not as 103 |
| `sm_103a` / `sm_100f` PTX | **no** | yes |
| baseline `compute_100` / `compute_90` PTX | yes (JIT) | yes (JIT) |

Only the last row gives one artifact that runs on both. §3 explains why it is
worthless for this engine.

---

## 3. NVFP4 takes a different instruction path on each part

This is the crux of the whole question. `[vendor]` PTX ISA 9.4, Target ISA Notes.

**GB300-class (`sm_100`/`sm_103`) uses `tcgen05.mma`:**

> Supported on following architectures: `sm_100a`, `sm_101a` (Renamed to
> `sm_110a` from PTX ISA version 9.0). And is supported on following
> family-specific architectures from PTX ISA version 8.8: `sm_100f` or higher in
> the same family, `sm_101f` or higher in the same family …, `sm_110f` or higher
> in the same family.

That is the **TMEM** programming model: accumulators live in tensor memory, with
`tcgen05.alloc/dealloc`, `tcgen05.ld/st`, `.cta_group` (CTA-pair MMA),
`.warpx2`, `.pack::16b`, `scale-input-d`. In PTX Table 73 ("features promoted to
family-specific architecture") every one of those is listed for `sm_100f` /
`sm_101f` — and **`sm_120a` / `sm_121a` appear in none of those rows.**

**GB10-class (`sm_120`/`sm_121`) uses `mma` / `mma.sp` with block scaling:**

> `.kind`, `.block_scale`, `.scale_vec_size` qualifier requires `sm_120a` and
> are supported on `sm_120f` and later generation targets in the same family from
> PTX ISA version 8.8 **except for `.kind::mxf4nvf4` / `.kind::mxf4`**.
> Qualifiers `.kind::mxf4nvf4` and `.kind::mxf4` are supported on following
> architectures: `sm_120a`, `sm_121a`.

Register-accumulator MMA, no TMEM. And note the sting in the tail: **the NVFP4
kinds themselves are arch-specific, not family-portable.** Block scaling is
promoted to the family (`sm_120f`), but the MXF4/NVFP4 kind is not.

Consequence for this project: **GB10 NVFP4 MMA kernels must target `sm_121a`,
and `sm_121a` PTX can only ever run on `sm_121`** — not on RTX 50 (`sm_120`),
not on GB300 (`sm_103`). A GB10 FP4 kernel is, by construction, a one-machine
kernel.

`[vendor]` A second and much blunter difference, from the same document —
maximum **statically allocated shared memory per CTA**:

| target | static smem / CTA |
|---|---|
| `sm_90a`, `sm_100a`, `sm_103a`, `sm_107a`, `sm_110a` | 228 KB |
| `sm_120a`, `sm_121a` | **100 KB** |

`[measured]` This engine already hit that ceiling on its own, before this
document existed: [`ROADMAP.md`](ROADMAP.md) M1 records "GB10 caps dynamic
shared memory at 99 KB/block and 100 KB/SM", which is exactly why
`kernels/gemv.cu` keeps the activation tile in registers and uses no shared
memory at all. The same kernel on GB300 could spend 2.28× more shared memory
per CTA — a different design point, not a bigger version of the same one.

> **Caveat, stated honestly.** CUTLASS users have filed #2614, #2800 and #2947
> asking for `tcgen05`/FP4 on `sm_120`/`sm_121`; #2947 argues that GB10's
> "1 PFLOP FP4" marketing implies hardware support (§4). Those issues concern
> the CuTe DSL whitelist and are closed without any ISA change. The statement
> above is what **PTX currently exposes**. If a future CUDA/PTX release adds
> `sm_121` to the `tcgen05` target list, revisit this section. Until then, read
> "GB10 has no tcgen05" as "the ISA does not expose it".

---

## 4. The memory systems move the roofline by ~26×

`[vendor]` NVIDIA product specifications:

| | GB10 (DGX Spark) | GB300 (DGX Station) |
|---|---|---|
| memory | 128 GB LPDDR5X, coherent unified | 252 GB HBM3e + 496 GB LPDDR5X |
| bandwidth | **273 GB/s** (228 GB/s probe `[measured]`) | **7.1 TB/s** HBM3e + 396 GB/s CPU side |
| FP4 tensor core | 1 PFLOPS (with sparsity) | 20 / 15 / 3 PFLOPS |
| FP8 / FP6 | — | 10 PFLOPS |
| FP16 / BF16 | — | 5 PFLOPS |
| FP32 | — | 80 TFLOPS |
| interconnect | ConnectX-7, 200 Gb/s | NVLink-C2C 900 GB/s, CX-8 800 Gb/s |
| power | 140 W (whole GB10, CPU+GPU) | kW-class GPU |

~26× the bandwidth, ~20× the FP4 throughput. This is the part that matters more
than the ISA: a kernel shaped to saturate 250 GB/s — batch-1 GEMV, tiny tiles,
minimal shared memory, occupancy above all — is precisely **not** the kernel
that fills 7.1 TB/s of HBM on a part with TMEM, 228 KB of shared memory per CTA
and CTA-pair MMA. One kernel cannot sit near both rooflines. The *method*
(roofline analysis, bandwidth saturation, occupancy tuning) transfers; the
*numbers* must be re-derived.

---

## 5. Portability matrix

| layer | reusable? | notes |
|---|---|---|
| cubin / `a`-suffixed PTX | **no** | pinned to one target, even inside a family |
| `f`-suffixed PTX | partial | inside the family only (`sm_120f` → sm_120, sm_121) |
| baseline PTX (`compute_90` / `compute_100`) | yes | both JIT it, but no FP4 tensor-core path |
| TMA / `cp.async.bulk` pipelines | yes | family-promoted on both sides |
| 128/256-bit vector loads, cvt + dequant arithmetic | yes | where our current GEMV kernels live |
| kernel abstraction + dispatch + graph executor | yes | design for this now, not later |
| paged KV, continuous batching, prefix caching, MTP/draft scheduling | yes | the genuinely portable asset |
| tokenizer, chat template, config, safetensors reader, endpoints | yes | `gb10-core` / `gb10-server` are not arch-specific |
| NVFP4 GEMM/GEMV **kernels themselves** | **no** | different instruction paths; effectively a rewrite |
| tile sizes, occupancy constants, smem budgets | **no** | method transfers, values must be re-searched |

---

## 6. What this means for this repository

### 6.1 `build.rs` targets the baseline on purpose — and that has a deadline

`crates/gb10-cuda/build.rs` compiles with `-arch sm_121` (`CUDA_ARCH =
"sm_121"`), which expands to `compute_121` + `sm_121` — the **baseline** feature
set.

That is correct *today*: the GEMV kernels are plain vector loads plus integer
and dequant arithmetic, they use no tensor-core instruction, and baseline PTX
keeps the comment in that file header honest ("PTX … so the driver JITs for the
exact device it finds").

It stops being correct the moment tensor-core MMAs land for prefill (M2/M7).
`mma` / `mma.sp` with `.kind::mxf4nvf4`, `.block_scale` or `.scale_vec_size`
**cannot be expressed at `compute_121`**. At that point `CUDA_ARCH` must become
`sm_121a` — arch-specific and GB10-only, which is the deliberate scope of this
engine. Worth encoding the intent now:

```rust
// "sm_121"  -> compute_121 + sm_121: baseline features only.
//              Fine for the current GEMV kernels (vector loads + dequant maths).
// "sm_121a" -> required once any tensor-core MMA with NVFP4 block scaling
//              lands; pinned to GB10 by construction (the intended scope).
const CUDA_ARCH: &str = "sm_121";
```

If an RTX-50-capable build is ever wanted, the two can coexist:

```bash
-gencode arch=compute_121a,code=sm_121a \
-gencode arch=compute_120f,code=sm_120f
```

but remember from §3 that the `f` variant covers sm_120 + sm_121 **without** the
NVFP4 kinds.

### 6.2 Gate on families, never on a single CC

Three real-world failures, all from the same mistake:

- `[report]` SGLang excluded DeepGEMM with `cc == 120`, so it stayed enabled on
  GB10's `121` (sglang#36551).
- `[report]` vLLM-omni gated a quack FP8 kernel to data-centre Blackwell
  `sm_100.x` — the pattern to copy when a kernel genuinely is DC-only.
- `[report]` CUTLASS's CuTe DSL whitelisted `tcgen05` to `sm_100a`/`sm_103a` and
  rejected `sm_121` (cutlass#2947).

So write `is_consumer_blackwell(cc) = (cc == 120 || cc == 121)`, never `cc ==
121`. The same lesson shows up one level up the stack: SGLang's DGX Station
guidance requires `--attention-backend flashinfer` because `trtllm_mha` fails
CUDA-graph capture on SM103 — same engine, different kernel that works per
target.

### 6.3 Layout

Keep target-specific kernels in their own directory so a second target is an
*addition*, not a fork:

```
kernels/sm121/    # current: baseline GEMV (vector loads, dequant maths)
kernels/sm103/    # future:  tcgen05 + TMEM path, 228 KB smem budget
```

`build.rs` already enumerates `kernels/*.cu`; extending it to one PTX set per
target directory, with runtime selection by device compute capability, is a
small change made cheap now and expensive later.

### 6.4 If a DGX Station ever enters the picture

Budget it as **kernel-layer rewrite, upper-layer reuse**: new FP4 GEMM/GEMV on
`tcgen05`, new tiling for 7.1 TB/s and a 228 KB smem budget, fully re-tuned
occupancy. `gb10-core` (config, safetensors, tokenizer, chat template) and the
serving layer (continuous batching, paged KV, MTP, endpoints) carry over
unchanged. In other words `gb10-cuda` is really *the Blackwell-family CUDA
backend*; only its name would need to change.

---

## 7. Reproduce the toolchain claims

```bash
# 1. Targets this toolchain can emit (both machines appear in the same list)
nvcc --list-gpu-arch
nvcc --list-gpu-code

# 2. What the crate actually builds — the PTX .target line
find target -name '*.ptx' -print -quit | xargs grep -m1 '\.target'
#    expect: .target sm_121     (baseline, while CUDA_ARCH = "sm_121")

# 3. Baseline vs arch-specific compilation of the current kernels.
#    Both pass today because they use no arch-specific instruction; re-run this
#    the day one does, because only the 'a' target will accept it.
nvcc -arch=sm_121  -ptx kernels/gemv.cu -o /tmp/gemv-base.ptx
nvcc -arch=sm_121a -ptx kernels/gemv.cu -o /tmp/gemv-arch.ptx
```

Source material for the `[vendor]` claims: CUDA 13.0 `nvcc` documentation
(target rules), PTX ISA 9.4 (Target ISA Notes; Table 73, "List of features
promoted to family-specific architecture"), the Blackwell Architecture
Compatibility Guide, NVIDIA's compute-capability table, and the DGX Spark /
DGX Station specification pages.
