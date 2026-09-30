# llama.cpp CUDA Flash-Attention prefill kernels: code-level design brief

Source: `/home/wayne/dsh/QWen3.8-27B-GB10/tools/llama.cpp`, commit `4260903678a7525f43419dc234a942b551a8951e`
("fix(mamba) : make time-step projection input contiguous (#28832)", 2026-09-20).
All `file:line` refer to files under `ggml/src/ggml-cuda/`.

**Path drift vs. your task description.** This checkout has no `fattn-vec-f16.cuh` and no
`fattn-wmma-f16.cu`. The current file set is:

```
fattn.cu            745 lines   dispatch / kernel selection
fattn-common.cuh   1298 lines   constants, launch_fattn, combine/fixup kernels
fattn-mma-f16.cuh  2138 lines   THE prefill MMA kernel (vec/tile/mma dispatch target)
fattn-tile.cuh     1355 lines   generic SIMT tile kernel
fattn-tile.cu        60 lines
fattn-vec.cuh       609 lines   decode (1-2 query row) kernel
fattn-swizzle.cuh   126 lines   smem XOR swizzle + ldmatrix helpers
mma.cuh            1456 lines   tile<> types, mma(), ldmatrix, movmatrix
cp-async.cuh         57 lines   cp.async wrappers
```

Terminology warning: llama.cpp has **no `Br`/`Bc`**. The equivalent concepts are
`ncols1` (query rows per block), `ncols2` (Q heads batched per KV head, i.e. the GQA grouping),
`ncols = ncols1*ncols2` (total Q columns in the KQ tile), and `nbatch_fa` (KV rows per tile).

---

## 1. TILE GEOMETRY

### 1.1 The constants

`fattn-common.cuh:9`:
```c
#define FATTN_KQ_STRIDE       256
```

The MMA kernel's per-(DKQ,DV,ncols) config table is `fattn-mma-f16.cuh:11-25` (struct) and the
Ampere/Blackwell table at `fattn-mma-f16.cuh:39-88`:

```c
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.
```

`nbatch_fa` is Bc (KV tile). `ncols1` is Br (query rows). `ncols2` is the number of Q heads
sharing one K/V tile.

### 1.2 head_dim = 256, MMA kernel, Ampere config (this is what sm_121 uses)

`fattn-mma-f16.cuh:70-73`:
```c
GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4,  32, 128, 128, 128, 2, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);
```
Macro arg order (`fattn-mma-f16.cuh:26`):
`(DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine, nstages_target, Q_in_reg)`

| ncols | nthreads | occ | Bc=nbatch_fa | nbatch_K2 | nbatch_V2 | nbatch_combine | nstages | Q_in_reg |
|---|---|---|---|---|---|---|---|---|
| 8  | 128 (4 warps) | 2 | 64 | 128 | 128 | 128 | 2 | true |
| 16 |  64 (2 warps) | 4 | 32 | 128 | 128 | 128 | 2 | true |
| 32 | 128 (4 warps) | 2 | 32 | 128 | 128 | 128 | 2 | true |
| 64 | 128 (4 warps) | 2 | 32 | 128 | 128 | 128 | 2 | true |

**Key fact for head_dim=256: `nbatch_K2 = nbatch_V2 = 128 = DKQ/2 = DV/2`.** The whole 256-wide
head is loaded as one tile; both the K loop and the V loop run exactly one iteration.
Derived in `flash_attn_ext_f16_iter`:
`fattn-mma-f16.cuh:644` `for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2)`
-> `127 - 127%128 = 0`, single iteration.
`fattn-mma-f16.cuh:995` `for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2)`
-> `DV=256, 2*128=256`, single iteration.

Derived warp geometry (`fattn-mma-f16.cuh:601`, `1195-1197`, `get_cols_per_thread()` at `:330-336`,
`get_cols_per_warp()` at `:338-345`):

```c
constexpr int cols_per_warp   = T_B_KQ::I;
constexpr int cols_per_thread = get_cols_per_thread();   // == 2 on CUDA
constexpr int np = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // parallel warps per Q column
```

| ncols | cols_per_warp | np | Q cols per warp |
|---|---|---|---|
| 8  | 8  | 4 | 8 |
| 16 | 16 | 2 | 8 |
| 32 | 16 | 2 | 16 |
| 64 | 16 | 1 | 16 |

### 1.3 The case that actually runs for Qwen3.5-27B (gqa_ratio = 24/4 = 6), prefill

Dispatch (`fattn.cu:242-245`, `146-167`) picks **ncols2 = 8** (because `gqa_ratio > 4`) and, for
`Q->ne[1] > 4`, **ncols1 = 8** (from `ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/8, 8>`).
So `ncols = 64` and the config is the **last row**:

```
Br  = ncols1    = 8 query rows per block
Bc  = nbatch_fa = 32 KV rows
ncols2 = 8 Q heads share one K/V tile (6 are real, 2 are zero-padded, see below)
ncols  = 64 Q columns in the KQ tile
nthreads = 128 (4 warps), occupancy target 2, nstages 2, Q_in_reg true
```

Note the **25% wasted QK work**: `gqa_ratio=6` but `ncols2=8`, so columns c=6,7 of the Q tile are
loaded as zeros (`fattn-mma-f16.cuh:1270,1283`) and their outputs are skipped on writeback
(`:1714`). llama.cpp accepts this to avoid re-reading K/V 3x (ncols2=2 would need
`iter_z_gqa = ceil(6/2) = 3` passes over K/V).

### 1.4 TILE kernel (no tensor cores) for head_dim=256

`fattn-tile.cuh:70-74` (nvidia fp16):
```c
GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 2,  64,  64)
GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 2,  64,  64)
GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64,  64)
GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  64,  64)
GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  64,  64)
```
Macro order (`fattn-tile.cuh:12`): `(DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K)`.
So Bc = 64, and the head dim is consumed in chunks of `nbatch_K = 64` (4 chunks).

### 1.5 VEC kernel (decode only)

`fattn-vec.cuh:4-10`: `nthreads = 128` fixed, `__launch_bounds__(128, 1)` (`:20`).
No K/V tiling at all: the whole head dim is held per thread
(`Q_reg[ncols][(D/2)/nthreads_KQ]`, `fattn-vec.cuh:142-147`). `cols_per_block` is **1** if
`Q->ne[1]==1`, else **2** (`fattn-vec.cuh:551-567`). So the vec kernel has no `nbatch_fa`; it
streams KV rows one warp-width at a time (`:254`).

---

## 2. THE TWO MATMULS

**Both are `mma.sync` tensor-core ops in the MMA kernel.** Neither is FFMA. They differ in
accumulator precision.

### 2.1 QK^T = m16n8k16, fp32 accumulate

Wide path (ncols != 8, i.e. the case that runs for Qwen3.5 prefill), `fattn-mma-f16.cuh:676`:
```c
// swap A and B for CUDA.
mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
```
with types from `mma_tile_sizes<DV,ncols>` (`fattn-mma-f16.cuh:1069-1076`, CUDA branch):
```c
using T_A_KQ  = tile<16,  8, half2>; // row-major
using T_B_KQ  = tile<16,  8, half2>; // column-major
using T_C_KQ  = tile<16, 16, float>; // column-major
```
The overload (`mma.cuh:1196-1209`):
```c
template <data_layout dl_ab, data_layout dl_d>
static __device__ __forceinline__ void mma(
        tile<16, 16, float, dl_d> & D, const tile<16, 8, half2, dl_ab> & A, const tile<16, 8, half2, dl_ab> & B) {
    ...
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" ...);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" ...);
```
i.e. **two m16n8k16 issues build one 16x16 fp32 tile**. For ncols=8 the narrow path is used
(`fattn-mma-f16.cuh:668` `mma(KQ_C[i], K_A, Q_B[k])` with `T_C_KQ = tile<16,8,float>`,
`mma.cuh:1156-1179`, one `m16n8k16.row.col.f32.f16.f16.f32`).

### 2.2 P*V = m16n8k16, **fp16 accumulate**

`fattn-mma-f16.cuh:1015-1034`:
```c
for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
    for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
        const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

        T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
        ggml_cuda_fattn_smem_swizzle::load_ldmatrix_trans<stride_tile_V, swz_V>(A, tile_V, ...);
        if constexpr (T_B_KQ::I == 8) {
            mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
        } else {
            // Wide version of VKQ_C is column-major.
            // swap A and B for CUDA.
            mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);
        }
    }
}
```
`T_A_VKQ = tile<16,8,half2>`, `T_B_VKQ = tile<16,8,half2>`, `T_C_VKQ = tile<16,8,half2>`
(`fattn-mma-f16.cuh:1073-1075`). Overload (`mma.cuh:995-1022`):
```c
static __device__ __forceinline__ void mma(
        tile<16, 8, half2> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
    ...
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};" ...);
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};" ...);
```
For ncols==8 the narrow overload is `mma.cuh:970-993`
(`tile<16,4,half2>& D, tile<16,8,half2> A, tile<8,8,half2> B`, one `m16n8k16...f16.f16.f16.f16`).

**So the output accumulator VKQ_C is fp16, not fp32.** This is a deliberate precision trade
(the KQ row max is shifted up by `FATTN_KQ_MAX_OFFSET = 3*ln2 = 2.0794` so the exp() values sit
in a range where fp16 has enough mantissa; see `fattn-common.cuh:13-19`).

### 2.3 Fragment layouts (CUDA / Turing-Ampere-Blackwell branch)

`mma.cuh:226-271` for `tile<I,J,float,DATA_LAYOUT_I_MAJOR>`, `ne = I*J/32`:
```c
static __device__ __forceinline__ int get_i(const int l) {
    ...
    } else if constexpr (I == 16 && J == 8) {
        return ((l / 2) * 8) + (threadIdx.x / 4);
    } else if constexpr (I == 16 && J == 16) {
        return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
    ...
static __device__ __forceinline__ int get_j(const int l) {
    ...
    } else if constexpr (I == 16 && J == 8) {
        return ((threadIdx.x % 4) * 2) + (l % 2);
    } else if constexpr (I == 16 && J == 16) {
        return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
```
`mma.cuh:400-428` for `tile<I,J,half2,DATA_LAYOUT_I_MAJOR>`, `ne = I*J/32`:
```c
    } else if constexpr (I == 16 && J == 8) {
        return ((l % 2) * 8) + (threadIdx.x / 4);     // get_i
    ...
    } else if constexpr (I == 16 && J == 8) {
        return ((l / 2) * 4) + (threadIdx.x % 4);     // get_j
```
These are exactly the standard PTX `m16n8k16` A/B/C register-to-element maps, with
`threadIdx.x` == lane (the block is launched as `dim3(warp_size, nwarps, 1)`,
`fattn-common.cuh:1125`), so `threadIdx.y` is the warp.

Register footprint for the Qwen3.5 prefill config (DKQ=DV=256, ncols=64, 128 threads):
`Q_B[16]` x `tile<16,8,half2>` (4 regs) = 64 regs for Q, `VKQ_C[16]` x 4 regs = 64 regs for the
output accumulator, `KQ_C[2]` x `tile<16,16,float>` (8 regs) = 16 regs. ~144 regs of live data per
thread, which is why `occupancy = 2`.

---

## 3. THE P MATRIX (the key answer)

**P never touches shared memory in the MMA kernel.** After the softmax, the fp32 KQ accumulator
fragments are converted to fp16 fragments **in registers** and fed straight into the P*V `mma` as
the B operand. `fattn-mma-f16.cuh:961-973`:

```c
    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }
```

The conversion helpers (`mma.cuh:711-728`, CUDA branch):
```c
#if defined(TURING_MMA_AVAILABLE)
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
        }
        return ret;
    }

    static __device__ __forceinline__ tile<8, 8, half2> get_transposed(const tile<16, 4, half2> & t) {
        tile<8, 8, half2> ret;
        ret.x[0] = ggml_cuda_movmatrix(t.x[0]);
        ret.x[1] = ggml_cuda_movmatrix(t.x[1]);

        return ret;
    }
```
`ggml_cuda_movmatrix` is `movmatrix.sync.aligned.m8n8.trans.b16` (`mma.cuh:28-39`). So even the
narrow-path transpose is a register-only warp op, not a smem round trip.

Contrast with the two non-MMA kernels, which do exactly what your kernel does:

* **VEC** (`fattn-vec.cuh:125-129`):
  ```c
  __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
  ...
  __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
  ```
  written at `fattn-vec.cuh:303` (`KQ[j*nthreads + tid] = KQ_reg[j];`) and consumed by a scalar
  FFMA loop (`fattn-vec.cuh:351`):
  ```c
  VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
  ```
* **TILE** (`fattn-tile.cuh:662` comment "Calculate KQ softmax, write to shared KQ buffer,
  re-scale VKQ accumulators", then `fattn-tile.cuh:707-709` writes `tmp` into the `KQ` smem
  buffer, then a vectorized-but-still-SIMT `VKQ += KQ_k * V_k` loop at `fattn-tile.cuh:713-760`).

So: **your smem P round trip plus scalar FFMA is the vec/tile design. The fast design keeps P in
registers as fp16 mma fragments.** This alone is the single biggest structural difference.

---

## 4. ONLINE SOFTMAX

Running max/sum live in **registers**, one pair per KQ column, `cols_per_thread = 2` per thread
(`fattn-mma-f16.cuh:727-732`):
```c
    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};
```
`KQ_max` is initialised to `-FLT_MAX/2.0f` (`fattn-mma-f16.cuh:1242`).

Per key tile, the wide path (`fattn-mma-f16.cuh:826-903`):

1. Tile-local max, with the numerical-range offset (`:840`):
   ```c
   KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
   ```
   where `KQ_idx = (l/2) % 2` (`:838`) — because a KQ column is spread across 4 threads.
2. Cross-thread reduction inside the warp (`:864-867`):
   ```c
   for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
       KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
   }
   ```
   with `offset_first = 2, offset_last = 1` for Turing+ (`:849-850`); for ncols==8 it is
   `offset = 16,8,4` (`:772`).
3. exp and local rowsum accumulation (`:882-886`):
   ```c
   KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
   KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
   ```
   out-of-range rows are zeroed, not exp'd (`:885`).
4. **Rescale of the accumulator, applied directly to the VKQ fragments** (`:891-927`):
   ```c
   {
       float KQ_max_scale[cols_per_thread];
       for (int col = 0; col < cols_per_thread; ++col) {
           const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
           KQ_max_scale[col] = expf(KQ_max_diff);
           KQ_max[col] = KQ_max_new[col];

           *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

           // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
           KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
       }

       if constexpr (cols_per_warp == 8) {
           const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
           for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
               for (int l = 0; l < T_C_VKQ::ne; ++l) {
                   VKQ_C[i].x[l] *= KQ_max_scale_h2;
               }
           }
       } else {
           for (int col = 0; col < cols_per_thread; ++col) {
               const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
               for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
                   for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                       VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                   }
               }
           }
       }
   }
   ```
   Notes worth porting:
   * The rescale is a **half2 multiply on the fp16 accumulator fragments in place** — 64 registers
     for ncols=64, ~32 HMUL2. No smem, no fp32 accumulator.
   * `*((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;` is a bit-trick
     flush-to-zero: multiply the float's bit pattern by 0 or 1 (`SOFTMAX_FTZ_THRESHOLD = -20.0f`,
     `fattn-common.cuh:11`). Since `KQ_max_diff <= 0` always, this zeroes denormal exp() results.
   * `FATTN_KQ_MAX_OFFSET` (`fattn-common.cuh:19`) shifts the max up by `3*ln2` so exp values are
     representable in fp16.
   * There is **no rescale inside the K loop for KQ** — the KQ tile is computed for the whole
     `nbatch_fa` block, then softmaxed, then P*V, then the next key tile.
5. The final rowsum is combined across the `np` parallel warps and used to divide the output at the
   very end (`fattn-mma-f16.cuh:1732-1736`):
   ```c
   if (!needs_fixup && !is_fixup) {
       const float KQ_rowsum_j = meta_j[1];
       dstk_val.x /= KQ_rowsum_j;
       dstk_val.y /= KQ_rowsum_j;
   }
   ```
   and the cross-warp combine (`:1560-1595`) recomputes `KQ_cmn = max(...)`,
   `KQ_cms[i] = expf(meta[i].x - KQ_cmn)`, `KQ_crs += KQ_cms[i]*meta[i].y`.
6. Stream-k partial tiles are combined by a separate kernel
   (`flash_attn_stream_k_fixup_uniform` / `_general`, launched at `fattn-common.cuh:1265,1282`;
   the scalar reference math is `fattn-common.cuh:793-794`).

---

## 5. MEMORY PIPELINE

### 5.1 Loads: `cp.async.cg` 16B, or plain 16B copies

`fattn-mma-f16.cuh:367-477` `flash_attn_ext_f16_load_tile<stride_tile, swz, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>`.
Two branches:
```c
if constexpr (use_cp_async) {
    static_assert(warp_size == 32, "bad warp_size");
    constexpr int preload = 64;
    const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);
    ...
    cp_async_cg_16<preload>(tile_KV_32 + smem_offs_b, KV + i_KV*stride_KV + k*h2_per_chunk);
```
vs. the synchronous path using `ggml_cuda_memcpy_1<16>` (`:462-464`).
`cp_async_cg_16` emits `cp.async.cg.shared.global.L2::64B [dst], [src], 16;` (`cp-async.cuh:22-46`).
Only 16-byte copies are used (`cp-async.cuh:20`: "Only the 16 bit copy is exposed because 4 and 8
bit copies did not yield performance improvements" — "16 bit" here means 16 **byte**).

The load is done with decreasing granularity in the head-dim direction for better bandwidth
(`:372-374`): `ggml_cuda_unroll<6>{}(load)` sweeps `stride_k = warp_size >> n` for n=0..5, i.e.
32, 16, 8, 4, 2, 1 half2 per row, handling the 256/2 = 128 half2 row as 4x32 + 2x... (the
`k0_start`/`k0_stop` arithmetic at `:385-386` keeps each chunk aligned).

### 5.2 Shared-memory layout: XOR-swizzled, no padding when possible

`fattn-swizzle.cuh:6-47`:
```c
// XOR swizzle for K/V SMEM tiles to avoid bank conflicts without row padding (Turing+ only).
// Stride must be a multiple of 32 half2 columns, otherwise we keep +4 row padding.
static __host__ __device__ constexpr bool bank_aligned(const int nbatch_2) {
    return nbatch_2 >= 32 && nbatch_2 % 32 == 0;
}
...
static __device__ constexpr int tile_stride(const int nbatch_2) {
    return enabled(nbatch_2) ? nbatch_2 : nbatch_2 + 4;
}
...
// Swizzled byte offset for tile element (row, col_h2), same map used for writes and reads.
template<int stride_h2>
static __device__ __forceinline__ int bytes_rc(const int row, const int col_h2) {
    static_assert(bank_aligned(stride_h2), "swizzled tile needs a stride that is a multiple of 32");
    return ((row * stride_h2 + col_h2) * (int) sizeof(half2)) ^ ((row & 7) << 4);
}
```
For head_dim=256: `nbatch_K2 = nbatch_V2 = 128`, which is a multiple of 32, so **swizzle is
enabled and there is no padding** (`stride_tile_K = stride_tile_V = 128`). For other head dims
(e.g. 80 -> nbatch=40) it falls back to `stride = nbatch + 4`.

Smem layout (`fattn-mma-f16.cuh:1220-1223`):
```c
extern __shared__ half2 tile_Q[];
half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
half2 * tile_V    =           nstages > 1 ? tile_K + nbatch_fa * stride_tile_K : tile_K;
half  * tile_mask = (half *) (nstages > 1 ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);
```
Sizes (`fattn-mma-f16.cuh:1990-2003`):
```c
const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(stride_tile_K,  stride_tile_V) * sizeof(half2);
const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (stride_tile_K + stride_tile_V) * sizeof(half2);
const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);
```
For the Qwen3.5 prefill config: Q 64x132x4 = 33792 B, KV(2-stage) 32x(128+128)x4 = 32768 B,
mask 8x20x4 = 640 B, combine 4x16x132x4 = 33792 B -> total = **33792 B** dynamic smem
(`Q_in_reg` means `max(combine, max(Q, KV+mask))`). The dynamic smem attribute is raised once per
device via `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
(`fattn-mma-f16.cuh:2026,2035,2048,2061`).

Note `tile_Q` is **also** the epilogue combine buffer; the 16 bytes of per-row padding
(`stride_tile_Q = DKQ/2 + 4`) hold the `float2(KQ_max, KQ_rowsum)` metadata
(`fattn-mma-f16.cuh:1492-1493,1532`).

### 5.3 Pipelining: 2 stages, one K buffer + one V buffer

`nstages_target` is 2 for all head_dim=256 configs and is clamped to {1,2}
(`static_assert((nstages_target_) >= 1 && (nstages_target_) <= 2)`, `fattn-mma-f16.cuh:34`), and
is forced to 0 when `ncols2 < 2` (`:350`, `:356`). So this is **not** a multi-stage ring; it is a
two-buffer scheme where K(i+1) is prefetched while V(i) is being multiplied:

* Prologue (`fattn-mma-f16.cuh:1305-1317`): load mask and K for key tile 0 with `cp_async`.
* Top of `iter` (`fattn-mma-f16.cuh:623-631`): wait, sync, then issue the **V tile for the current
  key tile** with cp.async:
  ```c
  if constexpr (nstages > 1) {
      static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
      static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
      static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
      constexpr bool use_cp_async = true;
      cp_async_wait_all();
      __syncthreads();
      flash_attn_ext_f16_load_tile<stride_tile_V, swz_V, ...>(V_h2, tile_V, nbatch_V2, stride_V, k_VKQ_0, k_VKQ_sup, nullptr);
  }
  ```
  Then the QK^T mma consumes the already-resident `tile_K`.
* After softmax, before P*V (`fattn-mma-f16.cuh:975-990`): wait for the V copy, then issue the
  **K tile for the next key tile**:
  ```c
  cp_async_wait_all();
  __syncthreads();
  if (!last_iter) {
      ...
      flash_attn_ext_f16_load_tile<stride_tile_K, swz_K, ...>(K_h2, tile_K, nbatch_K2, stride_K, k_VKQ_0 + nbatch_fa, k_VKQ_sup, nullptr);
  }
  ```
  Then the P*V mma consumes `tile_V`.
* `cp_async_wait_all()` is `cp.async.wait_all` (`cp-async.cuh:51-57`) — it waits for **all**
  outstanding copies, and `__syncthreads()` is required separately
  (`cp-async.cuh:48-50`).

With `nstages <= 1` the kernel instead does synchronous `ggml_cuda_memcpy_1<16>` loads with
`__syncthreads()` before each matmul (`fattn-mma-f16.cuh:647-656, 999-1009`).

---

## 6. KV LAYOUT

### 6.1 Layout used

K and V are plain ggml tensors with `ne[0] = head_dim`, `ne[1] = n_kv`, `ne[2] = n_head_kv`,
`ne[3] = n_seq`. The kernel only ever uses the strides:
`fattn-mma-f16.cuh:1854-1857`:
```c
const int stride_K    = nb11 / sizeof(half2);
...
const int stride_V = V_is_K_view ? stride_K : nb21 / sizeof(half2);
```
and indexes rows as `KV + i_KV*stride_KV + k*h2_per_chunk` (`:416,418,459,464`).

**No interleaving is required.** Head selection is by stride:
`fattn-mma-f16.cuh:1885` `K_h2 = (const half2 *) (K + nb13*sequence + nb12*z_KV);` and GQA maps Q
head `z` to KV head `z / gqa_ratio` via `const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2;`
(`:1882`). That is the standard ggml `[head_dim, n_kv, n_head, n_seq]` layout.

The only layout constraints come from the "GQA optimization" gate, `fattn.cu:183-194`:
```c
bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
for (const ggml_tensor * t : {Q, K, V, mask}) {
    if (t == nullptr || ggml_is_quantized(t->type)) { continue; }
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (t->nb[i] % 16 != 0) { use_gqa_opt = false; break; }
    }
}
```
So: **K/V must be padded so that `n_kv` is a multiple of 256 (`FATTN_KQ_STRIDE`)**, and all
higher-dim strides must be 16-byte aligned. The padding is applied by the KV cache itself
(`src/llama-kv-cache.cpp:1255` `const uint32_t n_pad_cur = std::max(n_pad, 256u);`). Without this,
`ncols2` collapses to 1 and you lose all GQA batching.

### 6.2 V is NOT stored transposed; `ldmatrix.trans` is used

`tile_V` is loaded in the natural `[kv_row][dv]` orientation (rows = KV index, columns = head dim;
the `V_h2 + i0_start/2` offset at `:1004` walks the head-dim direction). The transposition happens
at fragment-load time:
`fattn-mma-f16.cuh:1021-1022`:
```c
T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
ggml_cuda_fattn_smem_swizzle::load_ldmatrix_trans<stride_tile_V, swz_V>(A, tile_V, ...);
```
which is `ldmatrix.sync.aligned.m8n8.x4.trans.b16` (`fattn-swizzle.cuh:61-70`,
`mma.cuh:884-894`):
```c
asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
    : "l"(xs));
```
So: **store V as `[dv][kv]` (head-dim contiguous) and transpose on the ldmatrix, do not build a
transposed V copy.**

`V_is_K_view` (V aliasing K, for MLA) exists but is only enabled for `DKQ == 576`
(`fattn-mma-f16.cuh:1988`), not for 256.

---

## 7. DISPATCH THRESHOLDS

`fattn.cu:517-688` `ggml_cuda_get_best_fattn_kernel`. The NVIDIA tensor-core branch
(`fattn.cu:609-636`):

```c
    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) { return BEST_FATTN_KERNEL_VEC; }
                } else {
                    if (Q->ne[1] == 1) { return BEST_FATTN_KERNEL_VEC; }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }
```

**Summary for an NVIDIA part with tensor cores (sm_121 included):**

| condition | kernel |
|---|---|
| fp16/bf16 K&V, `Q->ne[1] == 1`, `Q->ne[3] == 1`, `!(gqa_ratio>4 && n_kv>=8192)`, cc>=Ada | VEC |
| fp16/bf16 K&V, `Q->ne[1] == 1`, but `gqa_ratio>4 && n_kv>=8192` (your case at long ctx) | MMA |
| `Q->ne[1] == 1` and GQA opt does not apply (no mask / ALiBi / unpadded KV) | VEC |
| quantized K or V, `Q->ne[1] <= 2` (Ada+) | VEC |
| **`Q->ne[1] >= 2` (any prefill)** | **MMA** |
| no tensor cores at all | TILE (or VEC if `Q->ne[1] <= 2`) |

Volta-only thresholds (`fattn.cu:644-652`) and AMD MFMA/WMMA thresholds
(`fattn.cu:654-671`) are separate; ignore them for sm_121.

**So for Qwen3.5-27B prefill, the answer is unambiguous: the MMA kernel, always.**
`BEST_FATTN_KERNEL_TILE` is only reachable without tensor cores
(`fattn.cu:673-687` "If there are no tensor cores available, use the generic tile kernel").
`BEST_FATTN_KERNEL_VEC` never applies for `Q->ne[1] > 2`.

Then within the MMA kernel, `ncols2` from GQA (`fattn.cu:169-262`) and `ncols1` from query length
(`fattn.cu:132-167`):
```c
    if (use_gqa_opt && gqa_ratio > 4) { ...switch_ncols1<DKQ, DV, 8>(ctx, dst); return; }
    if (use_gqa_opt && gqa_ratio > 2) { ...switch_ncols1<DKQ, DV, 4>(ctx, dst); return; }
    if (use_gqa_opt && gqa_ratio > 1) { ...switch_ncols1<DKQ, DV, 2>(ctx, dst); return; }
```
and
```c
    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) { case<DKQ, DV, 8/ncols2, ncols2>; return; }
    }
    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) { case<DKQ, DV, 16/ncols2, ncols2>; return; }
    }
    if (Q->ne[1] <= 32/ncols2 || ...) { case<DKQ, DV, 32/ncols2, ncols2>; return; }
    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
```
Also `ncols2_max = 8` for DKQ=256 (`fattn.cu:638`), and only head dims
{40,64,72,80,96,112,128,192,256,320,512,576} are supported at all (`fattn.cu:552-599`);
head_dim=256 is accepted only when `V->ne[0] == K->ne[0]` (`fattn.cu:560-563`).

**Scheduling** (`fattn-common.cuh:1135-1170`): stream-k is enabled unconditionally on Ada and
newer (`should_use_stream_k` returns `true` for `cc >= GGML_CUDA_CC_ADA_LOVELACE`,
`fattn-common.cuh:1140-1142`), so on sm_121 the KV dimension is split across
`min(max_blocks_per_sm*nsm, ntiles_KV*ntiles_dst)` blocks and partial results are combined by a
second kernel. On a 48-SM part this matters: with `ncols1=8`, a 4096-token prefill gives
`ntiles_dst = 512` and only `2*48 = 96` resident blocks, i.e. ~5.3 waves without stream-k.

---

## 8. WHY IT IS FAST, AND WHAT A MINIMAL sm_121 KERNEL LOOKS LIKE

### 8.1 The design decisions that matter

1. **Both matmuls on tensor cores, with the right accumulator types.**
   QK^T = `m16n8k16 f32.f16.f16.f32` (needs fp32 for the softmax), P*V =
   `m16n8k16 f16.f16.f16.f16` (fp16 is fine because P is a probability and is bounded, and the
   max-offset trick keeps it in range). Neither is FFMA.
2. **P stays in registers.** fp32 KQ accumulator -> `get_half2` (+ `movmatrix` transpose for the
   narrow case) -> mma B operand. Zero shared-memory traffic for P, zero FFMA. This is the design
   element your kernel is missing.
3. **The whole head dim is one tile.** `nbatch_K2 = nbatch_V2 = DKQ/2 = 128`, so there is no inner
   k-loop over the head dim; the QK^T is a single `mma` per 16x16 output tile and the P*V is a
   single `mma` per 16(dv)x16(col) output tile. This maximises the reuse of each loaded K/V byte
   inside the register file.
4. **GQA batching (`ncols2`) + query-row batching (`ncols1`) raise arithmetic intensity per KV
   byte.** For ncols=64, each 32x256 K/V tile (32 KB) feeds 2.1 MFLOP -> ~64 FLOP/byte of K/V.
   On a GB10-class part with LPDDR5X bandwidth this reuse ratio is the actual limiter, not the
   tensor-core issue rate.
5. **`cp.async` double buffering** (K(i+1) prefetched during P*V(i), V(i) issued during QK^T(i)),
   with `L2::64B` prefetch hints, and **XOR-swizzled smem with no row padding** so the ldmatrix
   loads are bank-conflict-free without wasting smem.
6. **`ldmatrix` / `ldmatrix.trans` for every smem->register fragment move.** K and Q use
   `ldmatrix.x4.b16`; V uses `ldmatrix.x4.trans.b16` so no transposed V copy is needed.
7. **Swizzled shared-memory staging for the epilogue** with the row-max/rowsum metadata packed
   into the 16 bytes of padding per row (`tile_stride = nbatch_combine + 4`,
   `fattn-mma-f16.cuh:1477,1493,1532`), so the final divide is fused into the global write.
8. **Stream-k work distribution** for load balance across the (few) SMs.

### 8.2 Minimal correct FA2-style prefill kernel for head_dim=256, GQA 6:1, 48 SMs, sm_121

Port exactly the `ncols=64` configuration. Concretely:

```
Block:    128 threads = 4 warps, __launch_bounds__(128, 2)
Q tile:   Br = 8 query rows x ncols2 = 8 Q heads = 64 Q columns
KV tile:  Bc = 32 rows, full head_dim 256 (128 half2) -> 1 K tile + 1 V tile
Grid:     ntiles_x = ceil(n_q / 8); stream-k over the KV dimension
          (max concurrent blocks = 2 * 48 = 96)
```

Per warp: 16 of the 64 Q columns, all 256 head-dim elements, full KV range.

Steps:
1. Load Q for the 8x8 (row, head) block, multiply by `scale`, convert to fp16, stage in smem with
   `stride = DKQ/2 + 4 = 132` half2 (`fattn-mma-f16.cuh:1212,1276`), then `ldmatrix.x4` into
   `Q_B[16]` register fragments (`:1295-1297`) — 64 registers.
2. `cp.async.cg` 16B loads for the K tile and V tile, XOR-swizzled
   (`off = (row*128 + col_h2)*4 ^ ((row & 7) << 4)`), 2-stage.
3. QK^T: for each 16-KV-row chunk (2 chunks of 16 to cover Bc=32) and each 16-wide head-dim chunk
   (16 chunks), issue `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` x2 to produce
   `tile<16,16,float>` (16 ncols x 16 kv). fp32 accumulator.
4. Softmax per KQ column: `fmaxf` over fragments + `__shfl_xor_sync` at offsets 2 and 1;
   `expf(v - max)`; accumulate rowsum; rescale the fp16 `VKQ_C[16]` accumulator in place with
   `half2 *= make_half2(expf(max_old - max_new), ...)`; zero out-of-range rows.
5. P*V: `get_half2` the fp32 KQ fragment into `tile<16,8,half2>` (register-only, in place), load
   the V^T fragment with `ldmatrix.x4.trans`, then
   `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` with fp16 accumulation.
6. Epilogue: write `VKQ_C` to the swizzled `tile_Q` buffer, store `float2(max, rowsum)` into the
   row padding, `__syncthreads`, then divide by rowsum and write fp32 to global (or to the stream-k
   fixup buffer if the block only covered part of the tile).
7. Combine stream-k partials in a second kernel.

Expected impact for your kernel: your ~1.3% figure is consistent with (a) P going through smem and
being re-read by scalar FFMA, which serialises the P*V behind the softmax, and (b) a tile shape
with low KV reuse. Fixing P-in-registers and using the `ncols1=8, ncols2=8, nbatch_fa=32` shape
addresses both.

### 8.3 Caveats / things NOT in this source

* **There is no `tcgen05` / TMA / `wgmma` anywhere in `ggml-cuda`** (verified: `grep -rn tcgen05 .`
  returns 0 hits). The FA kernels use classic `mma.sync.m16n8k16`. On sm_121 (GB10 / DGX Spark)
  `mma.sync` is supported but is not the peak-throughput path; the bf16 "peak" you are comparing
  against is likely the tcgen05 number, so a `mma.sync` implementation will top out below it. If
  you need the full Blackwell tensor-core rate you will have to write `tcgen05.mma` yourself;
  llama.cpp gives you a correct, well-tuned `mma.sync` reference, not a Blackwell-optimal one.
* `blackwell_mma_available(cc)` (`common.cuh:376-379`) only gates `mxf4`/`mxf8` mma helpers in
  `mma.cuh:1138-1145` (used by quantized matmul, not FA).
* The FA kernel requires `Q->type == GGML_TYPE_F32` and `KQV->type == GGML_TYPE_F32`
  (`fattn-common.cuh:994-995`), and `mask->type == GGML_TYPE_F16` if a mask is present (`:1001`).
* K/V in fp32 or bf16 are converted to fp16 first (`fattn-common.cuh:1026-1088`); quantized
  K/V types are only supported by the VEC kernel and are dequantized on the fly there. The MMA
  kernel's config table is fp16-only for the A/B operands.
* `ncols2` values are powers of two only (1,2,4,8,16,32), so `gqa_ratio = 6` cannot be matched
  exactly; 25% of the QK^T work in the ncols2=8 tile is wasted. If you control the kernel, a
  `ncols2 = 6` variant (or 3 x ncols2=2) would recover it, at the cost of more K/V traffic.
