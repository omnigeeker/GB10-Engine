// Batched-prefill GEMM kernels.
//
// The decode GEMV and the prefill GEMM want opposite tilings. Decode reads
// each weight exactly once and produces one output, so it blocks over N and
// streams W -- it is purely DRAM-bound. Prefill reuses every weight across all
// T prompt tokens, so it must block over (N, T) and hold a T-wide accumulator.
//
// Two earlier attempts at this failed, and the failures are why the design
// below looks the way it does:
//
//   * NR=1, TILE_T=8 (each warp owns one row, x staged in shared): 2054 ms.
//     Every FMA pair costs 2 shared loads, so it is shared-throughput-bound.
//   * NR=4 (register-blocking the rows): 2650 ms. Worse -- 80 registers drops
//     occupancy to 50% and the kernel is latency-bound.
//   * TILE_T=64, NR=1 (to read each weight once instead of once per T-tile):
//     2838 ms. `xv0[64]`/`xv1[64]` is 128 registers on top of a 64-wide
//     accumulator, so it spilled.
//
// So both obvious levers are blocked by register pressure, and the tiling
// itself has to change. This version stages *both* operands in shared memory
// and gives each thread a 2D register tile:
//
//   * block covers TILE_N=64 rows x TILE_T=64 tokens
//   * Wtile and xtile are staged in shared, stored TRANSPOSED as [k][row] --
//     stored as [row][k] every inner-loop read is a 32-way bank conflict,
//     because the stride between threads is exactly the tile width
//   * each thread owns a 4x4 sub-tile, so the inner loop reads 4 W and 4 x
//     values per 16 FMAs (1:2 shared-to-FMA) with a 16-register accumulator
//   * TILE_T covers the whole prompt, so each weight is read exactly once
//
// x traffic is `(N / 64) * T * K * 4` bytes but is served from L1/L2 because
// every thread in the block shares the same tile.

#include "gemv_common.cuh"
#include <cuda_bf16.h>

using namespace gb10;

#define GB10_TN 64    // rows of N per block
#define GB10_TT 64    // prompt tokens per block
#define GB10_KC 32    // k values per chunk
// K is split across blockIdx.z so the prefill GEMM can offer more than the 5.7
// blocks/SM it does today (measured 47% occupancy, bound by BOTH the register
// budget of 5.3 blocks/SM and the grid -- docs/NEXT.md round 148). The atomics
// this needs were measured at ~100x the required rate (~1% of a prefill).
//
// kc_half must be EVEN: the double-buffer parity below is (c ^ 1) & 1, which
// stays correct only because c now starts at an even number. nchunk = 160 and a
// 2-way split gives 80, so it holds here. The host must fall back to grid.z = 1
// when nchunk / GB10_KSPLIT is odd.
#define GB10_KSPLIT 2 // chunks of K per block; host sets grid.z
#define GB10_TM 8     // rows of N owned by one thread
#define GB10_TNREG 4  // tokens owned by one thread
#define GB10_GEMM_BLOCK 128

// Padded strides. The pad must keep each row 16-byte aligned (a multiple of 4
// floats) so the inner loop can read a whole 4-wide sub-tile with one LDS.128
// instead of four LDS.32; the extra float beyond that keeps consecutive `k`
// slices off the same shared bank.
#define GB10_WSTRIDE GB10_TN
#define GB10_XSTRIDE GB10_TT

// Stage the [TILE_N, KC] weight chunk, decoded to fp32, as `wt[k][n]`.
template <int KC>
__device__ __forceinline__ void stage_wtile(uint16_t (*wt)[GB10_WSTRIDE],
                                            const uint8_t* __restrict__ w,
                                            const uint8_t* __restrict__ sc,
                                            const float* __restrict__ s2, int K, int nbase,
                                            int N, int c) {
    // 16 consecutive NVFP4 elements per thread instead of 8: one 8-byte load
    // and one group scale cover the pair, so the staging loop runs half as many
    // iterations and issues half as many loads for the same bytes.
    constexpr int PAIRS = KC / 16;
    constexpr int UNITS = GB10_TN * PAIRS;
    constexpr int P = (UNITS + GB10_GEMM_BLOCK - 1) / GB10_GEMM_BLOCK;
    (void)s2;
    uint2 pk[P];
    float scl[P];
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int u = threadIdx.x + p * GB10_GEMM_BLOCK;
        const int nl = u / PAIRS, pr = u % PAIRS;
        const int n = nbase + nl;
        pk[p] = make_uint2(0u, 0u);
        scl[p] = 0.0f;
        if (u < UNITS && n < N) {
            const int kbase = c * KC + pr * 16;
            pk[p] = *reinterpret_cast<const uint2*>(w + (size_t)n * (K >> 1) + (kbase >> 1));
            scl[p] = e4m3_to_float(__ldg(sc + (size_t)n * (K >> 4) + (kbase >> 4)));
        }
    }
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int u = threadIdx.x + p * GB10_GEMM_BLOCK;
        const int nl = u / PAIRS, pr = u % PAIRS;
        const int n = nbase + nl;
        if (u < UNITS && n < N) {
            const uint32_t lo = pk[p].x, hi = pk[p].y;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint8_t n0 = (uint8_t)((lo >> (4 * j)) & 0xF);
                const uint8_t n1 = (uint8_t)((hi >> (4 * j)) & 0xF);
                wt[pr * 16 + j][nl] =
                    __bfloat16_as_ushort(__float2bfloat16_rn(e2m1_to_float(n0) * scl[p]));
                wt[pr * 16 + 8 + j][nl] =
                    __bfloat16_as_ushort(__float2bfloat16_rn(e2m1_to_float(n1) * scl[p]));
            }
        }
    }
}

template <int KC>
__device__ __forceinline__ void stage_wtile_fp8(uint16_t (*wt)[GB10_WSTRIDE],
                                                const uint8_t* __restrict__ w,
                                                const float* __restrict__ s1, int K, int nbase,
                                                int N, int c) {
    // 16 elements per thread, matching the nvfp4 path: one 16-byte load and no
    // group scales to fetch (fp8 carries a single per-tensor scale).
    constexpr int GROUPS = KC / 16;
    constexpr int UNITS = GB10_TN * GROUPS;
    constexpr int P = (UNITS + GB10_GEMM_BLOCK - 1) / GB10_GEMM_BLOCK;
    const float wscale = __ldg(s1);
    uint4 pk[P];
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int u = threadIdx.x + p * GB10_GEMM_BLOCK;
        const int nl = u / GROUPS, gr = u % GROUPS;
        const int n = nbase + nl;
        pk[p] = make_uint4(0u, 0u, 0u, 0u);
        if (u < UNITS && n < N)
            pk[p] = *reinterpret_cast<const uint4*>(w + (size_t)n * K + c * KC + gr * 16);
    }
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int u = threadIdx.x + p * GB10_GEMM_BLOCK;
        const int nl = u / GROUPS, gr = u % GROUPS;
        const int n = nbase + nl;
        if (u < UNITS && n < N) {
            const uint8_t* pb = reinterpret_cast<const uint8_t*>(&pk[p]);
#pragma unroll
            for (int j = 0; j < 16; ++j)
                wt[gr * 16 + j][nl] =
                    __bfloat16_as_ushort(__float2bfloat16_rn(e4m3_to_float(pb[j]) * wscale));
        }
    }
}

template <int KC>
__device__ __forceinline__ void stage_wtile_bf16(uint16_t (*wt)[GB10_WSTRIDE],
                                                 const uint16_t* __restrict__ w, int K, int nbase,
                                                 int N, int c) {
    for (int u = threadIdx.x; u < GB10_TN * GB10_KC; u += GB10_GEMM_BLOCK) {
        const int nl = u / GB10_KC, kl = u % GB10_KC;
        const int n = nbase + nl;
        uint16_t v = 0;
        if (n < N) v = __ldg(w + (size_t)n * K + c * GB10_KC + kl);
        wt[kl][nl] = v;
    }
}

// Stage the [TILE_T, KC] activation chunk as `xt[k][t]`.
//
// Read as float4: the naive element-at-a-time version issues 8 scalar global
// loads per thread per chunk, which at ~500 cycles of latency each is what the
// double buffering was struggling to hide. Two float4 loads per thread do the
// same work, and x is contiguous in k so the vector is naturally aligned
// (K, KC and the segment stride are all multiples of 4).
template <int KC>
__device__ __forceinline__ void stage_xtile(float (*xt)[GB10_XSTRIDE],
                                            const float* __restrict__ x, int K, int T, int t0,
                                            int c) {
    // 16 k-values per thread as four float4 loads. Previously each thread took
    // 8, which at the block sizes here meant one iteration of mostly-zero writes
    // whenever `t < T` -- and at t=1 that is every thread but one.
    constexpr int GROUPS = KC / 16;
    constexpr int UNITS = GB10_TT * GROUPS;
    constexpr int P = (UNITS + GB10_GEMM_BLOCK - 1) / GB10_GEMM_BLOCK;
    float4 v[P][4];
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int u = threadIdx.x + p * GB10_GEMM_BLOCK;
        const int tl = u / GROUPS, gr = u % GROUPS;
        const int t = t0 + tl;
        if (u < UNITS && t < T) {
            const float* src = x + (size_t)t * K + c * KC + gr * 16;
#pragma unroll
            for (int q = 0; q < 4; ++q)
                v[p][q] = *reinterpret_cast<const float4*>(src + q * 4);
        }
    }
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int u = threadIdx.x + p * GB10_GEMM_BLOCK;
        const int tl = u / GROUPS, gr = u % GROUPS;
        const int t = t0 + tl;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (u < UNITS && t < T) {
                const float4 f = v[p][q];
                xt[gr * 16 + q * 4 + 0][tl] = f.x;
                xt[gr * 16 + q * 4 + 1][tl] = f.y;
                xt[gr * 16 + q * 4 + 2][tl] = f.z;
                xt[gr * 16 + q * 4 + 3][tl] = f.w;
            } else if (u < UNITS) {
                xt[gr * 16 + q * 4 + 0][tl] = 0.0f;
                xt[gr * 16 + q * 4 + 1][tl] = 0.0f;
                xt[gr * 16 + q * 4 + 2][tl] = 0.0f;
                xt[gr * 16 + q * 4 + 3][tl] = 0.0f;
            }
        }
    }
}

// The 4x4 outer product. Identical for every quantisation.
__device__ __forceinline__ void gemm2d_outer(const float (*wt)[GB10_WSTRIDE],
                                             const float (*xt)[GB10_XSTRIDE],
                                             float (&acc)[GB10_TM][GB10_TNREG], int ty, int tx) {
    static_assert(GB10_TM == 8 && GB10_TNREG == 4, "float4 path assumes 8x4 tiles");
#pragma unroll
    for (int k = 0; k < GB10_KC; ++k) {
        // TM=8/TNREG=4: two weight loads and ONE activation load per k, so each
        // activation feeds 8 rows instead of 4 (40 B -> 32 B of shared per 32 FMA).
        const float4 wv0 = *reinterpret_cast<const float4*>(&wt[k][ty * GB10_TM]);
        const float4 wv1 = *reinterpret_cast<const float4*>(&wt[k][ty * GB10_TM + 4]);
        float4 xv = *reinterpret_cast<const float4*>(&xt[k][tx * GB10_TNREG]);
        // Diagnostic harness, off unless built with -DGB10_SIM_BF16_ACT=1 (add
        // the define in crates/gb10-cuda/build.rs). It simulates the activation
        // rounding a bf16 tensor-core GEMM imposes, to answer whether the
        // token-exact gate tolerates it. Measured: it does -- `generate --n 16
        // --repeat 8 --oracle` stays 16/16, which is what clears the cuBLAS
        // rewrite to proceed. See bench/longctx/tensorcore-plan.md.
#ifdef GB10_SIM_BF16_ACT
        xv.x = __bfloat162float(__float2bfloat16_rn(xv.x));
        xv.y = __bfloat162float(__float2bfloat16_rn(xv.y));
        xv.z = __bfloat162float(__float2bfloat16_rn(xv.z));
        xv.w = __bfloat162float(__float2bfloat16_rn(xv.w));
#endif
        acc[0][0] = fmaf(wv0.x, xv.x, acc[0][0]);
        acc[0][1] = fmaf(wv0.y, xv.y, acc[0][1]);
        acc[0][2] = fmaf(wv0.z, xv.z, acc[0][2]);
        acc[0][3] = fmaf(wv0.w, xv.w, acc[0][3]);
        acc[1][0] = fmaf(wv0.x, xv.x, acc[1][0]);
        acc[1][1] = fmaf(wv0.y, xv.y, acc[1][1]);
        acc[1][2] = fmaf(wv0.z, xv.z, acc[1][2]);
        acc[1][3] = fmaf(wv0.w, xv.w, acc[1][3]);
        acc[2][0] = fmaf(wv0.x, xv.x, acc[2][0]);
        acc[2][1] = fmaf(wv0.y, xv.y, acc[2][1]);
        acc[2][2] = fmaf(wv0.z, xv.z, acc[2][2]);
        acc[2][3] = fmaf(wv0.w, xv.w, acc[2][3]);
        acc[3][0] = fmaf(wv0.x, xv.x, acc[3][0]);
        acc[3][1] = fmaf(wv0.y, xv.y, acc[3][1]);
        acc[3][2] = fmaf(wv0.z, xv.z, acc[3][2]);
        acc[3][3] = fmaf(wv0.w, xv.w, acc[3][3]);
        acc[4][0] = fmaf(wv1.x, xv.x, acc[4][0]);
        acc[4][1] = fmaf(wv1.y, xv.y, acc[4][1]);
        acc[4][2] = fmaf(wv1.z, xv.z, acc[4][2]);
        acc[4][3] = fmaf(wv1.w, xv.w, acc[4][3]);
        acc[5][0] = fmaf(wv1.x, xv.x, acc[5][0]);
        acc[5][1] = fmaf(wv1.y, xv.y, acc[5][1]);
        acc[5][2] = fmaf(wv1.z, xv.z, acc[5][2]);
        acc[5][3] = fmaf(wv1.w, xv.w, acc[5][3]);
        acc[6][0] = fmaf(wv1.x, xv.x, acc[6][0]);
        acc[6][1] = fmaf(wv1.y, xv.y, acc[6][1]);
        acc[6][2] = fmaf(wv1.z, xv.z, acc[6][2]);
        acc[6][3] = fmaf(wv1.w, xv.w, acc[6][3]);
        acc[7][0] = fmaf(wv1.x, xv.x, acc[7][0]);
        acc[7][1] = fmaf(wv1.y, xv.y, acc[7][1]);
        acc[7][2] = fmaf(wv1.z, xv.z, acc[7][2]);
        acc[7][3] = fmaf(wv1.w, xv.w, acc[7][3]);
    }
}

// Same 4x4 outer product, reading the bf16 weight tile.
__device__ __forceinline__ void gemm2d_outer_bf16(const uint16_t (*wt)[GB10_WSTRIDE],
                                                  const float (*xt)[GB10_XSTRIDE],
                                                  float (&acc)[GB10_TM][GB10_TNREG], int ty,
                                                  int tx) {
    static_assert(GB10_TM == 8 && GB10_TNREG == 4, "bf16 path assumes 8x4 tiles");
#pragma unroll
    for (int k = 0; k < GB10_KC; ++k) {
        const uint4 wq = *reinterpret_cast<const uint4*>(&wt[k][ty * GB10_TM]);
        const float2 a01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.x));
        const float2 a23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.y));
        const float2 a45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.z));
        const float2 a67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.w));
        const float wv[8] = {a01.x, a01.y, a23.x, a23.y, a45.x, a45.y, a67.x, a67.y};
        const float4 xv = *reinterpret_cast<const float4*>(&xt[k][tx * GB10_TNREG]);
        acc[0][0] = fmaf(wv[0], xv.x, acc[0][0]);
        acc[0][1] = fmaf(wv[0], xv.y, acc[0][1]);
        acc[0][2] = fmaf(wv[0], xv.z, acc[0][2]);
        acc[0][3] = fmaf(wv[0], xv.w, acc[0][3]);
        acc[1][0] = fmaf(wv[1], xv.x, acc[1][0]);
        acc[1][1] = fmaf(wv[1], xv.y, acc[1][1]);
        acc[1][2] = fmaf(wv[1], xv.z, acc[1][2]);
        acc[1][3] = fmaf(wv[1], xv.w, acc[1][3]);
        acc[2][0] = fmaf(wv[2], xv.x, acc[2][0]);
        acc[2][1] = fmaf(wv[2], xv.y, acc[2][1]);
        acc[2][2] = fmaf(wv[2], xv.z, acc[2][2]);
        acc[2][3] = fmaf(wv[2], xv.w, acc[2][3]);
        acc[3][0] = fmaf(wv[3], xv.x, acc[3][0]);
        acc[3][1] = fmaf(wv[3], xv.y, acc[3][1]);
        acc[3][2] = fmaf(wv[3], xv.z, acc[3][2]);
        acc[3][3] = fmaf(wv[3], xv.w, acc[3][3]);
        acc[4][0] = fmaf(wv[4], xv.x, acc[4][0]);
        acc[4][1] = fmaf(wv[4], xv.y, acc[4][1]);
        acc[4][2] = fmaf(wv[4], xv.z, acc[4][2]);
        acc[4][3] = fmaf(wv[4], xv.w, acc[4][3]);
        acc[5][0] = fmaf(wv[5], xv.x, acc[5][0]);
        acc[5][1] = fmaf(wv[5], xv.y, acc[5][1]);
        acc[5][2] = fmaf(wv[5], xv.z, acc[5][2]);
        acc[5][3] = fmaf(wv[5], xv.w, acc[5][3]);
        acc[6][0] = fmaf(wv[6], xv.x, acc[6][0]);
        acc[6][1] = fmaf(wv[6], xv.y, acc[6][1]);
        acc[6][2] = fmaf(wv[6], xv.z, acc[6][2]);
        acc[6][3] = fmaf(wv[6], xv.w, acc[6][3]);
        acc[7][0] = fmaf(wv[7], xv.x, acc[7][0]);
        acc[7][1] = fmaf(wv[7], xv.y, acc[7][1]);
        acc[7][2] = fmaf(wv[7], xv.z, acc[7][2]);
        acc[7][3] = fmaf(wv[7], xv.w, acc[7][3]);
    }
}

__device__ __forceinline__ void gemm2d_store(float (&acc)[GB10_TM][GB10_TNREG],
                                             float* __restrict__ y, int nbase, int t0, int N,
                                             int T, int ty, int tx) {
#pragma unroll
    for (int i = 0; i < GB10_TM; ++i) {
        const int n = nbase + ty * GB10_TM + i;
        if (n < N) {
#pragma unroll
            for (int j = 0; j < GB10_TNREG; ++j) {
                const int t = t0 + tx * GB10_TNREG + j;
                // With a K split, each block holds a PARTIAL sum for this element.
                // block 0 writes, blocks > 0 accumulate -- which is why the host
                // must zero y before launching grid.z > 1.
                if (t < T) {
                    if (blockIdx.z == 0) y[(size_t)t * N + n] = acc[i][j];
                    else atomicAdd(&y[(size_t)t * N + n], acc[i][j]);
                }
            }
        }
    }
}

// `gemm2d_store` with the per-tensor weight scale applied once, rather than
// folding it into every staged weight.
__device__ __forceinline__ void gemm2d_store_scaled(float (&acc)[GB10_TM][GB10_TNREG],
                                                    float* __restrict__ y, int nbase, int t0,
                                                    int N, int T, int ty, int tx, float s2) {
#pragma unroll
    for (int i = 0; i < GB10_TM; ++i)
#pragma unroll
        for (int j = 0; j < GB10_TNREG; ++j) acc[i][j] *= s2;
    gemm2d_store(acc, y, nbase, t0, N, T, ty, tx);
}

__device__ __forceinline__ void gemm2d_begin(float (&acc)[GB10_TM][GB10_TNREG]) {
#pragma unroll
    for (int i = 0; i < GB10_TM; ++i)
#pragma unroll
        for (int j = 0; j < GB10_TNREG; ++j) acc[i][j] = 0.0f;
}

__device__ __forceinline__ void gemm2d_ids(int& ty, int& tx) {
    ty = threadIdx.x >> 4;  // 8 groups of 8 rows = 64
    tx = threadIdx.x & 15;  // 16 groups of 4 tokens = 64
}

template <int DUMMY>
__device__ __forceinline__ void nvfp4_gemm_body(const uint8_t* __restrict__ w,
                                                const uint8_t* __restrict__ sc,
                                                const float* __restrict__ s2,
                                                const float* __restrict__ x,
                                                float* __restrict__ y, int N, int K, int T) {
    // Double buffered: staging chunk c+1 while computing chunk c hides the
    // staging load latency, which is what this kernel is actually bound by.
    // bf16 weight tile: the E2M1 x E4M3 product has at most 4 significant
    // bits, so bf16 (8) stores it exactly and the per-tensor `wscale2` can be
    // folded into the accumulator once at the end instead. Halving this tile
    // is what lifts occupancy from 2 to 3 blocks/SM.
    __shared__ uint16_t wt[2][GB10_KC][GB10_WSTRIDE];
    __shared__ float xt[2][GB10_KC][GB10_XSTRIDE];

    const int nbase = blockIdx.x * GB10_TN;
    const int t0 = blockIdx.y * GB10_TT;
    int ty, tx;
    gemm2d_ids(ty, tx);

    float acc[GB10_TM][GB10_TNREG];
    gemm2d_begin(acc);

    const int nchunk = K / GB10_KC;
    // This block's share of the K chunks. With grid.z == 1 (the host default)
    // kc0 == 0 and kc1 == nchunk, so this is exactly the old loop.
    // MUST derive from gridDim.z, not from GB10_KSPLIT. Using the constant made a
    // grid.z == 1 launch cover only half of K, which the gate caught as 0/3 before
    // any timing was taken -- the exact "a broken kernel reports a fast number"
    // failure this project keeps hitting.
    const int nsplit = (int)gridDim.z;
    const int kc_half = (nchunk + nsplit - 1) / nsplit;
    const int kc0 = blockIdx.z * kc_half;
    const int kc1 = min(kc0 + kc_half, nchunk);
    stage_wtile<GB10_KC>(wt[1], w, sc, s2, K, nbase, N, kc0);
    stage_xtile<GB10_KC>(xt[1], x, K, T, t0, kc0);
    __syncthreads();
    // kc0 is even by construction (see GB10_KSPLIT above), so the existing parity
    // expression stays correct with no change and no `rel` variable.
    for (int c = kc0; c < kc1; ++c) {
        const int cur = (c ^ 1) & 1, nxt = c & 1;
        if (c + 1 < kc1) {
            stage_wtile<GB10_KC>(wt[nxt], w, sc, s2, K, nbase, N, c + 1);
            stage_xtile<GB10_KC>(xt[nxt], x, K, T, t0, c + 1);
        }
        gemm2d_outer_bf16(wt[cur], xt[cur], acc, ty, tx);
        __syncthreads();
    }

    gemm2d_store_scaled(acc, y, nbase, t0, N, T, ty, tx, __ldg(s2));
}

__device__ __forceinline__ void fp8_gemm_body(const uint8_t* __restrict__ w,
                                              const float* __restrict__ s1,
                                              const float* __restrict__ x,
                                              float* __restrict__ y, int N, int K, int T) {
    // Double buffered: staging chunk c+1 while computing chunk c hides the
    // staging load latency, which is what this kernel is actually bound by.
    __shared__ uint16_t wt[2][GB10_KC][GB10_WSTRIDE];
    __shared__ float xt[2][GB10_KC][GB10_XSTRIDE];

    const int nbase = blockIdx.x * GB10_TN;
    const int t0 = blockIdx.y * GB10_TT;
    int ty, tx;
    gemm2d_ids(ty, tx);

    float acc[GB10_TM][GB10_TNREG];
    gemm2d_begin(acc);

    const int nchunk = K / GB10_KC;
    stage_wtile_fp8<GB10_KC>(wt[1], w, s1, K, nbase, N, 0);
    stage_xtile<GB10_KC>(xt[1], x, K, T, t0, 0);
    __syncthreads();
    for (int c = 0; c < nchunk; ++c) {
        const int cur = (c ^ 1) & 1, nxt = c & 1;
        if (c + 1 < nchunk) {
            stage_wtile_fp8<GB10_KC>(wt[nxt], w, s1, K, nbase, N, c + 1);
            stage_xtile<GB10_KC>(xt[nxt], x, K, T, t0, c + 1);
        }
        gemm2d_outer_bf16(wt[cur], xt[cur], acc, ty, tx);
        __syncthreads();
    }

    gemm2d_store(acc, y, nbase, t0, N, T, ty, tx);
}

__device__ __forceinline__ void bf16_gemm_body(const uint16_t* __restrict__ w,
                                               const float* __restrict__ x,
                                               float* __restrict__ y, int N, int K, int T) {
    // Double buffered: staging chunk c+1 while computing chunk c hides the
    // staging load latency, which is what this kernel is actually bound by.
    __shared__ uint16_t wt[2][GB10_KC][GB10_WSTRIDE];
    __shared__ float xt[2][GB10_KC][GB10_XSTRIDE];

    const int nbase = blockIdx.x * GB10_TN;
    const int t0 = blockIdx.y * GB10_TT;
    int ty, tx;
    gemm2d_ids(ty, tx);

    float acc[GB10_TM][GB10_TNREG];
    gemm2d_begin(acc);

    const int nchunk = K / GB10_KC;
    stage_wtile_bf16<GB10_KC>(wt[1], w, K, nbase, N, 0);
    stage_xtile<GB10_KC>(xt[1], x, K, T, t0, 0);
    __syncthreads();
    for (int c = 0; c < nchunk; ++c) {
        const int cur = (c ^ 1) & 1, nxt = c & 1;
        if (c + 1 < nchunk) {
            stage_wtile_bf16<GB10_KC>(wt[nxt], w, K, nbase, N, c + 1);
            stage_xtile<GB10_KC>(xt[nxt], x, K, T, t0, c + 1);
        }
        gemm2d_outer_bf16(wt[cur], xt[cur], acc, ty, tx);
        __syncthreads();
    }

    gemm2d_store(acc, y, nbase, t0, N, T, ty, tx);
}

// NVRTC looks kernels up by symbol name, and a template instantiation mangles
// (`_Z16nvfp4_gemm_kernelILi8EEv...`), so each entry point is a plain
// `extern "C" __global__` wrapper over the templated device body.
extern "C" __global__ void __launch_bounds__(GB10_GEMM_BLOCK) nvfp4_gemm_kernel(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
    const float* __restrict__ s2, const float* __restrict__ x, float* __restrict__ y, int N,
    int K, int T) {
    nvfp4_gemm_body<0>(w, sc, s2, x, y, N, K, T);
}

extern "C" __global__ void __launch_bounds__(GB10_GEMM_BLOCK) fp8_gemm_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ s1, const float* __restrict__ x,
    float* __restrict__ y, int N, int K, int T) {
    fp8_gemm_body(w, s1, x, y, N, K, T);
}

extern "C" __global__ void __launch_bounds__(GB10_GEMM_BLOCK) bf16_gemm_kernel(
    const uint16_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int N,
    int K, int T) {
    bf16_gemm_body(w, x, y, N, K, T);
}

// ---- NVFP4 -> bf16 whole-matrix dequantise -------------------------------
//
// Feeds the cuBLAS bf16 prefill path: the tensor core GEMM needs both operands
// in bf16, while the weights on disk are NVFP4. The nibble order, the group-16
// scale row and the E4M3 scale decode are all copied from `stage_wtile` above,
// which is the authoritative on-device layout.
//
// `s2` (the per-tensor scale) is deliberately NOT applied here. The existing
// NVFP4 GEMM applies it as a post-scale on the fp32 accumulator
// (`gemm2d_store_scaled`), so folding it into the bf16 weights would round each
// weight after scaling instead of scaling the fp32 sum -- a different number.
// The caller scales the GEMM output instead.
extern "C" __global__ void dequant_nvfp4_to_bf16_kernel(const uint8_t* __restrict__ w,
                                             const uint8_t* __restrict__ sc,
                                             __nv_bfloat16* __restrict__ out,
                                             int N, int K) {
    const size_t total = (size_t)N * (size_t)K;
    const int kk = K;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int n = (int)(idx / (size_t)kk);
        const int k = (int)(idx % (size_t)kk);
        const uint8_t byte = __ldg(w + (size_t)n * (size_t)(kk >> 1) + (size_t)(k >> 1));
        const uint8_t nib = (k & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
        const float s = e4m3_to_float(__ldg(sc + (size_t)n * (size_t)(kk >> 4) + (size_t)(k >> 4)));
        out[idx] = __float2bfloat16_rn(e2m1_to_float(nib) * s);
    }
}

// ---- FP8 (E4M3) -> bf16 whole-matrix dequantise --------------------------
//
// Unlike the NVFP4 path there is no per-tensor post-scale to defer: fp8 carries
// a single per-tensor scale which `stage_wtile_fp8` applies *in staging*
// (`wscale = __ldg(s1)`), so the same scale belongs in this kernel.
extern "C" __global__ void dequant_fp8_to_bf16_kernel(const uint8_t* __restrict__ w,
                                                      const float* __restrict__ s1,
                                                      __nv_bfloat16* __restrict__ out,
                                                      int N, int K) {
    const size_t total = (size_t)N * (size_t)K;
    const float wscale = __ldg(s1);
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x)
        out[idx] = __float2bfloat16_rn(e4m3_to_float(__ldg(w + idx)) * wscale);
}

// ---- fp32 -> bf16 activation cast ----------------------------------------
//
// The tensor cores need bf16 activations. `generate` stays 16/16 token-exact
// with this rounding applied (see the -DGB10_SIM_BF16_ACT harness in
// gemm2d_outer_bf16), which is what cleared this path to proceed.
extern "C" __global__ void f32_to_bf16_kernel(const float* __restrict__ x,
                                              __nv_bfloat16* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(__ldg(x + i));
}

// ---- bf16 -> fp32 epilogue, with the optional per-tensor nvfp4 scale -------
//
// cuBLAS hands back bf16 (cudarc's safe Gemm fixes A, B and C all to
// CUDA_R_16BF) while the rest of the layer works in fp32, and the nvfp4 path
// still owes its `s2`. Both happen here. `has_scale` is an int rather than a
// nullable pointer so the caller can always pass a valid (possibly dummy) slice.
extern "C" __global__ void bf16_to_f32_scaled_kernel(const __nv_bfloat16* __restrict__ x,
                                                     float* __restrict__ out,
                                                     const float* __restrict__ s2,
                                                     int has_scale, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __bfloat162float(__ldg(x + i));
        if (has_scale) v *= __ldg(s2);
        out[i] = v;
    }
}

// ---- bf16 weights held as u16 -> bf16 tensor-core operand -----------------
//
// `LinearData::Bf16` stores weights as `CudaSlice<u16>`; cuBLAS needs
// `CudaSlice<half::bf16>`. Same bits, different Rust type.
extern "C" __global__ void u16_to_bf16_kernel(const uint16_t* __restrict__ x,
                                              __nv_bfloat16* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __ushort_as_bfloat16(__ldg(x + i));
}

// In-place scale of an fp32 buffer by a single scalar.
//
// The tensor-core prefill GEMM now returns fp32 straight from cuBLAS so that the
// accumulator is not rounded to bf16 on the way out (see
// `cublas_gemm_bf16_f32`). That removed the bf16 -> fp32 epilogue, but the NVFP4
// path still owes its per-tensor `s2`, so this applies just that. `n` is the
// element count and `s` is one value, matching the `__ldg(s2)` scalar of the
// epilogue it replaces. In-place is safe: each thread reads and writes its own
// index with no cross-thread dependence.
extern "C" __global__ void f32_scale_kernel(float* __restrict__ x,
                                            const float* __restrict__ s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * __ldg(s);
}

// fp32 -> fp16 for the tensor-core activation operand.
//
// bf16 carries 8 mantissa bits, and casting the activations to it was enough to
// flip the greedy argmax at long context (see `forward_prefill_tensor_core`).
// fp16 carries 10, at the same tensor-core throughput, and it is lossless for
// the 4-bit NVFP4 / FP8 weights, which stay in bf16 -- `cublasGemmEx` takes
// Atype and Btype independently, so the pair is mixed rather than converted.
extern "C" __global__ void f32_to_f16_kernel(const float* __restrict__ x,
                                             __half* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(__ldg(x + i));
}

// ---- fp16 twins of the operand staging kernels ---------------------------
//
// The tensor-core prefill GEMM runs entirely in fp16 rather than bf16. bf16's 8
// mantissa bits were not enough: `generate` 16/16 was token-exact with bf16
// activations, but at 970+ prompt tokens the greedy argmax flipped to EOS and
// the model answered nothing (5/5 -> 0/5 on a fixed long-prompt battery, and
// llama.cpp answers the same prompts fine). fp16 has 10 mantissa bits at the
// same tensor-core throughput, and it is still lossless for these weights --
// NVFP4 carries 4 bits and FP8 E4M3 carries 4, both well inside fp16's 10.
extern "C" __global__ void dequant_nvfp4_to_f16_kernel(const uint8_t* __restrict__ w,
                                                       const uint8_t* __restrict__ sc,
                                                       __half* __restrict__ out,
                                                       int N, int K) {
    const size_t total = (size_t)N * (size_t)K;
    const int kk = K;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int n = (int)(idx / (size_t)kk);
        const int k = (int)(idx % (size_t)kk);
        const uint8_t byte = __ldg(w + (size_t)n * (size_t)(kk >> 1) + (size_t)(k >> 1));
        const uint8_t nib = (k & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
        const float s = e4m3_to_float(__ldg(sc + (size_t)n * (size_t)(kk >> 4) + (size_t)(k >> 4)));
        out[idx] = __float2half_rn(e2m1_to_float(nib) * s);
    }
}

extern "C" __global__ void dequant_fp8_to_f16_kernel(const uint8_t* __restrict__ w,
                                                     const float* __restrict__ s1,
                                                     __half* __restrict__ out,
                                                     int N, int K) {
    const size_t total = (size_t)N * (size_t)K;
    const float wscale = __ldg(s1);
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x)
        out[idx] = __float2half_rn(e4m3_to_float(__ldg(w + idx)) * wscale);
}

extern "C" __global__ void u16_to_f16_kernel(const uint16_t* __restrict__ x,
                                             __half* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __ushort_as_half(__ldg(x + i));
}

// ---- fp32 -> bf16 hi/lo split (split-precision activation) ---------------
//
// bf16 operands broke this model (8 mantissa bits; see
// `forward_prefill_tensor_core`) and fp16 operands (10 bits) still broke it, so
// the activation needs more precision than a single 16-bit format carries.
//
// Splitting restores it: `hi` is the bf16 rounding of x and `lo` is the bf16
// rounding of the residual, so `hi + lo` carries ~16 mantissa bits -- six more
// than fp16, and the residual is exact enough that the pair is close to fp32 for
// this purpose. The GEMM is then run twice, W(hi) + W(lo), accumulating into the
// same fp32 output, so the weights stay a single bf16 operand (they are 4-bit
// NVFP4 / FP8 and therefore already lossless in bf16) and only the activation is
// doubled.
//
// This costs one extra GEMM on the tensor cores instead of the ~11x slower fp32
// CUDA-core GEMM, which is the point: it buys the precision back without giving
// back the speed.
extern "C" __global__ void f32_split_bf16_kernel(const float* __restrict__ x,
                                                 __nv_bfloat16* __restrict__ hi,
                                                 __nv_bfloat16* __restrict__ lo,
                                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = __ldg(x + i);
        const __nv_bfloat16 h = __float2bfloat16_rn(v);
        hi[i] = h;
        lo[i] = __float2bfloat16_rn(v - __bfloat162float(h));
    }
}
