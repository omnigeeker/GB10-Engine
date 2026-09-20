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

using namespace gb10;

#define GB10_TN 64    // rows of N per block
#define GB10_TT 64    // prompt tokens per block
#define GB10_KC 32    // k values per chunk
#define GB10_TM 4     // rows of N owned by one thread
#define GB10_TNREG 4  // tokens owned by one thread
#define GB10_GEMM_BLOCK 256

// Padded strides. The pad must keep each row 16-byte aligned (a multiple of 4
// floats) so the inner loop can read a whole 4-wide sub-tile with one LDS.128
// instead of four LDS.32; the extra float beyond that keeps consecutive `k`
// slices off the same shared bank.
#define GB10_WSTRIDE (GB10_TN + 4)
#define GB10_XSTRIDE (GB10_TT + 4)

// Stage the [TILE_N, KC] weight chunk, decoded to fp32, as `wt[k][n]`.
template <int KC>
__device__ __forceinline__ void stage_wtile(float (*wt)[GB10_WSTRIDE],
                                            const uint8_t* __restrict__ w,
                                            const uint8_t* __restrict__ sc,
                                            const float* __restrict__ s2, int K, int nbase,
                                            int N, int c) {
    static_assert(GB10_TN == 64 && GB10_KC == 32, "staging map assumes 64x32");
    // One thread per (row, 8-wide k segment). The naive element-at-a-time loop
    // issues one scale load per element, i.e. 16 redundant loads per group
    // byte -- that is 713 MB of staging traffic for a 44.6 MB matrix. Loading
    // the packed byte as a uint32 and the scale once per segment makes it 2
    // loads for 8 elements instead of 16.
    const int nl = threadIdx.x >> 2;
    const int seg = threadIdx.x & 3;
    const int n = nbase + nl;
    if (n < N) {
        const int kbase = c * KC + seg * 8;
        const uint32_t packed =
            *reinterpret_cast<const uint32_t*>(w + (size_t)n * (K >> 1) + (kbase >> 1));
        const float s =
            e4m3_to_float(__ldg(sc + (size_t)n * (K >> 4) + (kbase >> 4))) * __ldg(s2);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint8_t nib = (uint8_t)((packed >> (4 * j)) & 0xF);
            wt[seg * 8 + j][nl] = e2m1_to_float(nib) * s;
        }
    }
}

template <int KC>
__device__ __forceinline__ void stage_wtile_fp8(float (*wt)[GB10_WSTRIDE],
                                                const uint8_t* __restrict__ w,
                                                const float* __restrict__ s1, int K, int nbase,
                                                int N, int c) {
    static_assert(GB10_TN == 64 && GB10_KC == 32, "staging map assumes 64x32");
    // Same map as the NVFP4 path: one thread per (row, 8-wide k segment),
    // reading two uint32 instead of eight bytes. FP8 has no group scales, so
    // the per-tensor scale is hoisted out entirely.
    const int nl = threadIdx.x >> 2;
    const int seg = threadIdx.x & 3;
    const int n = nbase + nl;
    if (n < N) {
        const int kbase = c * KC + seg * 8;
        const uint2 pk = *reinterpret_cast<const uint2*>(w + (size_t)n * K + kbase);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&pk);
        const float wscale = __ldg(s1);
#pragma unroll
        for (int j = 0; j < 8; ++j) wt[seg * 8 + j][nl] = e4m3_to_float(pb[j]) * wscale;
    }
}

template <int KC>
__device__ __forceinline__ void stage_wtile_bf16(float (*wt)[GB10_WSTRIDE],
                                                 const uint16_t* __restrict__ w, int K, int nbase,
                                                 int N, int c) {
    for (int idx = threadIdx.x; idx < GB10_TN * KC; idx += GB10_GEMM_BLOCK) {
        const int nl = idx / KC, kl = idx % KC;
        const int n = nbase + nl;
        float v = 0.0f;
        if (n < N) v = bf16_to_float(__ldg(w + (size_t)n * K + c * KC + kl));
        wt[kl][nl] = v;
    }
}

// Stage the [TILE_T, KC] activation chunk as `xt[k][t]`.
template <int KC>
__device__ __forceinline__ void stage_xtile(float (*xt)[GB10_XSTRIDE],
                                            const float* __restrict__ x, int K, int T, int t0,
                                            int c) {
    for (int idx = threadIdx.x; idx < GB10_TT * KC; idx += GB10_GEMM_BLOCK) {
        const int tl = idx / KC, kl = idx % KC;
        const int t = t0 + tl;
        xt[kl][tl] = (t < T) ? __ldg(x + (size_t)t * K + c * KC + kl) : 0.0f;
    }
}

// The 4x4 outer product. Identical for every quantisation.
__device__ __forceinline__ void gemm2d_outer(const float (*wt)[GB10_WSTRIDE],
                                             const float (*xt)[GB10_XSTRIDE],
                                             float (&acc)[GB10_TM][GB10_TNREG], int ty, int tx) {
    static_assert(GB10_TM == 4 && GB10_TNREG == 4, "float4 path assumes 4x4 tiles");
#pragma unroll
    for (int k = 0; k < GB10_KC; ++k) {
        // One 16-byte load per operand per k instead of four 4-byte ones.
        const float4 wv = *reinterpret_cast<const float4*>(&wt[k][ty * GB10_TM]);
        const float4 xv = *reinterpret_cast<const float4*>(&xt[k][tx * GB10_TNREG]);
        acc[0][0] = fmaf(wv.x, xv.x, acc[0][0]);
        acc[0][1] = fmaf(wv.x, xv.y, acc[0][1]);
        acc[0][2] = fmaf(wv.x, xv.z, acc[0][2]);
        acc[0][3] = fmaf(wv.x, xv.w, acc[0][3]);
        acc[1][0] = fmaf(wv.y, xv.x, acc[1][0]);
        acc[1][1] = fmaf(wv.y, xv.y, acc[1][1]);
        acc[1][2] = fmaf(wv.y, xv.z, acc[1][2]);
        acc[1][3] = fmaf(wv.y, xv.w, acc[1][3]);
        acc[2][0] = fmaf(wv.z, xv.x, acc[2][0]);
        acc[2][1] = fmaf(wv.z, xv.y, acc[2][1]);
        acc[2][2] = fmaf(wv.z, xv.z, acc[2][2]);
        acc[2][3] = fmaf(wv.z, xv.w, acc[2][3]);
        acc[3][0] = fmaf(wv.w, xv.x, acc[3][0]);
        acc[3][1] = fmaf(wv.w, xv.y, acc[3][1]);
        acc[3][2] = fmaf(wv.w, xv.z, acc[3][2]);
        acc[3][3] = fmaf(wv.w, xv.w, acc[3][3]);
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
                if (t < T) y[(size_t)t * N + n] = acc[i][j];
            }
        }
    }
}

__device__ __forceinline__ void gemm2d_begin(float (&acc)[GB10_TM][GB10_TNREG]) {
#pragma unroll
    for (int i = 0; i < GB10_TM; ++i)
#pragma unroll
        for (int j = 0; j < GB10_TNREG; ++j) acc[i][j] = 0.0f;
}

__device__ __forceinline__ void gemm2d_ids(int& ty, int& tx) {
    ty = threadIdx.x >> 4;  // 16 groups of 4 rows = 64
    tx = threadIdx.x & 15;  // 16 groups of 4 tokens = 64
}

template <int DUMMY>
__device__ __forceinline__ void nvfp4_gemm_body(const uint8_t* __restrict__ w,
                                                const uint8_t* __restrict__ sc,
                                                const float* __restrict__ s2,
                                                const float* __restrict__ x,
                                                float* __restrict__ y, int N, int K, int T) {
    __shared__ float wt[GB10_KC][GB10_WSTRIDE];
    __shared__ float xt[GB10_KC][GB10_XSTRIDE];

    const int nbase = blockIdx.x * GB10_TN;
    const int t0 = blockIdx.y * GB10_TT;
    int ty, tx;
    gemm2d_ids(ty, tx);

    float acc[GB10_TM][GB10_TNREG];
    gemm2d_begin(acc);

    for (int c = 0; c < K / GB10_KC; ++c) {
        stage_wtile<GB10_KC>(wt, w, sc, s2, K, nbase, N, c);
        stage_xtile<GB10_KC>(xt, x, K, T, t0, c);
        __syncthreads();
        gemm2d_outer(wt, xt, acc, ty, tx);
        __syncthreads();
    }

    gemm2d_store(acc, y, nbase, t0, N, T, ty, tx);
}

__device__ __forceinline__ void fp8_gemm_body(const uint8_t* __restrict__ w,
                                              const float* __restrict__ s1,
                                              const float* __restrict__ x,
                                              float* __restrict__ y, int N, int K, int T) {
    __shared__ float wt[GB10_KC][GB10_WSTRIDE];
    __shared__ float xt[GB10_KC][GB10_XSTRIDE];

    const int nbase = blockIdx.x * GB10_TN;
    const int t0 = blockIdx.y * GB10_TT;
    int ty, tx;
    gemm2d_ids(ty, tx);

    float acc[GB10_TM][GB10_TNREG];
    gemm2d_begin(acc);

    for (int c = 0; c < K / GB10_KC; ++c) {
        stage_wtile_fp8<GB10_KC>(wt, w, s1, K, nbase, N, c);
        stage_xtile<GB10_KC>(xt, x, K, T, t0, c);
        __syncthreads();
        gemm2d_outer(wt, xt, acc, ty, tx);
        __syncthreads();
    }

    gemm2d_store(acc, y, nbase, t0, N, T, ty, tx);
}

__device__ __forceinline__ void bf16_gemm_body(const uint16_t* __restrict__ w,
                                               const float* __restrict__ x,
                                               float* __restrict__ y, int N, int K, int T) {
    __shared__ float wt[GB10_KC][GB10_WSTRIDE];
    __shared__ float xt[GB10_KC][GB10_XSTRIDE];

    const int nbase = blockIdx.x * GB10_TN;
    const int t0 = blockIdx.y * GB10_TT;
    int ty, tx;
    gemm2d_ids(ty, tx);

    float acc[GB10_TM][GB10_TNREG];
    gemm2d_begin(acc);

    for (int c = 0; c < K / GB10_KC; ++c) {
        stage_wtile_bf16<GB10_KC>(wt, w, K, nbase, N, c);
        stage_xtile<GB10_KC>(xt, x, K, T, t0, c);
        __syncthreads();
        gemm2d_outer(wt, xt, acc, ty, tx);
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
