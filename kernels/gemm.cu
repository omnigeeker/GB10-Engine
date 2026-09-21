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
#define GB10_TT 32    // prompt tokens per block
#define GB10_KC 64    // k values per chunk
#define GB10_TM 4     // rows of N owned by one thread
#define GB10_TNREG 4  // tokens owned by one thread
#define GB10_GEMM_BLOCK 128

// Padded strides. The pad must keep each row 16-byte aligned (a multiple of 4
// floats) so the inner loop can read a whole 4-wide sub-tile with one LDS.128
// instead of four LDS.32; the extra float beyond that keeps consecutive `k`
// slices off the same shared bank.
#define GB10_WSTRIDE (GB10_TN + 4)
#define GB10_XSTRIDE (GB10_TT + 4)

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

// Same 4x4 outer product, reading the bf16 weight tile.
__device__ __forceinline__ void gemm2d_outer_bf16(const uint16_t (*wt)[GB10_WSTRIDE],
                                                  const float (*xt)[GB10_XSTRIDE],
                                                  float (&acc)[GB10_TM][GB10_TNREG], int ty,
                                                  int tx) {
    static_assert(GB10_TM == 4 && GB10_TNREG == 4, "bf16 path assumes 4x4 tiles");
#pragma unroll
    for (int k = 0; k < GB10_KC; ++k) {
        const uint2 wq = *reinterpret_cast<const uint2*>(&wt[k][ty * GB10_TM]);
        const float w0 = bf16_to_float((uint16_t)(wq.x & 0xFFFF));
        const float w1 = bf16_to_float((uint16_t)(wq.x >> 16));
        const float w2 = bf16_to_float((uint16_t)(wq.y & 0xFFFF));
        const float w3 = bf16_to_float((uint16_t)(wq.y >> 16));
        const float4 xv = *reinterpret_cast<const float4*>(&xt[k][tx * GB10_TNREG]);
        acc[0][0] = fmaf(w0, xv.x, acc[0][0]);
        acc[0][1] = fmaf(w0, xv.y, acc[0][1]);
        acc[0][2] = fmaf(w0, xv.z, acc[0][2]);
        acc[0][3] = fmaf(w0, xv.w, acc[0][3]);
        acc[1][0] = fmaf(w1, xv.x, acc[1][0]);
        acc[1][1] = fmaf(w1, xv.y, acc[1][1]);
        acc[1][2] = fmaf(w1, xv.z, acc[1][2]);
        acc[1][3] = fmaf(w1, xv.w, acc[1][3]);
        acc[2][0] = fmaf(w2, xv.x, acc[2][0]);
        acc[2][1] = fmaf(w2, xv.y, acc[2][1]);
        acc[2][2] = fmaf(w2, xv.z, acc[2][2]);
        acc[2][3] = fmaf(w2, xv.w, acc[2][3]);
        acc[3][0] = fmaf(w3, xv.x, acc[3][0]);
        acc[3][1] = fmaf(w3, xv.y, acc[3][1]);
        acc[3][2] = fmaf(w3, xv.z, acc[3][2]);
        acc[3][3] = fmaf(w3, xv.w, acc[3][3]);
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
    ty = threadIdx.x >> 3;  // 16 groups of 4 rows = 64
    tx = threadIdx.x & 7;   // 8 groups of 4 tokens = 32
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
    stage_wtile<GB10_KC>(wt[1], w, sc, s2, K, nbase, N, 0);
    stage_xtile<GB10_KC>(xt[1], x, K, T, t0, 0);
    __syncthreads();
    for (int c = 0; c < nchunk; ++c) {
        const int cur = (c ^ 1) & 1, nxt = c & 1;
        if (c + 1 < nchunk) {
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
