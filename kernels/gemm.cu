// Batched-prefill GEMM kernels.
//
// The decode GEMV and the prefill GEMM want opposite tilings. Decode reads
// each weight exactly once and produces one output, so it blocks over N and
// streams W -- it is purely DRAM-bound. Prefill reuses every weight across all
// T prompt tokens, so it must block over (N, T) and hold a T-wide accumulator;
// reusing the GEMV with `grid.y = T` instead re-reads the whole weight matrix
// T times and buys nothing but launch overhead.
//
// Layout per block (256 threads = 8 warps):
//   * each warp owns one output row `n`
//   * each block covers TILE_T prompt tokens
//   * the K loop is chunked; the x tile for the chunk is staged in shared
//     memory and broadcast across the block's warps
//   * lanes cover k with stride 1, so the weight read is coalesced and the
//     shared-memory read is conflict-free
//   * the T-wide partials are reduced across the warp once, at the very end
//
// x traffic is `(N / 8) * T * K * 4` bytes but is served from L1/L2 because
// every warp in the block shares the same tile.

#include "gemv_common.cuh"

using namespace gb10;

// Prompt tokens covered by one block. 8 keeps the T-wide accumulator in 8
// registers while still splitting a 59-token prompt across 8 blocks, which is
// what keeps the SMs fed on short prompts.
#define GB10_TILE_T 8
#define GB10_GEMM_WARPS 8
#define GB10_GEMM_BLOCK (GB10_GEMM_WARPS * 32)

// Stage the `[TILE_T, KC]` x chunk into shared memory. Every warp in the block
// reads the same tile, so this is the only place x touches global memory.
template <int TILE_T, int KC>
__device__ __forceinline__ void gemm_stage_x(float (*xs)[KC], const float* __restrict__ x, int K,
                                             int T, int t0, int c) {
    for (int i = threadIdx.x; i < TILE_T * KC; i += GB10_GEMM_BLOCK) {
        const int tt = i / KC, kk = i % KC;
        const int gt = t0 + tt;
        xs[tt][kk] = (gt < T) ? __ldg(x + (size_t)gt * K + c * KC + kk) : 0.0f;
    }
    __syncthreads();
}

// Reduce the T-wide per-lane partials and write one row per warp.
template <int TILE_T>
__device__ __forceinline__ void gemm_store(float* __restrict__ y, float (&acc)[TILE_T], int n,
                                           int N, int t0, int T, int lane) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int i = 0; i < TILE_T; ++i)
            acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off);

    if (lane == 0 && n < N) {
#pragma unroll
        for (int i = 0; i < TILE_T; ++i) {
            const int tt = t0 + i;
            if (tt < T) y[(size_t)tt * N + n] = acc[i];
        }
    }
}

template <int TILE_T>
__device__ __forceinline__ void nvfp4_gemm_body(const uint8_t* __restrict__ w,
                                                const uint8_t* __restrict__ sc,
                                                const float* __restrict__ s2,
                                                const float* __restrict__ x,
                                                float* __restrict__ y, int N, int K, int T) {
    constexpr int KC = 64;  // 32 packed bytes per chunk
    __shared__ float xs[TILE_T][KC];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * GB10_GEMM_WARPS + warp;
    const int t0 = blockIdx.y * TILE_T;
    const int nchunk = K / KC;
    const float wscale2 = __ldg(s2);

    float acc[TILE_T];
#pragma unroll
    for (int i = 0; i < TILE_T; ++i) acc[i] = 0.0f;

    const uint8_t* __restrict__ wrow = w + (size_t)n * (K >> 1);
    const uint8_t* __restrict__ srow = sc + (size_t)n * (K >> 4);

    for (int c = 0; c < nchunk; ++c) {
        gemm_stage_x<TILE_T, KC>(xs, x, K, T, t0, c);

        if (n < N) {
            // Lane l owns packed byte `l` == k values {2l, 2l+1}; both live in
            // group l/8, so a single scale byte covers both nibbles.
            const uint8_t byte = __ldg(wrow + c * 32 + lane);
            const float s = e4m3_to_float(__ldg(srow + c * 4 + (lane >> 3))) * wscale2;
            const float w0 = e2m1_to_float(byte & 0xF) * s;
            const float w1 = e2m1_to_float(byte >> 4) * s;
            const int k = lane * 2;
#pragma unroll
            for (int i = 0; i < TILE_T; ++i) {
                acc[i] = fmaf(w0, xs[i][k], acc[i]);
                acc[i] = fmaf(w1, xs[i][k + 1], acc[i]);
            }
        }
        __syncthreads();
    }

    gemm_store<TILE_T>(y, acc, n, N, t0, T, lane);
}

template <int TILE_T>
__device__ __forceinline__ void fp8_gemm_body(const uint8_t* __restrict__ w,
                                              const float* __restrict__ s1,
                                              const float* __restrict__ x,
                                              float* __restrict__ y, int N, int K, int T) {
    constexpr int KC = 128;
    __shared__ float xs[TILE_T][KC];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * GB10_GEMM_WARPS + warp;
    const int t0 = blockIdx.y * TILE_T;
    const int nchunk = K / KC;
    const float wscale = __ldg(s1);

    float acc[TILE_T];
#pragma unroll
    for (int i = 0; i < TILE_T; ++i) acc[i] = 0.0f;

    const uint8_t* __restrict__ wrow = w + (size_t)n * K;

    for (int c = 0; c < nchunk; ++c) {
        gemm_stage_x<TILE_T, KC>(xs, x, K, T, t0, c);

        if (n < N) {
            // Stride-1 lane-to-k mapping keeps `xs[i][k]` conflict-free.
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = lane + 32 * j;
                const float wv = e4m3_to_float(__ldg(wrow + c * KC + k)) * wscale;
#pragma unroll
                for (int i = 0; i < TILE_T; ++i) acc[i] = fmaf(wv, xs[i][k], acc[i]);
            }
        }
        __syncthreads();
    }

    gemm_store<TILE_T>(y, acc, n, N, t0, T, lane);
}

template <int TILE_T>
__device__ __forceinline__ void bf16_gemm_body(const uint16_t* __restrict__ w,
                                               const float* __restrict__ x,
                                               float* __restrict__ y, int N, int K, int T) {
    constexpr int KC = 128;
    __shared__ float xs[TILE_T][KC];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * GB10_GEMM_WARPS + warp;
    const int t0 = blockIdx.y * TILE_T;
    const int nchunk = K / KC;

    float acc[TILE_T];
#pragma unroll
    for (int i = 0; i < TILE_T; ++i) acc[i] = 0.0f;

    const uint16_t* __restrict__ wrow = w + (size_t)n * K;

    for (int c = 0; c < nchunk; ++c) {
        gemm_stage_x<TILE_T, KC>(xs, x, K, T, t0, c);

        if (n < N) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = lane + 32 * j;
                const float wv = bf16_to_float(__ldg(wrow + c * KC + k));
#pragma unroll
                for (int i = 0; i < TILE_T; ++i) acc[i] = fmaf(wv, xs[i][k], acc[i]);
            }
        }
        __syncthreads();
    }

    gemm_store<TILE_T>(y, acc, n, N, t0, T, lane);
}

// NVRTC looks kernels up by symbol name, and a template instantiation mangles
// (`_Z16nvfp4_gemm_kernelILi8EEv...`), so each entry point is a plain
// `extern "C" __global__` wrapper over the templated device body.
extern "C" __global__ void __launch_bounds__(GB10_GEMM_BLOCK) nvfp4_gemm_kernel(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
    const float* __restrict__ s2, const float* __restrict__ x, float* __restrict__ y, int N,
    int K, int T) {
    nvfp4_gemm_body<GB10_TILE_T>(w, sc, s2, x, y, N, K, T);
}

extern "C" __global__ void __launch_bounds__(GB10_GEMM_BLOCK) fp8_gemm_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ s1, const float* __restrict__ x,
    float* __restrict__ y, int N, int K, int T) {
    fp8_gemm_body<GB10_TILE_T>(w, s1, x, y, N, K, T);
}

extern "C" __global__ void __launch_bounds__(GB10_GEMM_BLOCK) bf16_gemm_kernel(
    const uint16_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int N,
    int K, int T) {
    bf16_gemm_body<GB10_TILE_T>(w, x, y, N, K, T);
}
