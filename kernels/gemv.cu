// GB10-Engine: quantized GEMV kernels for the decode path.
//
// Decode (batch 1) is a matrix-vector product whose weight operand is read
// from LPDDR5X exactly once per token, so these kernels are written to make
// the *weight* stream perfectly coalesced.
//
// Design notes
// ------------
// GB10 caps opt-in dynamic shared memory at 99 KB per block and 100 KB per SM
// (measured), so staging the activation vector in shared memory would cap
// occupancy at one block per SM. Instead the activation tile is held in
// *registers* and hoisted out of the row loop: a warp owns ROWS output rows
// and walks k in tiles, loading each 16-element activation tile once and
// reusing it for all ROWS rows. That removes shared memory entirely and
// amortises activation traffic by ROWS.
//
// Per lane, per k-tile:
//   NVFP4 : 8 weight bytes (uint2) + 1 E4M3 group scale  -> 16 elements
//   FP8   : 16 weight bytes (uint4)                      -> 16 elements
//   bf16  : 32 weight bytes (2x uint4)                   -> 16 elements
// Across a warp that is 256 B / 512 B / 1024 B of weight traffic per tile,
// i.e. exactly the format's bytes-per-element with no waste.
//
// Layouts, verified against nv-community/Qwen3.8-27B-NVFP4:
//   NVFP4  weight         U8    [N, K/2]   two E2M1 nibbles per byte
//          weight_scale   E4M3  [N, K/16]  one scale per 16-element group
//          weight_scale_2 F32   []         per-tensor global scale
//   FP8    weight         E4M3  [N, K]
//          weight_scale   F32   []         per-tensor scale
//
// K should be a multiple of 512 for the vectorised path; a scalar fallback
// handles the remainder so the kernels stay correct for any K.

#include "gemv_common.cuh"

using namespace gb10;

namespace {

constexpr int kWarp = 32;
constexpr int kVec = 16;             // elements per lane per k-tile
constexpr int kTile = kWarp * kVec;  // 512 elements per warp k-tile

// Activation tile hoisted into registers: 4 x float4.
struct XVec {
    float4 a0, a1, a2, a3;
};

__device__ __forceinline__ XVec load_x(const float* __restrict__ xb, int e0) {
    const float4* p = reinterpret_cast<const float4*>(xb + e0);
    XVec v;
    v.a0 = __ldg(p + 0);
    v.a1 = __ldg(p + 1);
    v.a2 = __ldg(p + 2);
    v.a3 = __ldg(p + 3);
    return v;
}

__device__ __forceinline__ float dot16(const float* w, const XVec& v) {
    float t = w[0] * v.a0.x + w[1] * v.a0.y + w[2] * v.a0.z + w[3] * v.a0.w;
    t = fmaf(w[4], v.a1.x, t);
    t = fmaf(w[5], v.a1.y, t);
    t = fmaf(w[6], v.a1.z, t);
    t = fmaf(w[7], v.a1.w, t);
    t = fmaf(w[8], v.a2.x, t);
    t = fmaf(w[9], v.a2.y, t);
    t = fmaf(w[10], v.a2.z, t);
    t = fmaf(w[11], v.a2.w, t);
    t = fmaf(w[12], v.a3.x, t);
    t = fmaf(w[13], v.a3.y, t);
    t = fmaf(w[14], v.a3.z, t);
    t = fmaf(w[15], v.a3.w, t);
    return t;
}

}  // namespace

// ---------------------------------------------------------------------------
// NVFP4 GEMV: y[b,n] = scale2 * sum_k dequant(W[n,k]) * x[b,k]
// ---------------------------------------------------------------------------
template <int ROWS>
__device__ __forceinline__ void nvfp4_gemv_tmpl(
    const float* __restrict__ x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ wscale,
    const float* __restrict__ scale2,
    float* __restrict__ y,
    int N, int K) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int b = blockIdx.y;

    const float* __restrict__ xb = x + (size_t)b * K;
    const int rowbytes = K >> 1;
    const int scalerow = K >> 4;
    const int full_tiles = K / kTile;
    const int row_stride = gridDim.x * nwarps * ROWS;

    for (int rbase = (blockIdx.x * nwarps + warp) * ROWS; rbase < N;
         rbase += row_stride) {
        float acc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

        // ---- vectorised body: full 512-element k-tiles ----
        for (int i = 0; i < full_tiles; ++i) {
            const int e0 = i * kTile + lane * kVec;
            const XVec xv = load_x(xb, e0);
            const int so = i * kWarp + lane;

            // Issue every row's weight and scale load BEFORE any of the
            // arithmetic. Fusing the loads into the compute loop leaves the
            // compiler free to keep only one load in flight per warp, which
            // starves DRAM; separating them gives ROWS independent loads.
            uint2 pk[ROWS];
            float sc[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = rbase + r;
                if (row < N) {
                    const uint8_t* __restrict__ wr = w + (size_t)row * rowbytes;
                    pk[r] = *reinterpret_cast<const uint2*>(wr + (e0 >> 1));
                    sc[r] = e4m3_to_float(wscale[(size_t)row * scalerow + so]);
                }
            }

#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = rbase + r;
                if (row < N) {
                    const uint2 packed = pk[r];
                    const float s = sc[r];

                    float lo[8], hi[8];
                    e2m1x8_to_float(packed.x, lo);
                    e2m1x8_to_float(packed.y, hi);

                    float t = lo[0] * xv.a0.x + lo[1] * xv.a0.y + lo[2] * xv.a0.z +
                              lo[3] * xv.a0.w;
                    t = fmaf(lo[4], xv.a1.x, t);
                    t = fmaf(lo[5], xv.a1.y, t);
                    t = fmaf(lo[6], xv.a1.z, t);
                    t = fmaf(lo[7], xv.a1.w, t);
                    t = fmaf(hi[0], xv.a2.x, t);
                    t = fmaf(hi[1], xv.a2.y, t);
                    t = fmaf(hi[2], xv.a2.z, t);
                    t = fmaf(hi[3], xv.a2.w, t);
                    t = fmaf(hi[4], xv.a3.x, t);
                    t = fmaf(hi[5], xv.a3.y, t);
                    t = fmaf(hi[6], xv.a3.z, t);
                    t = fmaf(hi[7], xv.a3.w, t);

                    acc[r] = fmaf(t, s, acc[r]);
                }
            }
        }

        // ---- scalar tail: K % 512 != 0 ----
        for (int e = full_tiles * kTile + lane; e < K; e += kWarp) {
            const float xv = __ldg(xb + e);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = rbase + r;
                if (row < N) {
                    const uint8_t byte = w[(size_t)row * rowbytes + (e >> 1)];
                    const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0xF);
                    const float sc =
                        e4m3_to_float(wscale[(size_t)row * scalerow + (e >> 4)]);
                    acc[r] = fmaf(e2m1_to_float(nib) * sc, xv, acc[r]);
                }
            }
        }

        const float s2 = __ldg(scale2);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = rbase + r;
            if (row < N) {
                const float a = warp_reduce_sum(acc[r]);
                if (lane == 0) y[(size_t)b * N + row] = a * s2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 (E4M3) GEMV: y[b,n] = wscale * sum_k W[n,k] * x[b,k]
// ---------------------------------------------------------------------------
template <int ROWS>
__device__ __forceinline__ void fp8_gemv_tmpl(
    const float* __restrict__ x,
    const uint8_t* __restrict__ w,
    const float* __restrict__ wscale,
    float* __restrict__ y,
    int N, int K) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int b = blockIdx.y;

    const float* __restrict__ xb = x + (size_t)b * K;
    const int full_tiles = K / kTile;
    const int row_stride = gridDim.x * nwarps * ROWS;

    for (int rbase = (blockIdx.x * nwarps + warp) * ROWS; rbase < N;
         rbase += row_stride) {
        float acc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

        for (int i = 0; i < full_tiles; ++i) {
            const int e0 = i * kTile + lane * kVec;
            const XVec xv = load_x(xb, e0);

#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = rbase + r;
                if (row < N) {
                    const uint4 packed =
                        *reinterpret_cast<const uint4*>(w + (size_t)row * K + e0);
                    const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed);
                    float wv[kVec];
#pragma unroll
                    for (int j = 0; j < kVec; ++j) wv[j] = e4m3_to_float(pb[j]);
                    acc[r] += dot16(wv, xv);
                }
            }
        }

        for (int e = full_tiles * kTile + lane; e < K; e += kWarp) {
            const float xv = __ldg(xb + e);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = rbase + r;
                if (row < N) {
                    acc[r] = fmaf(e4m3_to_float(w[(size_t)row * K + e]), xv, acc[r]);
                }
            }
        }

        const float s = __ldg(wscale);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = rbase + r;
            if (row < N) {
                const float a = warp_reduce_sum(acc[r]);
                if (lane == 0) y[(size_t)b * N + row] = a * s;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bf16 GEMV (MTP projections; embeddings use a gather)
// ---------------------------------------------------------------------------
template <int ROWS>
__device__ __forceinline__ void bf16_gemv_tmpl(
    const float* __restrict__ x,
    const uint16_t* __restrict__ w,
    float* __restrict__ y,
    int N, int K) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int b = blockIdx.y;

    const float* __restrict__ xb = x + (size_t)b * K;
    const int full_tiles = K / kTile;
    const int row_stride = gridDim.x * nwarps * ROWS;

    for (int rbase = (blockIdx.x * nwarps + warp) * ROWS; rbase < N;
         rbase += row_stride) {
        float acc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

        for (int i = 0; i < full_tiles; ++i) {
            const int e0 = i * kTile + lane * kVec;
            const XVec xv = load_x(xb, e0);

#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = rbase + r;
                if (row < N) {
                    const uint16_t* wr = w + (size_t)row * K + e0;
                    float wv[kVec];
#pragma unroll
                    for (int j = 0; j < kVec; ++j)
                        wv[j] = bf16_to_float(__ldg(wr + j));
                    acc[r] += dot16(wv, xv);
                }
            }
        }

        for (int e = full_tiles * kTile + lane; e < K; e += kWarp) {
            const float xv = __ldg(xb + e);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = rbase + r;
                if (row < N) {
                    acc[r] =
                        fmaf(bf16_to_float(__ldg(w + (size_t)row * K + e)), xv, acc[r]);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = rbase + r;
            if (row < N) {
                const float a = warp_reduce_sum(acc[r]);
                if (lane == 0) y[(size_t)b * N + row] = a;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Exported entry points (row tiling chosen on the host)
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
nvfp4_gemv_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w,
                  const uint8_t* __restrict__ wscale, const float* __restrict__ scale2,
                  float* __restrict__ y, int N, int K) {
    nvfp4_gemv_tmpl<4>(x, w, wscale, scale2, y, N, K);
}

extern "C" __global__ void __launch_bounds__(256)
fp8_gemv_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w,
                const float* __restrict__ wscale, float* __restrict__ y, int N, int K) {
    fp8_gemv_tmpl<4>(x, w, wscale, y, N, K);
}

extern "C" __global__ void __launch_bounds__(256)
bf16_gemv_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                 float* __restrict__ y, int N, int K) {
    bf16_gemv_tmpl<4>(x, w, y, N, K);
}
