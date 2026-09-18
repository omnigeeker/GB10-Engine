// Shared device helpers for GB10-Engine kernels.
//
// Target: NVIDIA GB10 (Blackwell), sm_121, 48 SMs, ~200 GB/s usable LPDDR5X
// read bandwidth. Decode is bandwidth-bound: every weight byte is read once
// per token, so kernels are written to make weight reads perfectly coalesced
// and to keep activation traffic in shared memory.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace gb10 {

// ---------------------------------------------------------------------------
// FP4 E2M1 decode
// ---------------------------------------------------------------------------
// 1 sign, 2 exponent, 1 mantissa. Magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6.
//
// Decoded arithmetically rather than via a table: the decode is pure ALU work
// and this kernel is bandwidth-bound at 0.5 byte/element, so spare FLOPs are
// free, while a table would add a memory operand to the critical path.
__device__ __forceinline__ float e2m1_to_float(uint8_t nibble) {
    const uint32_t e = (nibble >> 1) & 0x3u;  // exponent field
    const uint32_t m = nibble & 0x1u;         // mantissa bit
    float v;
    if (e == 0) {
        v = m ? 0.5f : 0.0f;                  // subnormal
    } else {
        v = (m ? 1.5f : 1.0f) * exp2f((float)e - 1.0f);
    }
    return (nibble & 0x8u) ? -v : v;
}

// Decode 8 nibbles packed in a uint32 into 8 floats, low nibble first.
__device__ __forceinline__ void e2m1x8_to_float(uint32_t packed, float* out) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = e2m1_to_float((uint8_t)((packed >> (4 * i)) & 0xF));
    }
}

// ---------------------------------------------------------------------------
// FP8 E4M3 decode
// ---------------------------------------------------------------------------
// 1 sign, 4 exponent (bias 7), 3 mantissa. Subnormals at exponent 0.
__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
    uint32_t sign = (uint32_t)(b >> 7) << 31;
    uint32_t exp = (b >> 3) & 0xF;
    uint32_t man = b & 0x7;
    float v;
    if (exp == 0) {
        // subnormal: man/8 * 2^-6
        v = (float)man * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else if (exp == 0xF && man == 0x7) {
        v = __int_as_float(0x7FC00000u);  // NaN
    } else {
        v = __int_as_float(((exp - 7 + 127) << 23) | (man << 20));
    }
    return __int_as_float(__float_as_int(v) | (int)sign);
}

// ---------------------------------------------------------------------------
// bf16 decode
// ---------------------------------------------------------------------------
// bf16 is the upper 16 bits of an fp32, so the widening is exact and free:
// no arithmetic, no table, just a shift. (This is why `__ushort_as_half` must
// NOT be used here — that would reinterpret the bits as IEEE fp16.)
__device__ __forceinline__ float bf16_to_float(uint16_t h) {
    return __uint_as_float(((uint32_t)h) << 16);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Vector types
// ---------------------------------------------------------------------------
struct __align__(8) uint2_t {
    uint32_t x, y;
};

}  // namespace gb10
