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
// Decoded with integer ops into an fp32 bit pattern. The obvious arithmetic
// version -- `(m ? 1.5f : 1.0f) * exp2f(e - 1)` -- is a trap: exp2f compiles to
// MUFU.EX2, an SFU instruction with roughly a quarter of the FMA throughput, so
// the decode stops being free. A token streams ~18.4e9 NVFP4 weights, i.e. one
// exp2f each, which costs more than the memory traffic it was supposed to hide.
//
// Magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6, which is exactly
//   e == 0 : subnormal, 0 or 0.5
//   e >= 1 : (1 + m/2) * 2^(e-1), i.e. biased exponent 126 + e and mantissa
//            bit m placed at bit 22
// so the value is assembled directly with shifts and one select.
__device__ __forceinline__ float e2m1_to_float(uint8_t nibble) {
    const uint32_t e = (nibble >> 1) & 0x3u;  // exponent field
    const uint32_t m = nibble & 0x1u;         // mantissa bit
    const uint32_t mag = (e == 0) ? (m ? 0x3F000000u : 0u)
                                  : (((126u + e) << 23) | (m << 22));
    return __uint_as_float(mag | ((uint32_t)(nibble & 0x8u) << 28));
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
