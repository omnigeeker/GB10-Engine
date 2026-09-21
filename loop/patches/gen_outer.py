#!/usr/bin/env python3
"""Regenerate the two gemm2d_outer bodies in kernels/gemm.cu for a given tile shape.

Usage:  python3 loop/patches/gen_outer.py TM TNREG

The two operands are indexed differently, and getting this wrong raises IndexError
on 'xyzw'[4..7] -- see loop/patches/tm8-tnreg4.md round 118:

  * the WEIGHT float4 component is selected by the ROW: wv0 holds rows 0..3 and
    wv1 rows 4..7, because a thread's weight for row r is a scalar.
  * the ACTIVATION float4 component is selected by the COLUMN: xv covers columns
    0..3 and xw columns 4..7.

Shapes must satisfy: 64 % TM == 0, 64 % TNREG == 0, and
(64 // TM) * (64 // TNREG) == GB10_GEMM_BLOCK.
"""
import re
import sys

Q = "xyzw"


def bodies(tm, tnreg):
    assert 64 % tm == 0 and 64 % tnreg == 0, "shape must divide the 64x64 tile"
    assert (64 // tm) * (64 // tnreg) == 64 // tm * (64 // tnreg)
    # WEIGHT: component selected by ROW.
    W = lambda r, c: ("wv0.%s" % Q[r]) if r < 4 else ("wv1.%s" % Q[r - 4])
    # ACTIVATION: component selected by COLUMN.
    X = lambda c: ("xv.%s" % Q[c]) if c < 4 else ("xw.%s" % Q[c - 4])
    fma = "\n".join(
        "        acc[%d][%d] = fmaf(%s, %s, acc[%d][%d]);" % (r, c, W(r, c), X(c), r, c)
        for r in range(tm)
        for c in range(tnreg)
    )
    bf16_wv = "\n".join(
        "        const float2 a%s = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.%s));"
        % ("%d%d" % (2 * i, 2 * i + 1), Q[i])
        for i in range(tm // 2)
    )
    fma_bf = "\n".join(
        "        acc[%d][%d] = fmaf(wv[%d], %s, acc[%d][%d]);" % (r, c, r, X(c), r, c)
        for r in range(tm)
        for c in range(tnreg)
    )
    bf16_init = ", ".join("a%d%d.x, a%d%d.y" % (2 * i, 2 * i + 1, 2 * i, 2 * i + 1) for i in range(tm // 2))
    f32 = """__device__ __forceinline__ void gemm2d_outer(const float (*wt)[GB10_WSTRIDE],
                                             const float (*xt)[GB10_XSTRIDE],
                                             float (&acc)[GB10_TM][GB10_TNREG], int ty, int tx) {
    static_assert(GB10_TM == %d && GB10_TNREG == %d, "float4 path assumes %dx%d tiles");
#pragma unroll
    for (int k = 0; k < GB10_KC; ++k) {
%s
        const float4 xv = *reinterpret_cast<const float4*>(&xt[k][tx * GB10_TNREG]);
%s
%s
    }
}
""" % (
        tm,
        tnreg,
        tm,
        tnreg,
        "\n".join(
            "        const float4 wv%d = *reinterpret_cast<const float4*>(&wt[k][ty * GB10_TM + %d]);" % (i, 4 * i)
            for i in range(tm // 4)
        ),
        "        const float4 xw = *reinterpret_cast<const float4*>(&xt[k][tx * GB10_TNREG + 4]);"
        if tnreg > 4
        else "",
        fma,
    )
    bf = """__device__ __forceinline__ void gemm2d_outer_bf16(const uint16_t (*wt)[GB10_WSTRIDE],
                                                  const float (*xt)[GB10_XSTRIDE],
                                                  float (&acc)[GB10_TM][GB10_TNREG], int ty,
                                                  int tx) {
    static_assert(GB10_TM == %d && GB10_TNREG == %d, "bf16 path assumes %dx%d tiles");
#pragma unroll
    for (int k = 0; k < GB10_KC; ++k) {
        const uint4 wq = *reinterpret_cast<const uint4*>(&wt[k][ty * GB10_TM]);
%s
        const float wv[%d] = {%s};
        const float4 xv = *reinterpret_cast<const float4*>(&xt[k][tx * GB10_TNREG]);
%s
%s
    }
}
""" % (
        tm,
        tnreg,
        tm,
        tnreg,
        bf16_wv,
        tm,
        bf16_init,
        "        const float4 xw = *reinterpret_cast<const float4*>(&xt[k][tx * GB10_TNREG + 4]);"
        if tnreg > 4
        else "",
        fma_bf,
    )
    return f32, bf


def main():
    tm, tnreg = int(sys.argv[1]), int(sys.argv[2])
    f32, bf = bodies(tm, tnreg)
    p = "kernels/gemm.cu"
    s = open(p).read()
    s, n1 = re.subn(r"__device__ __forceinline__ void gemm2d_outer\(const float \(\*wt\).*?\n\}\n", f32, s, count=1, flags=re.S)
    s, n2 = re.subn(
        r"__device__ __forceinline__ void gemm2d_outer_bf16\(const uint16_t \(\*wt\).*?\n\}\n", bf, s, count=1, flags=re.S
    )
    assert n1 == 1 and n2 == 1, "failed to locate the outer products (%d, %d)" % (n1, n2)
    s = s.replace("#define GB10_TM %d" % (4 if tm == 8 else 8), "#define GB10_TM %d" % tm)
    s = s.replace("#define GB10_TNREG %d" % (8 if tnreg == 4 else 4), "#define GB10_TNREG %d" % tnreg)
    open(p, "w").write(s)
    print("regenerated gemm2d_outer{,_bf16} for TM=%d TNREG=%d" % (tm, tnreg))


if __name__ == "__main__":
    main()
