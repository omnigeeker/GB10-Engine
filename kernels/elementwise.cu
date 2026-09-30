// GB10-Engine: elementwise, norm, convolution, recurrent-state and attention
// kernels for the Qwen3.5 decode/prefill path.
//
// These are not the bandwidth bottleneck (the 17.6 GB of quantized weights
// streamed by the GEMV kernels dominates by ~100x), so they are written for
// obvious correctness first. The one exception is the Gated DeltaNet
// recurrence, whose per-sequence state is 3.1 MB per layer, and which is
// written to be bank-conflict-free.
//
// Conventions that are easy to get wrong, verified in docs/ARCHITECTURE.md:
//   * Qwen3_5RMSNorm is ZERO-CENTERED: gain is (1 + w), not w.
//   * Qwen3_5RMSNormGated is the ordinary form: gain is w, and the
//     normalisation happens BEFORE the gate.
//   * The attention output gate is sigmoid, not swish.

#include <cuda_fp16.h>
#include "gemv_common.cuh"

using namespace gb10;

namespace {

// Block-wide sum, broadcast to every thread through shared memory.
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float s[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) s[warp] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    if (threadIdx.x < nw) v = s[lane];
    else v = 0.0f;
    if (warp == 0) v = warp_reduce_sum(v);
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    const float r = s[0];
    __syncthreads();  // s is reused by later calls
    return r;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }

}  // namespace

// ---------------------------------------------------------------------------
// Norms
// ---------------------------------------------------------------------------

/// Zero-centered RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w).
/// One block per row.
extern "C" __global__ void rmsnorm_zero_centered_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
    int n, float eps) {
    const float* __restrict__ xr = x + (size_t)blockIdx.x * n;
    float* __restrict__ yr = y + (size_t)blockIdx.x * n;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block_reduce_sum(ss) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        yr[i] = xr[i] * inv * (1.0f + w[i]);
    }
}

/// Gated RMSNorm: y = (x * rsqrt(mean(x^2)+eps) * w) * silu(z).
/// One block per (row, head-group of size n).
extern "C" __global__ void rmsnorm_gated_kernel(
    const float* __restrict__ x, const float* __restrict__ z, const float* __restrict__ w,
    float* __restrict__ y, int n, float eps) {
    const float* __restrict__ xr = x + (size_t)blockIdx.x * n;
    const float* __restrict__ zr = z + (size_t)blockIdx.x * n;
    float* __restrict__ yr = y + (size_t)blockIdx.x * n;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block_reduce_sum(ss) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        yr[i] = xr[i] * inv * w[i] * silu_f(zr[i]);
    }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

/// y = silu(gate) * up   (SwiGLU)
extern "C" __global__ void swiglu_kernel(const float* __restrict__ gate,
                                         const float* __restrict__ up,
                                         float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = silu_f(gate[i]) * up[i];
}

/// y = a + b
extern "C" __global__ void add_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b, float* __restrict__ y,
                                      int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

/// y *= sigmoid(gate)   — the attention output gate.
extern "C" __global__ void sigmoid_mul_kernel(float* __restrict__ y,
                                              const float* __restrict__ gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= 1.0f / (1.0f + __expf(-gate[i]));
}

/// In-place L2 normalisation followed by a fixed scale:
/// x *= scale * rsqrt(sum(x^2) + eps). One block per vector of length n.
extern "C" __global__ void l2norm_scale_kernel(float* __restrict__ x, int offset, int n,
                                               float scale, float eps) {
    float* __restrict__ xr = x + offset + (size_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        ss = fmaf(v, v, ss);
    }
    const float mul = scale * rsqrtf(block_reduce_sum(ss) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) xr[i] *= mul;
}

// ---------------------------------------------------------------------------
// RoPE — GPT-NeoX (non-interleaved) rotation over the first `2*half` channels
// ---------------------------------------------------------------------------
extern "C" __global__ void rope_neox_kernel(float* __restrict__ q, float* __restrict__ k,
                                            const float* __restrict__ cs,
                                            const float* __restrict__ sn, int n_q_heads,
                                            int n_k_heads, int head_dim, int half) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (n_q_heads + n_k_heads) * half;
    if (idx >= total) return;

    float* __restrict__ base;
    int i;
    if (idx < n_q_heads * half) {
        base = q + (size_t)(idx / half) * head_dim;
        i = idx % half;
    } else {
        const int t = idx - n_q_heads * half;
        base = k + (size_t)(t / half) * head_dim;
        i = t % half;
    }
    const float c = cs[i], s = sn[i];
    const float a = base[i], b = base[i + half];
    base[i] = fmaf(-b, s, a * c);
    base[i + half] = fmaf(a, s, b * c);
}

// ---------------------------------------------------------------------------
// Depthwise causal conv1d (kernel 4, left padding 3) + SiLU, one step.
//
// out[c] = silu(w[c][0]*h0 + w[c][1]*h1 + w[c][2]*h2 + w[c][3]*x[c])
// where h0..h2 are the three previous tokens. The history is shifted in place.
// ---------------------------------------------------------------------------
extern "C" __global__ void conv1d_step_silu_kernel(const float* __restrict__ x,
                                                   const float* __restrict__ w,
                                                   float* __restrict__ hist,
                                                   float* __restrict__ y, int channels,
                                                   int base) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* __restrict__ wc = w + (size_t)c * 4;
    float* __restrict__ h = hist + base + (size_t)c * 3;
    const float xc = x[c];
    const float acc =
        fmaf(wc[0], h[0], fmaf(wc[1], h[1], fmaf(wc[2], h[2], wc[3] * xc)));
    y[c] = silu_f(acc);
    h[0] = h[1];
    h[1] = h[2];
    h[2] = xc;
}

// ---------------------------------------------------------------------------
// Gated DeltaNet recurrence, one decode step.
//
// One block per (value head, batch). 128 threads, one per value channel.
//
// The state S is [128 x 128] fp32 (k_head_dim x v_head_dim) and is the reason
// this kernel exists as a separate pass: 48 heads x 64 KB = 3.1 MB per layer
// per sequence. It is staged in padded shared memory so that the inner loops —
// where all 128 threads read S[i][j] for the same i and consecutive j — are
// bank-conflict-free.
//
//   S      *= exp(g)                      (decay)
//   kv_mem  = sum_i S[i][:] * k[i]
//   delta   = (v - kv_mem) * beta
//   S      += outer(k, delta)
//   out     = sum_i S[i][:] * q[i]
//
// `q` and `k` must already be L2-normalised and scaled by head_dim^-0.5.
// ---------------------------------------------------------------------------
extern "C" __global__ void gated_delta_rule_step_kernel(
    const float* __restrict__ qkv, int q_off, int k_off, int v_off, int row_stride,
    const float* __restrict__ decay, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ out, int n_v_heads, int n_k_heads,
    int group, int base) {
    constexpr int D = 128;
    const int hv = blockIdx.x;
    const int b = blockIdx.y;
    const int j = threadIdx.x;
    const int kh = hv / group;

    __shared__ float S[D][D + 1];
    __shared__ float sk[D];

    const float* __restrict__ row = qkv + (size_t)b * row_stride;
    const float* __restrict__ qh = row + q_off + (size_t)kh * D;
    const float* __restrict__ khp = row + k_off + (size_t)kh * D;
    const float* __restrict__ vh = row + v_off + (size_t)hv * D;
    float* __restrict__ sh = state + base + ((size_t)b * n_v_heads + hv) * D * D;

    // Load the persistent state. The caller zeroes it at sequence start; this
    // kernel must not, or every decode step would forget the whole prefix.
    for (int i = threadIdx.x; i < D * D; i += blockDim.x) S[i / D][i % D] = sh[i];
    for (int i = threadIdx.x; i < D; i += blockDim.x) sk[i] = khp[i];
    __syncthreads();

    const float dec = decay[(size_t)b * n_v_heads + hv];
    const float bet = beta[(size_t)b * n_v_heads + hv];
    const float vj = vh[j];

    float kv = 0.0f;
    for (int i = 0; i < D; ++i) {
        const float s = S[i][j] * dec;
        S[i][j] = s;
        kv = fmaf(s, sk[i], kv);
    }
    const float delta = (vj - kv) * bet;

    float o = 0.0f;
    for (int i = 0; i < D; ++i) {
        const float s = fmaf(sk[i], delta, S[i][j]);
        S[i][j] = s;
        o = fmaf(s, qh[i], o);
    }
    out[(size_t)b * n_v_heads * D + (size_t)hv * D + j] = o;

    __syncthreads();
    for (int i = threadIdx.x; i < D * D; i += blockDim.x) sh[i] = S[i / D][i % D];
}

// ---------------------------------------------------------------------------
// Prefill causal attention with GQA, one block per (query head, query token).
//
// Simple and obviously correct: the head-dim reduction is a block reduction
// repeated once per key. That is O(T) block reductions per query, which is
// fine for the correctness fixtures but will be replaced by a tiled
// flash-attention kernel for real prefill throughput (M6).
// ---------------------------------------------------------------------------
// Pack two fp16 values into one .b32 for an mma fragment, low half first.
// The order matters and is not symmetric: verified against a CPU reference
// before use (bench/longctx/comparison.md, "mma-layout").
// Shared-memory window address for ldmatrix.
__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ unsigned pk2(__half lo, __half hi) {
    return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}

// NOTE: loading a whole fragment register as one 32-bit word (`pk2_at`, six
// loads per k-step instead of twelve) was tried and measured SLOWER -- 16384
// 0.64 -> 0.75 s and 65536 10.38 -> 11.9 s, reproducible over three runs. The
// two 2-byte loads are not the cost. Kept out deliberately; see the three
// failed load-count hypotheses in bench/longctx/comparison.md.

extern "C" __global__ void attn_prefill_kernel(
    const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    float* __restrict__ out, int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    float scale, int start, int kv_base) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const int group = n_q_heads / n_kv_heads;
    const int kh = h / group;

    // `scores` holds one entry per *key*, and the key window is
    // `0..=start+t`: `start` is how many keys were already in the cache before
    // these tokens. Row `t` of `q`/`out` is the `t`-th *new* token, while `k`/`v`
    // are the whole cache for this sequence, offset by `kv_base`.
    extern __shared__ float scores[];  // start + n_tokens entries

    // Absolute position of this query row.
    const int win = start + t;

    const bool active = d < head_dim;
    const float qv = active ? q[((size_t)t * n_q_heads + h) * head_dim + d] : 0.0f;

    // One block reduction per key. O(T) reductions per query: correct and
    // simple, but quadratic in T. Replaced by a tiled kernel in M6.
    for (int s = 0; s <= win; ++s) {
        const float kv_ = active
            ? __half2float(k[(size_t)kv_base + ((size_t)s * n_kv_heads + kh) * head_dim + d]) : 0.0f;
        const float dot = block_reduce_sum(qv * kv_) * scale;
        if (d == 0) scores[s] = dot;
        __syncthreads();
    }

    // Every thread redundantly recomputes the softmax (T is small here), which
    // avoids further synchronisation.
    float mx = -INFINITY;
    for (int s = 0; s <= win; ++s) mx = fmaxf(mx, scores[s]);
    float sum = 0.0f;
    for (int s = 0; s <= win; ++s) sum += __expf(scores[s] - mx);
    const float inv = 1.0f / sum;

    if (active) {
        float acc = 0.0f;
        for (int s = 0; s <= win; ++s) {
            const float p = __expf(scores[s] - mx) * inv;
            acc = fmaf(p, __half2float(v[(size_t)kv_base + ((size_t)s * n_kv_heads + kh) * head_dim + d]), acc);
        }
        out[((size_t)t * n_q_heads + h) * head_dim + d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Tiled causal prefill attention with GQA and online softmax.
//
// `attn_prefill_kernel` above keeps one score per key in dynamic shared memory
// -- it asks for `(start + n_tokens) * 4` bytes -- which caps the reachable
// context at the shared-memory limit (48 KB here, so 12,288 keys) and fails
// outright past it. Its inner loop also runs one block reduction per key.
//
// This kernel instead keeps only a `BQ x BK` score tile resident and streams
// the key range in tiles of BK, carrying the running max, running sum and
// output accumulator (the online-softmax recurrence). Shared memory is then a
// constant independent of context length, so the reachable context is bounded
// by the KV cache rather than by smem.
//
// Layouts and semantics match `attn_prefill_kernel` exactly:
//   q/out : [n_tokens, n_q_heads, head_dim]
//   k/v   : [n_keys_total, n_kv_heads, head_dim], offset by `kv_base` floats
//   `start` keys precede these tokens, so row t attends over 0..=start+t.
//
// One block covers `BQ` query rows of a single query head, one thread per
// head dimension (blockDim.x == head_dim), so each thread owns the output for
// its dimension across all BQ rows and keeps those BQ accumulators in
// registers.
// ---------------------------------------------------------------------------
// PREFILL_BQ * PREFILL_BK MUST divide evenly into the score loop's passes.
// That loop uses two threads per (query, key) pair through
// __shfl_xor_sync(0xffffffff), so every lane of a warp must execute the same
// NUMBER of passes; the block is head_dim = 256 threads, so one pass consumes
// 128 pairs. 8 * 16 = 128 is exactly one pass, which is why this blocking was
// chosen. Retuning to 24 * 11 = 264 was measured at roughly 5x SLOWER for
// exactly this reason: lanes 0-15 ran a third pass while 16-255 did not, and
// the shuffle named lanes that were not executing. Raise BQ * BK only to a
// multiple of 128, and re-check the 48 KB shared-memory budget in
// gb10_cuda::ops::attn_prefill_tiled.
// The score loop pairs two threads per (i,j) with `__shfl_xor_sync(0xffffffff)`,
// so PREFILL_BQ * PREFILL_BK must be a multiple of blockDim.x / 2 and every lane
// of a warp must run the same number of passes. With blockDim.x == head_dim ==
// 256 that is a multiple of 128, and 24 * 16 == 384 == exactly 3 passes.
//
// V does not need shared memory. It is only ever read as `Vs[j * HD + tid]` --
// one column per thread for each j -- so each thread can hold its own BK values
// in registers. Dropping the BK * HD staging buffer is what makes room for the
// 3x larger BQ * BK tile: the shared budget is
// (BQ + BK) * HD + BQ * BK + 3 * BQ = 10696 floats = 42,784 B, against 48 KB.
#define PREFILL_BQ 24
#define PREFILL_BK 16

extern "C" __global__ void attn_prefill_tiled_kernel(
    const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    float* __restrict__ out, int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    float scale, int start, int kv_base) {
    extern __shared__ float smem[];
    const int HD = head_dim;
    // Rows are padded to HD + 1. With the natural HD stride every row of Q and K
    // starts on the same shared bank (HD is a multiple of 32), and the score loop
    // reads `Ks[j * HD + sub * half + d]` for 16 different j and 2 sub values at
    // the same d -- 32 distinct words, all in bank d % 32. That is a 32-way
    // conflict, i.e. 32 transactions per load instruction. The +1 makes the bank
    // (row + d) % 32, so the 16 rows spread across 16 banks and the only
    // remaining conflict is the 2-way one between the two `sub` halves, whose
    // offset (half == 128) is itself a multiple of 32. Numerically identical:
    // this only changes addresses.
    // The two `sub` halves are separated by one extra float as well, so their
    // offsets differ by `half + 1 == 129` and therefore by 1 bank (129 % 32).
    // With the row stride at `2 * (half + 1) == 258` (2 banks apart per row) the
    // 16 `j` values and 2 `sub` values of a warp now span
    // `(2 * j + sub + d) % 32` = all 32 banks, so the K read is a single
    // transaction instead of the 2-way conflict left by plain padding.
    // Q and K are staged in fp16 (not bf16: bf16's 8 mantissa bits failed the
    // attn-tile gate at ~1.2e-4 rms relative, and the failure pattern -- ntok=1
    // exact, ntok>=7 all wrong -- is pure score precision, not a layout bug.
    // fp16 has 11 mantissa bits for the same 2 bytes). That halves their shared
    // footprint, which is
    // what takes this kernel from two blocks per SM to four -- and the round-39
    // probe measured this kernel to be occupancy-sensitive (halving the blocks
    // cost 1.64x on the context-dependent term, with the GEMM's constant term
    // unchanged at 0.975x).
    //
    // the same 2-byte bank arithmetic applies to fp16, so:
    // PADH is 130, not 129, and that is not cosmetic. The bank of the element at
    // index i is (i * width / 4) % 32, so with 2-byte elements two neighbours
    // share a 4-byte bank and the `sub` offset has half the bank resolution it
    // has for fp32. PADH = 129 gives floor(129/2) = 64, and 64 % 32 = 0, so both
    // halves land in the same bank class: a 2-way conflict in the score loop.
    // PADH = 130 gives 65, and 65 % 32 = 1, which restores the odd shift that
    // spreads (j, sub) across all 32 banks -- the same property PADH = 129
    // provides for fp32. The intra-row gap below is 2 for the same reason.
    // See bench/longctx/comparison.md, round 41.
    // Row stride is a multiple of 8 halfs (16 bytes) so every 16-byte segment
    // ldmatrix reads is aligned, which is what lets ONE instruction replace six
    // shared loads plus their address arithmetic in the score mma. The old
    // `PADH = HD/2 + 2` gap (PS = 2*PADH) existed only to spread the SCALAR
    // kernel's `sub * PADH` accesses across banks; nothing reads Q or K
    // scalar-wise any more, and ldmatrix does its own conflict-free access, so
    // the gap is gone. That also removes the `koff` correction the mma fragment
    // loads needed to step over the gap.
    const int PS = HD + 8;
    __half* Qs = reinterpret_cast<__half*>(smem);  // BQ * PS fp16
    __half* Ks = Qs + PREFILL_BQ * PS;                    // BK * PS fp16
    float* S = reinterpret_cast<float*>(Ks + PREFILL_BK * PS);   // BQ * BK fp32
    float* red = S + PREFILL_BQ * PREFILL_BK;      // 3 * BQ  (m, l, correction)
    // A shared-memory staging tile for V, so its sixteen per-thread loads become
    // two uint4 loads -- the same transformation that took the K staging down 25%.
    // V was measured at 30.9% of this kernel by removing its loads outright.
    // The extra 8,448 B does NOT cost occupancy: 22,944 B and 31,392 B both give
    // three blocks per SM (102400/22944 = 4.46, 102400/31392 = 3.26, and the
    // kernel is at 3 by registers anyway), so the tile is free.
    __half* Vs = reinterpret_cast<__half*>(red + 3 * PREFILL_BQ);   // BK * PS fp16

    const int h = blockIdx.x;
    const int t0 = blockIdx.y * PREFILL_BQ;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int group = n_q_heads / n_kv_heads;
    const int kh = h / group;
    const int rows = min(PREFILL_BQ, n_tokens - t0);
    if (rows <= 0) return;

    // Q tile. Rows past `rows` are zero-filled and masked out below.
    for (int idx = tid; idx < PREFILL_BQ * HD; idx += nt) {
        const int i = idx / HD, d = idx % HD;
        Qs[i * PS + d] = __float2half(
            (i < rows) ? q[((size_t)(t0 + i) * n_q_heads + h) * HD + d] : 0.0f);
    }
    if (tid < PREFILL_BQ) {
        red[tid] = -INFINITY;                  // m
        red[PREFILL_BQ + tid] = 0.0f;          // l
    }

    // BQ accumulators in registers: thread `tid` owns output dim `tid`.
    float acc[PREFILL_BQ];
#pragma unroll
    for (int i = 0; i < PREFILL_BQ; ++i) acc[i] = 0.0f;
    float vr[PREFILL_BK];

    __syncthreads();

    // Highest key index any row in this block may attend to.
    const int win_max = start + t0 + rows - 1;

    for (int s0 = 0; s0 <= win_max; s0 += PREFILL_BK) {
        // Stage this key tile. Keys past `win_max` are irrelevant to every row
        // here and get zero-filled; the causal mask below excludes them anyway.
        //
        // Eight halves (16 bytes) per thread per pass rather than one, so this
        // is two 16-byte instructions per thread instead of sixteen 2-byte
        // ones. Both sides are 16-byte aligned: a group of eight never crosses
        // a row because `head_dim` is a multiple of 8 and `d` is a multiple of
        // 8, the global row base `(s * n_kv_heads + kh) * head_dim` is a
        // multiple of 8, and the shared row stride `PS = head_dim + 8` is too.
        // The kernel's staging is instruction-issue limited -- the comment
        // below records the measured ~32 loads/cycle/SM ceiling that the scalar
        // operand path hit -- so moving the same bytes in 8x fewer instructions
        // is the point.
        for (int idx = tid * 8; idx < PREFILL_BK * HD; idx += nt * 8) {
            const int j = idx / HD, d = idx % HD;
            const int s = s0 + j;
            uint4 kval, vval;
            if (s <= win_max) {
                const size_t base = (size_t)kv_base + ((size_t)s * n_kv_heads + kh) * HD + d;
                kval = *reinterpret_cast<const uint4*>(k + base);
                vval = *reinterpret_cast<const uint4*>(v + base);
            } else {
                kval = make_uint4(0u, 0u, 0u, 0u);
                vval = make_uint4(0u, 0u, 0u, 0u);
            }
            *reinterpret_cast<uint4*>(Ks + j * PS + d) = kval;
            *reinterpret_cast<uint4*>(Vs + j * PS + d) = vval;
        }
        __syncthreads();
        // This thread's column of V, held in registers, now read from the staged
        // tile. It MUST be after the barrier: unlike the global load it replaced,
        // these bytes were written by other threads in this block. The reads are
        // shared-memory, so they are cheap and do not need latency hiding.
#pragma unroll
        for (int j = 0; j < PREFILL_BK; ++j) {
            vr[j] = __half2float(Vs[j * PS + tid]);
        }

        // ---- score: S[BQ][BK] = Q . K^T * scale, on tensor cores ----------
        //
        // `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` computes
        // D[m][n] = sum_k A[m][k] * B[k][n] with B held column-major, i.e. with
        // BOTH operands k-contiguous -- which is exactly Q . K^T here, because a
        // key's head_dim is contiguous both in the global layout and in the
        // staged tile above. The fragment mapping was checked against a CPU
        // reference on its own before being wired in (128/128 exact):
        //   A: a0 = {Qs[m0+gid][kt+c], Qs[m0+gid][kt+c+1]}, a1 = row gid+8,
        //      a2/a3 = the same two rows at kt+c+8
        //   B: b0 = {Ks[n0+gid][kt+c], Ks[n0+gid][kt+c+1]}, b1 = same at kt+c+8
        //   D: d0 = S[m0+gid][n0+t4*2], d1 = col+1, d2/d3 = row gid+8
        //
        // The scalar version this replaces issued ~256 shared loads per thread
        // per tile and ran at the measured ~32 loads/cycle/SM operand-load
        // ceiling. That ceiling is why the two cheaper fixes both failed --
        // fewer loads per fma (round 44, BK=48) changed nothing, and cutting the
        // accumulator chain to twelve partial sums was a 12% regression. Moving
        // the operands into registers once per k-step and letting the tensor
        // core consume 16x8x16 MACs per instruction removes the resource that
        // was actually saturated.
        //
        // BK == 16 gives 2 x 8-key n-tiles and BQ == 24 needs 2 x 16-row
        // m-tiles, so four warps cover the whole tile with no cross-warp
        // reduction. The m-tiles OVERLAP by 8 rows (mt=0 covers rows 0..15,
        // mt=1 covers rows 8..23) rather than padding Qs to 32 rows: it keeps
        // the shared tile at its current size, and the two writers of rows 8..15
        // compute the identical value from identical inputs, so the duplicate
        // store is benign. The other four warps idle in this phase and rejoin
        // for the softmax; four warps of mma is already far past the scalar
        // loop, so there is nothing to gain by splitting it further.
        {
            const int warp = tid >> 5;
            if (warp < 4) {
                const int lane = tid & 31;
                const int gid = lane >> 2;
                const int t4 = lane & 3;
                const int c = t4 * 2;
                const int mt = warp >> 1;        // 0 -> rows 0..15, 1 -> rows 8..23
                const int ntile = warp & 1;      // 0 -> keys 0..7,  1 -> keys 8..15
                const int m0 = mt * 8;
                const int r0 = m0 + gid;
                const int r1 = r0 + 8;
                // ldmatrix address lanes. For x4 every lane names one 16-byte row
                // segment: lanes 0-15 cover rows m0+0..15 at column 0, lanes 16-31
                // the same rows at column 8, which lands the four resulting 8x8
                // matrices in exactly the mma's a0..a3. For x2 only lanes 0-15 are
                // read: rows n0+0..7 at columns 0 and 8, giving b0 and b1.
                const int arow = m0 + ((lane < 16) ? lane : (lane - 16));
                const int acol = (lane < 16) ? 0 : 8;
                const int brow = ntile * 8 + (lane & 7);
                const int bcol = ((lane & 8) != 0) ? 8 : 0;
                float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int kt = 0; kt < HD; kt += 16) {
                    unsigned a[4], b[2];
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                        : "r"(smem_addr(&Qs[arow * PS + kt + acol])));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(b[0]), "=r"(b[1])
                        : "r"(smem_addr(&Ks[brow * PS + kt + bcol])));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                          "r"(b[0]), "r"(b[1]));
                }
                // Scale, apply the row/causal mask, and scatter into S for the
                // online softmax below. d0/d1 are row r0, d2/d3 are row r1.
                const int cr[4] = {c, c + 1, c, c + 1};
                const int rr[4] = {r0, r0, r1, r1};
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int i = rr[u];
                    const int jc = ntile * 8 + cr[u];
                    const int s = s0 + jc;
                    const bool ok = (i < rows) && (s <= start + t0 + i);
                    S[i * PREFILL_BK + jc] = ok ? d[u] * scale : -INFINITY;
                }
            }
        }
        __syncthreads();

        // Online softmax, one thread per row.
        if (tid < PREFILL_BQ && tid < rows) {
            const int i = tid;
            const float m = red[i], l = red[PREFILL_BQ + i];
            float mt = m;
            for (int j = 0; j < PREFILL_BK; ++j)
                mt = fmaxf(mt, S[i * PREFILL_BK + j]);
            if (mt == -INFINITY) {
                // Entire tile masked for this row: leave m, l and acc alone.
                for (int j = 0; j < PREFILL_BK; ++j) S[i * PREFILL_BK + j] = 0.0f;
                red[2 * PREFILL_BQ + i] = 1.0f;
            } else {
                const float c = __expf(m - mt);
                float ls = 0.0f;
                for (int j = 0; j < PREFILL_BK; ++j) {
                    const float p = __expf(S[i * PREFILL_BK + j] - mt);
                    S[i * PREFILL_BK + j] = p;
                    ls += p;
                }
                red[i] = mt;
                red[PREFILL_BQ + i] = l * c + ls;
                red[2 * PREFILL_BQ + i] = c;
            }
        }
        __syncthreads();

        // acc[i] = acc[i] * c_i + P[i] . Vs
#pragma unroll
        for (int i = 0; i < PREFILL_BQ; ++i) {
            if (i >= rows) continue;
            const float c = red[2 * PREFILL_BQ + i];
            float a = acc[i] * c;
            const float* prow = S + i * PREFILL_BK;
            for (int j = 0; j < PREFILL_BK; ++j)
                a = fmaf(prow[j], vr[j], a);
            acc[i] = a;
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < PREFILL_BQ; ++i) {
        if (i >= rows || tid >= HD) continue;
        out[((size_t)(t0 + i) * n_q_heads + h) * HD + tid] = acc[i] / red[PREFILL_BQ + i];
    }
}

// ---------------------------------------------------------------------------
// FA2-style prefill attention: `attn_prefill_fa2_kernel`.
//
// Selected by `GB10_FA2=1` inside `Ops::attn_prefill_tiled`; it falls back to
// `attn_prefill_tiled_kernel` above whenever head_dim != 256 or the GQA ratio
// is not exactly 6. The old kernel is untouched.
//
// The two structural changes against the old kernel, both from
// fa_brief/llamacpp_fa_prefill_brief.md:
//
//   1. P never touches shared memory and is never consumed by FFMA. The QK^T
//      accumulator is converted to fp16 fragments IN REGISTERS (plain
//      `make_half2` pairing -- no `movmatrix`, because the QK^T below is laid
//      out with M = qcols so the fp32 KQ accumulator already comes out in the
//      `[qcols][keys]` order that the P*V A-operand wants) and fed straight
//      into the second `mma`.
//   2. One block covers all 6 query heads that share a KV head, so the 32 KB
//      K/V tile is reused 6x instead of once. Arithmetic intensity per byte of
//      K/V goes from 24 to 48 FLOP/B.
//
// Geometry (the addendum in fa_brief/FA2_IMPLEMENTATION_SPEC.md):
//
//   ncols1 = 8 query rows, ncols2 = 6 query heads (exact GQA, no padding),
//   ncols = 48 qcols = 3 warps x 16, 96 threads, key tile 32, head dim 256 in
//   one tile. Q is held in registers (64 regs) so there is no Q smem tile and
//   no epilogue combine buffer; smem is K 16 KB + V 16 KB = 32 KB, which is
//   3 CTAs/SM against the 101,376 B opt-in ceiling.
//
// Qcol index c in [0,48) is `head*8 + row`: warp w owns qcols [16w,16w+16),
// i.e. heads {2w, 2w+1} x query rows 0..7. A fragment rows 0..7 are head 2w
// and rows 8..15 are head 2w+1, so a lane's two KQ rows (`gid` and `gid+8`)
// are the SAME query row under two different heads. The causal boundary
// `s <= start + t0 + i` therefore depends only on `gid`, which is what makes
// one mask test serve both rows.
//
// Fragment layouts, verified against a CPU/probe reference on this part
// (sm_121, nvcc 13.0) rather than assumed:
//
//   QK^T: mma.m16n8k16.row.col.f32.f16.f16.f32, M = qcols(16), N = keys(8),
//         K = head_dim(16 per step, 16 steps). A = Q row-major (16x16), B = K
//         column-major, and since a key's head_dim is contiguous the plain
//         `ldmatrix.m8n8.x4` on the `[key][head_dim]` tile lands b0/b1 exactly.
//         The probe confirms one x4 (lanes 0-7 -> keys +0..7 @hd+0, 8-15 ->
//         keys +0..7 @hd+8, 16-23 -> keys +8..15 @hd+0, 24-31 -> keys +8..15
//         @hd+8) yields {b0(nt), b1(nt), b0(nt+1), b1(nt+1)} in r0..r3.
//   P*V:  mma.m16n8k16.row.col.f16.f16.f16.f16, M = qcols(16), N = dv(8),
//         K = keys(16 per step, 2 steps). B = V^T, loaded with
//         `ldmatrix.m8n8.x4.trans` straight off the `[key][dv]` tile so V is
//         NOT stored transposed. The probe confirms that with lanes 0-7 ->
//         keys +0..7 @dv+0, 8-15 -> keys +0..7 @dv+8, 16-23 -> keys +8..15
//         @dv+0, 24-31 -> keys +8..15 @dv+8, the transposed outputs are
//         {b0(dv0), b0(dv1), b1(dv0), b1(dv1)} in r0..r3 (the 1<->2 swap that
//         `mma.cuh:884-894` bakes into its asm output list).
//
// Softmax is llama.cpp's, because the fp16 P*V accumulator needs it:
// FATTN_KQ_MAX_OFFSET = 3*ln2, the -20.0f FTZ bit-trick on the rescale factor,
// an in-place half2 rescale of the fp16 VKQ fragments, the rowsum reduction
// over __shfl_xor_sync offsets 2 and 1 only (the qcol row is held by lanes
// with equal lane/4), masking by writing -INFINITY into the score, KQ_max
// initialised to -FLT_MAX/2 so an all-masked row cannot produce NaN, and the
// divide by rowsum deferred to the very end. P*V accumulating in fp16 is
// deliberate -- it is what keeps the register budget at 3 CTAs/SM.
// ---------------------------------------------------------------------------

#define FA2_HD         256
#define FA2_BC         32      // key rows per tile (nbatch_fa)
#define FA2_NROWS      8       // ncols1: query rows per block
#define FA2_GQA        6       // ncols2: query heads per KV head
#define FA2_NCOLS      48      // qcols = 8 * 6
#define FA2_THREADS    96      // 3 warps
#define FA2_STRIDE_H2  128     // K/V smem row stride in half2 (256 halves, no pad)
#define FA2_KQ_OFFSET  2.0794415f          // 3 * ln2
#define FA2_FTZ_THRESH (-20.0f)
// -FLT_MAX/2 (llama.cpp's KQ_max seed): an all-masked row must not go to NaN.
#define FA2_NEG_HUGE    (-1.7014117e38f)

// XOR swizzle from fattn-swizzle.cuh:6-47 with `stride = 128` half2, which is
// a multiple of 32 so no row padding is needed. The XOR touches bits 4-6 of
// the byte address, i.e. it permutes the 16-byte units inside each 128-byte
// group, so every ldmatrix row (one 16-byte unit) stays intact.
__device__ __forceinline__ unsigned fa2_swz(int row, int col_h2) {
    return (unsigned)((row * FA2_STRIDE_H2 + col_h2) * 4) ^ (unsigned)((row & 7) << 4);
}

// Pack two fp32 into one .b32 fp16 pair, low half = first argument. Same
// convention as `pk2` above and as the mma fragment docs: element k in the low
// half, element k+1 in the high half.
__device__ __forceinline__ unsigned fa2_pk2f(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Issue the 16-byte `cp.async` copies for one K or V tile (keys [s0, s0+32))
// into `dst`, then close the group. Keys past `win_max` are zero-filled with the
// `src-size` operand (the bytes past src-size are written as zero) instead of a
// branch, so out-of-range keys cost the same instruction as in-range ones and
// the whole tile is a single unconditional pass.
//
// One warp covers exactly one 512-byte key row per pass (idx & 31), so the
// copies are fully coalesced 16-byte transactions, exactly as the synchronous
// uint4 version was.
__device__ __forceinline__ void fa2_stage_async(
    unsigned char* dst, const __half* __restrict__ src, int s0, int win_max, int kh,
    int n_kv_heads, int kv_base, int tid) {
#pragma unroll 1
    for (int idx = tid; idx < FA2_BC * 32; idx += FA2_THREADS) {
        const int r = idx >> 5;          // key row within the tile
        const int u = idx & 31;          // 16-byte unit within the 512-byte row
        const int s = s0 + r;
        const bool ok = (s <= win_max);
        // When src_bytes == 0 the address is not dereferenced; clamp it to a
        // valid location anyway rather than forming an out-of-range pointer.
        const size_t off = ok
            ? (size_t)kv_base + ((size_t)s * n_kv_heads + kh) * FA2_HD + (size_t)u * 8
            : (size_t)kv_base;
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(smem_addr(dst + fa2_swz(r, u * 4))), "l"(src + off),
               "r"(ok ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n");
}

extern "C" __global__ void __launch_bounds__(FA2_THREADS, 3) attn_prefill_fa2_kernel(
    const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    float* __restrict__ out, int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    float scale, int start, int kv_base) {
    // 32 KB dynamic: K then V, both FA2_BC * FA2_STRIDE_H2 half2.
    extern __shared__ __align__(16) unsigned char fa2_raw[];
    unsigned char* tile_K = fa2_raw;
    unsigned char* tile_V = fa2_raw + (size_t)FA2_BC * FA2_STRIDE_H2 * 4;

    const int kh = blockIdx.x;                    // KV head this block serves
    const int t0 = blockIdx.y * FA2_NROWS;        // local query-row offset
    const int rows = min(FA2_NROWS, n_tokens - t0);
    if (rows <= 0) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;                    // 0..7: query row of this lane
    const int t4 = lane & 3;                      // 0..3: column pair in a fragment

    const int h0 = kh * FA2_GQA + 2 * warp;       // the two query heads of this warp
    const bool row_ok = (gid < rows);

    // ---- Q fragments, registers only (64 regs) ---------------------------
    // A = Q is row-major 16x16 per k-step: a0 = rows gid @ kt+t4*2..+1,
    // a1 = rows gid+8 (the other head, same query row), a2/a3 = the same at
    // +8. So a lane needs four head-dim elements per k-step per head, which is
    // two 8-byte loads -- the head dim is contiguous and 16-byte aligned.
    unsigned Q_B[16][4];
#pragma unroll
    for (int kt = 0; kt < 16; ++kt) {
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            float2 lo = make_float2(0.0f, 0.0f);
            float2 hi = make_float2(0.0f, 0.0f);
            if (row_ok) {
                const size_t base = ((size_t)(t0 + gid) * n_q_heads + (h0 + jj)) * FA2_HD
                                    + (size_t)kt * 16 + t4 * 2;
                lo = *reinterpret_cast<const float2*>(q + base);
                hi = *reinterpret_cast<const float2*>(q + base + 8);
            }
            // `scale` is 1/sqrt(256) == 2^-4 here, so folding it into Q before
            // the fp16 round is exact and identical to scaling S afterwards.
            Q_B[kt][jj]     = fa2_pk2f(lo.x * scale, lo.y * scale);
            Q_B[kt][2 + jj] = fa2_pk2f(hi.x * scale, hi.y * scale);
        }
    }

    float KQ_max[2] = {FA2_NEG_HUGE, FA2_NEG_HUGE};
    float KQ_rowsum[2] = {0.0f, 0.0f};
    // fp16 P*V accumulator: 32 dv n-tiles x {row gid, row gid+8}.
    unsigned VKQ_C[32][2];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        VKQ_C[i][0] = 0u;
        VKQ_C[i][1] = 0u;
    }

    // Highest key any row in this block may attend to, and this lane's own
    // causal boundary (identical for both of its heads).
    const int win_max = start + t0 + rows - 1;
    const int row_key_max = start + t0 + gid;

    // ---- software pipeline -------------------------------------------------
    // Two slots, no double buffering: tile_K and tile_V are already separate
    // buffers, so each one prefetches its OWN next tile and the two copies
    // overlap two different compute phases:
    //
    //   K(s0)   is in flight during iteration s0-32's mask/softmax/P/P*V
    //   V(s0)   is issued at the top of iteration s0 and in flight during
    //           QK^T(s0)
    //   K(s0+32) is issued immediately after QK^T(s0) and in flight during
    //           mask(s0) + softmax(s0) + P packing + P*V(s0)
    //
    // This is llama.cpp's `nstages == 2` scheme (`fattn-mma-f16.cuh:623-631`
    // and `:975-990`). It costs no shared memory and therefore no occupancy:
    // the footprint stays at K 16 KB + V 16 KB = 32 KB, i.e. 3 CTAs/SM, which
    // double-buffering both tiles (64 KB, 1 CTA/SM) would have destroyed.
    //
    // Hazard notes, since each barrier below is load-bearing:
    //   * the top-of-loop barrier is what lets V(s0) overwrite tile_V -- it
    //     proves every warp has finished the previous iteration's P*V read.
    //   * the mid-loop barrier (placed just after QK^T) is what lets K(s0+32)
    //     overwrite tile_K -- it proves every warp has finished this
    //     iteration's QK^T read.
    // Both are the same two barriers the fully synchronous version had; the
    // change is only that the copies now overlap compute instead of serialising.
    // `win_max >= 0` always (key 0 is visible to row 0), so the prologue always
    // has a tile to fetch.
    fa2_stage_async(tile_K, k, 0, win_max, kh, n_kv_heads, kv_base, tid);

    for (int s0 = 0; s0 <= win_max; s0 += FA2_BC) {
        // ---- K(s0) is resident; V(s0) starts loading now ------------------
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        fa2_stage_async(tile_V, v, s0, win_max, kh, n_kv_heads, kv_base, tid);

        // ---- QK^T: KQ_C[qcol][key] = Q . K^T (scale already in Q) --------
        float KQ_C[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int l = 0; l < 4; ++l) KQ_C[nt][l] = 0.0f;
        }
#pragma unroll
        for (int kt = 0; kt < 16; ++kt) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                // One x4 covers two 8-key n-tiles: r0/r1 = b0/b1 of n-tile
                // 2*np, r2/r3 = b0/b1 of n-tile 2*np+1.
                const int krow = np * 16 + (lane & 7) + ((lane & 16) ? 8 : 0);
                const int kcol = kt * 8 + ((lane & 8) ? 4 : 0);
                unsigned b[4];
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                    : "r"(smem_addr(tile_K + fa2_swz(krow, kcol))));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(KQ_C[np * 2][0]), "+f"(KQ_C[np * 2][1]),
                      "+f"(KQ_C[np * 2][2]), "+f"(KQ_C[np * 2][3])
                    : "r"(Q_B[kt][0]), "r"(Q_B[kt][1]), "r"(Q_B[kt][2]), "r"(Q_B[kt][3]),
                      "r"(b[0]), "r"(b[1]));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(KQ_C[np * 2 + 1][0]), "+f"(KQ_C[np * 2 + 1][1]),
                      "+f"(KQ_C[np * 2 + 1][2]), "+f"(KQ_C[np * 2 + 1][3])
                    : "r"(Q_B[kt][0]), "r"(Q_B[kt][1]), "r"(Q_B[kt][2]), "r"(Q_B[kt][3]),
                      "r"(b[2]), "r"(b[3]));
            }
        }

        // ---- K(s0) has now been fully consumed; start K(s0+32) loading ----
        // The barrier is the same mid-loop barrier the synchronous version had,
        // moved one phase earlier: QK^T is the only reader of tile_K, and the
        // mask/softmax below touch registers only, so waiting for V(s0) here
        // (rather than after the softmax) both frees tile_K for the next K
        // fetch AND keeps the barrier count at two. The K(s0+32) copies are
        // then in flight across the mask, the softmax, the P packing and P*V,
        // instead of across P*V alone.
        //
        // Measured against the alternative of a third barrier that lets V(s0)
        // and K(s0+32) overlap (`wait_group 1`): that form is NOT better. 32K
        // attention 5,601 ms (this form) vs 5,773 ms (overlapped) vs 5,763 ms
        // (two-barrier baseline); 8K 343 vs 347 vs 350. The two fetches are
        // better serialised than overlapped, so the extra barrier is not paid.
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (s0 + FA2_BC <= win_max) {
            fa2_stage_async(tile_K, k, s0 + FA2_BC, win_max, kh, n_kv_heads, kv_base, tid);
        }

        // ---- causal mask: write -INFINITY into masked scores -------------
        // x[0]/x[1] are row `gid`, x[2]/x[3] are row `gid+8`; both are the
        // same query row, so one boundary (`row_key_max`) serves all four.
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int key0 = s0 + nt * 8 + t4 * 2;
            if (!row_ok || key0 > row_key_max) {
                KQ_C[nt][0] = -INFINITY;
                KQ_C[nt][2] = -INFINITY;
            }
            if (!row_ok || key0 + 1 > row_key_max) {
                KQ_C[nt][1] = -INFINITY;
                KQ_C[nt][3] = -INFINITY;
            }
        }

        // ---- online softmax ----------------------------------------------
        float KQ_max_new[2] = {KQ_max[0], KQ_max[1]};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            KQ_max_new[0] = fmaxf(KQ_max_new[0], KQ_C[nt][0] + FA2_KQ_OFFSET);
            KQ_max_new[0] = fmaxf(KQ_max_new[0], KQ_C[nt][1] + FA2_KQ_OFFSET);
            KQ_max_new[1] = fmaxf(KQ_max_new[1], KQ_C[nt][2] + FA2_KQ_OFFSET);
            KQ_max_new[1] = fmaxf(KQ_max_new[1], KQ_C[nt][3] + FA2_KQ_OFFSET);
        }
        // A KQ row is spread over the four lanes with equal lane/4 (they hold
        // different key columns), so only offsets 2 and 1 are needed.
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            KQ_max_new[c] = fmaxf(KQ_max_new[c],
                                  __shfl_xor_sync(0xffffffffu, KQ_max_new[c], 2));
            KQ_max_new[c] = fmaxf(KQ_max_new[c],
                                  __shfl_xor_sync(0xffffffffu, KQ_max_new[c], 1));
        }

        float KQ_rowsum_add[2] = {0.0f, 0.0f};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float p = __expf(KQ_C[nt][l] - KQ_max_new[l >> 1]);
                KQ_C[nt][l] = p;
                KQ_rowsum_add[l >> 1] += p;
            }
        }

        // Rescale factor for the previous VKQ, with llama.cpp's FTZ bit-trick:
        // multiplying the float's bit pattern by 0 or 1 zeroes a denormal
        // exp() without a branch. KQ_max_diff <= 0 always.
        float KQ_max_scale[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const float diff = KQ_max[c] - KQ_max_new[c];
            float sc = __expf(diff);
            unsigned bits = *reinterpret_cast<unsigned*>(&sc);
            bits *= (diff >= FA2_FTZ_THRESH) ? 1u : 0u;
            KQ_max_scale[c] = *reinterpret_cast<const float*>(&bits);
            KQ_max[c] = KQ_max_new[c];
            KQ_rowsum[c] = KQ_max_scale[c] * KQ_rowsum[c] + KQ_rowsum_add[c];
        }

        // In-place half2 rescale of the fp16 P*V accumulator.
        {
            const __half2 sc0 = __float2half2_rn(KQ_max_scale[0]);
            const __half2 sc1 = __float2half2_rn(KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                __half2 v0 = *reinterpret_cast<const __half2*>(&VKQ_C[i][0]);
                __half2 v1 = *reinterpret_cast<const __half2*>(&VKQ_C[i][1]);
                v0 = __hmul2(v0, sc0);
                v1 = __hmul2(v1, sc1);
                VKQ_C[i][0] = *reinterpret_cast<const unsigned*>(&v0);
                VKQ_C[i][1] = *reinterpret_cast<const unsigned*>(&v1);
            }
        }

        // ---- P fragments: plain make_half2 pairing, no transpose ----------
        // A_PV = get_half2(KQ_C): k-step kk uses KQ n-tiles 2kk and 2kk+1.
        unsigned P[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            P[kk][0] = fa2_pk2f(KQ_C[kk * 2][0], KQ_C[kk * 2][1]);
            P[kk][1] = fa2_pk2f(KQ_C[kk * 2][2], KQ_C[kk * 2][3]);
            P[kk][2] = fa2_pk2f(KQ_C[kk * 2 + 1][0], KQ_C[kk * 2 + 1][1]);
            P[kk][3] = fa2_pk2f(KQ_C[kk * 2 + 1][2], KQ_C[kk * 2 + 1][3]);
        }

        // ---- P*V: VKQ_C += P . V, fp16 accumulate -------------------------
#pragma unroll
        for (int kc = 0; kc < 2; ++kc) {
#pragma unroll
            for (int dc = 0; dc < 16; ++dc) {
                const int vrow = kc * 16 + (lane & 7) + ((lane & 16) ? 8 : 0);
                const int vcol = dc * 8 + ((lane & 8) ? 4 : 0);
                unsigned r[4];
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                    : "r"(smem_addr(tile_V + fa2_swz(vrow, vcol))));
                // r0/r2 = b0/b1 of dv n-tile 2*dc, r1/r3 = those of 2*dc+1.
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                    : "+r"(VKQ_C[dc * 2][0]), "+r"(VKQ_C[dc * 2][1])
                    : "r"(P[kc][0]), "r"(P[kc][1]), "r"(P[kc][2]), "r"(P[kc][3]),
                      "r"(r[0]), "r"(r[2]));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                    : "+r"(VKQ_C[dc * 2 + 1][0]), "+r"(VKQ_C[dc * 2 + 1][1])
                    : "r"(P[kc][0]), "r"(P[kc][1]), "r"(P[kc][2]), "r"(P[kc][3]),
                      "r"(r[1]), "r"(r[3]));
            }
        }
        // No trailing barrier: the top-of-loop one next iteration covers the
        // tile_V read that P*V just did, and there is no other consumer.
    }

    // The partial rowsums are spread across the four lanes with equal lane/4
    // (each holds a quarter of the key columns), so they must be reduced -- but
    // only ONCE, here, not per key tile: the per-tile rescale factor is the
    // same for all four lanes (KQ_max is already reduced), so each lane's
    // running partial stays a consistent fraction of the whole. This is what
    // llama.cpp does at `fattn-mma-f16.cuh:1370-1392` (offset_first = 2,
    // offset_last = 1 for cols_per_warp != 8), and it costs 2 shuffles for the
    // whole kernel rather than 2 per key tile.
    //
    // Omitting it is not a subtle error: the output comes out exactly 4x too
    // large for multi-key rows, and +-inf for a single-key row (three of the
    // four lanes divide by a zero partial).
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        KQ_rowsum[c] += __shfl_xor_sync(0xffffffffu, KQ_rowsum[c], 2);
        KQ_rowsum[c] += __shfl_xor_sync(0xffffffffu, KQ_rowsum[c], 1);
    }

    // ---- epilogue: divide by rowsum, write fp32 ---------------------------
    // Guarded on `row_ok` rather than zero-filling the invalid rows. `rows` is
    // `min(8, n_tokens - t0)`, so `gid >= rows` implies `t0 + gid >= n_tokens`:
    // those query rows do not exist in `out` at all and writing them would be an
    // out-of-bounds store past the end of the buffer on any prefill whose token
    // count is not a multiple of 8 (the last block of a 59-token prompt, say).
    // The old kernel skips them for the same reason. Rows inside the block that
    // are past `rows` still never contribute: their Q is zeroed and every key
    // is masked to -INFINITY, so they cannot leak into a valid row either.
    if (row_ok) {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const size_t o0 = ((size_t)(t0 + gid) * n_q_heads + h0) * FA2_HD
                              + (size_t)i * 8 + t4 * 2;
            const size_t o1 = o0 + FA2_HD;   // the second head of this warp
            const float2 f0 =
                __half22float2(*reinterpret_cast<const __half2*>(&VKQ_C[i][0]));
            const float2 f1 =
                __half22float2(*reinterpret_cast<const __half2*>(&VKQ_C[i][1]));
            *reinterpret_cast<float2*>(out + o0) =
                make_float2(f0.x / KQ_rowsum[0], f0.y / KQ_rowsum[0]);
            *reinterpret_cast<float2*>(out + o1) =
                make_float2(f1.x / KQ_rowsum[1], f1.y / KQ_rowsum[1]);
        }
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet per-head gating, from Qwen3_5GatedDeltaNet.forward:
//
//   beta  = sigmoid(b)
//   g     = -exp(A_log) * softplus(a + dt_bias)
//   decay = exp(g)
//
// `g` is already negative, so `decay` lands in (0, 1). softplus is guarded for
// large positive input where exp would overflow.
// ---------------------------------------------------------------------------
extern "C" __global__ void delta_gate_kernel(const float* __restrict__ a,
                                             const float* __restrict__ b,
                                             const float* __restrict__ a_log,
                                             const float* __restrict__ dt_bias,
                                             float* __restrict__ decay,
                                             float* __restrict__ beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i] + dt_bias[i];
    const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
    decay[i] = __expf(-__expf(a_log[i]) * sp);
    beta[i] = 1.0f / (1.0f + __expf(-b[i]));
}

// ---------------------------------------------------------------------------
// Deinterleave per-head halves of a fused projection output.
//
// `q_proj` emits [n_heads, 2*head_dim] and the reference does
// `view(..., n_heads, 2*head_dim)` then `chunk(2, dim=-1)`. So for head h the
// query lives at [h*2*hd, h*2*hd + hd) and the gate at [h*2*hd + hd, ...).
// This gathers one of the two halves into a compact [n_heads, head_dim] block.
// ---------------------------------------------------------------------------
extern "C" __global__ void deinterleave_heads_kernel(const float* __restrict__ src,
                                                     float* __restrict__ dst,
                                                     int n_heads, int head_dim,
                                                     int src_off) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_heads * head_dim;
    if (idx >= total) return;
    const int h = idx / head_dim;
    const int d = idx % head_dim;
    dst[idx] = src[h * 2 * head_dim + src_off + d];
}

// ---------------------------------------------------------------------------
// Decode attention with GQA: one query token against `n_keys` cached keys.
//
// `q` is [n_q_heads, head_dim]; `k_cache`/`v_cache` are
// [max_seq, n_kv_heads, head_dim]. One block per query head, one thread per
// channel, so the head-dim reduction is a block reduction.
//
// This is the decode path: it costs O(n_keys) per token, and the KV cache read
// (2 * n_kv_heads * head_dim * 4 bytes per key) stays tiny next to the 17.6 GB
// of weights streamed per token.
// ---------------------------------------------------------------------------
extern "C" __global__ void attn_decode_kernel(const float* __restrict__ q,
                                              const __half* __restrict__ k_cache,
                                              const __half* __restrict__ v_cache,
                                              float* __restrict__ out, int n_keys,
                                              int n_q_heads, int n_kv_heads, int head_dim,
                                              float scale, int base, int q_off) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int group = n_q_heads / n_kv_heads;
    const int kh = h / group;

    const bool active = d < head_dim;
    const float qv = active ? q[q_off + h * head_dim + d] : 0.0f;

    // Online softmax, for the same reason as `attn_decode_multi_kernel`: an
    // `n_keys`-sized score buffer cannot survive a long context.
    float mx = -INFINITY, sum = 0.0f, acc = 0.0f;
    for (int s = 0; s < n_keys; ++s) {
        const float kk =
            active ? __half2float(k_cache[base + ((size_t)s * n_kv_heads + kh) * head_dim + d]) : 0.0f;
        const float dot = block_reduce_sum(qv * kk) * scale;
        const float m_new = fmaxf(mx, dot);
        const float corr = __expf(mx - m_new);
        const float p = __expf(dot - m_new);
        sum = sum * corr + p;
        acc = fmaf(p,
                   active ? __half2float(v_cache[base + ((size_t)s * n_kv_heads + kh) * head_dim + d]) : 0.0f,
                   acc * corr);
        mx = m_new;
    }

    if (active) out[q_off + h * head_dim + d] = (sum > 0.0f) ? acc / sum : 0.0f;
}

// Append one token's k/v row into the cache at position `pos`.
extern "C" __global__ void kv_cache_append_kernel(const float* __restrict__ k,
                                                  const float* __restrict__ v,
                                                  __half* __restrict__ k_cache,
                                                  __half* __restrict__ v_cache, int pos,
                                                  int n_kv_heads, int head_dim, int base, int src_off) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_kv_heads * head_dim;
    if (i >= n) return;
    k_cache[base + (size_t)pos * n + i] = __float2half(k[src_off + i]);
    v_cache[base + (size_t)pos * n + i] = __float2half(v[src_off + i]);
}

// ---------------------------------------------------------------------------
// Embedding gather
// ---------------------------------------------------------------------------
// One embedding row, bf16 table -> fp32 activation. This is a row gather, not
// a full table read: ~10 KB per token out of a 2.5 GB table, which is why
// `embed_tokens` is excluded from the per-token traffic budget in
// docs/PHYSICS.md.
extern "C" __global__ void embed_gather_kernel(const uint16_t* __restrict__ table,
                                               int token, float* __restrict__ out,
                                               int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    out[i] = bf16_to_float(table[(size_t)token * hidden + i]);
}

// ---------------------------------------------------------------------------
// Greedy argmax
// ---------------------------------------------------------------------------
// Single block, so no atomics and no second pass. Ties resolve to the lowest
// index, matching `torch.argmax`.
extern "C" __global__ void argmax_kernel(const float* __restrict__ x, int n,
                                         int* __restrict__ out_idx) {
    __shared__ float best_val[256];
    __shared__ int best_idx[256];
    const int tid = threadIdx.x;

    float bv = -INFINITY;
    int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    best_val[tid] = bv;
    best_idx[tid] = bi;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && best_val[tid + s] > best_val[tid]) {
            best_val[tid] = best_val[tid + s];
            best_idx[tid] = best_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out_idx[0] = best_idx[0];
}

// ---------------------------------------------------------------------------
// Batched-prefill variants.
//
// The decode path runs one token at a time, so every kernel above is written
// for a single row. Prompt processing needs the same maths over T rows at
// once, otherwise prefill costs exactly what decode costs per token -- which
// is what made TTFT ~85x worse than llama.cpp. Only the recurrence and the
// causal attention are genuinely sequential in T; everything else is
// embarrassingly parallel across rows.
// ---------------------------------------------------------------------------

// x is [T, row_stride]; normalise `vectors` independent n-element vectors per
// row starting at `offset`.
extern "C" __global__ void l2norm_scale_batched_kernel(float* __restrict__ x, int row_stride,
                                                       int offset, int n, float scale,
                                                       float eps) {
    float* __restrict__ xr = x + (size_t)blockIdx.y * row_stride + offset +
                             (size_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        ss = fmaf(v, v, ss);
    }
    const float mul = scale * rsqrtf(block_reduce_sum(ss) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) xr[i] *= mul;
}

// a/b/decay/beta are [T, n].
extern "C" __global__ void delta_gate_batched_kernel(const float* __restrict__ a,
                                                     const float* __restrict__ b,
                                                     const float* __restrict__ a_log,
                                                     const float* __restrict__ dt_bias,
                                                     float* __restrict__ decay,
                                                     float* __restrict__ beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t off = (size_t)blockIdx.y * n + i;
    const float x = a[off] + dt_bias[i];
    const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
    decay[off] = __expf(-__expf(a_log[i]) * sp);
    beta[off] = 1.0f / (1.0f + __expf(-b[off]));
}

// Causal depthwise conv over T rows. `hist` carries the `k-1` tokens before
// the chunk in, and the last `k-1` tokens back out, so a prompt can be
// processed in pieces without losing continuity.
extern "C" __global__ void conv1d_prefill_silu_kernel(const float* __restrict__ x,
                                                      const float* __restrict__ w,
                                                      float* __restrict__ hist,
                                                      float* __restrict__ y, int channels,
                                                      int T, int base) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* __restrict__ wc = w + (size_t)c * 4;
    float* __restrict__ h = hist + base + (size_t)c * 3;
    float h0 = h[0], h1 = h[1], h2 = h[2];
    for (int t = 0; t < T; ++t) {
        const float xc = x[(size_t)t * channels + c];
        const float acc = fmaf(wc[0], h0, fmaf(wc[1], h1, fmaf(wc[2], h2, wc[3] * xc)));
        y[(size_t)t * channels + c] = silu_f(acc);
        h0 = h1;
        h1 = h2;
        h2 = xc;
    }
    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
}

// RoPE over T rows. cos/sin are [T, half].
extern "C" __global__ void rope_neox_batched_kernel(float* __restrict__ q,
                                                    float* __restrict__ k,
                                                    const float* __restrict__ cs,
                                                    const float* __restrict__ sn, int n_q_heads,
                                                    int n_k_heads, int head_dim, int half) {
    const int t = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (n_q_heads + n_k_heads) * half;
    if (idx >= total) return;

    float* __restrict__ base;
    int i;
    if (idx < n_q_heads * half) {
        base = q + (size_t)t * n_q_heads * head_dim + (size_t)(idx / half) * head_dim;
        i = idx % half;
    } else {
        const int r = idx - n_q_heads * half;
        base = k + (size_t)t * n_k_heads * head_dim + (size_t)(r / half) * head_dim;
        i = r % half;
    }
    const size_t so = (size_t)t * half + i;
    const float c = cs[so], s = sn[so];
    const float a = base[i], b = base[i + half];
    base[i] = fmaf(-b, s, a * c);
    base[i + half] = fmaf(a, s, b * c);
}

// k/v are [T, n_kv_heads * head_dim]; appended starting at `start_pos`.
extern "C" __global__ void kv_cache_append_batched_kernel(const float* __restrict__ k,
                                                          const float* __restrict__ v,
                                                          __half* __restrict__ k_cache,
                                                          __half* __restrict__ v_cache,
                                                          int start_pos, int n_kv_heads,
                                                          int head_dim, int base) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_kv_heads * head_dim;
    if (i >= n) return;
    const int t = blockIdx.y;
    k_cache[base + (size_t)(start_pos + t) * n + i] = __float2half(k[(size_t)t * n + i]);
    v_cache[base + (size_t)(start_pos + t) * n + i] = __float2half(v[(size_t)t * n + i]);
}

// The Gated DeltaNet recurrence is the one part of prefill that cannot be
// parallelised across T, so it is run as a T-iteration loop inside a single
// launch with the recurrent state held in shared memory. Launching one kernel
// per token instead would cost ~30us of launch plus a 3.1 MB state round trip
// per layer per token.
extern "C" __global__ void gated_delta_rule_chunk_kernel(
    const float* __restrict__ qkv, int q_off, int k_off, int v_off, int row_stride,
    const float* __restrict__ decay, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ out, int T, int n_v_heads, int n_k_heads,
    int group, int base) {
    constexpr int D = 128;
    // Two threads per state column: the pair (2j, 2j+1) splits column `j`'s 128
    // rows between them, so each thread holds 64 floats. The one-thread form
    // needed 255 registers -- the hardware maximum -- and still spilled 140
    // stores to local memory (round 70), and spill traffic is global memory.
    // Every removal of inner-loop traffic from this kernel so far paid 9-17%.
    // The pair's partial sums are joined by one shfl_xor; adjacent lanes, so the
    // pair is always inside one warp.
    constexpr int HW = D / 2;
    const int hv = blockIdx.x;
    const int b = blockIdx.y;
    const int j = threadIdx.x >> 1;
    const int h = (int)(threadIdx.x & 1u);
    const int hbase = h * HW;
    const int kh = hv / group;

    // Thread `j` reads and writes only column `j` of the state, and the column is
    // exactly D floats -- so it fits in registers and never needs shared memory.
    // The old form put S in 66 KB of shared and touched it twice per fma (one
    // load, one store) in both hot loops; the kernel then ran at 0.6% of fp32
    // peak (round 66) with no per-chunk overhead to trim (round 68). Holding the
    // column in registers removes all of that traffic from the inner loops.
    // `sk` stays shared: it is the one input every column needs.
    __shared__ float sk[D];
    // `qh[i]` was a *global* load inside the second hot loop -- 128 of them per
    // thread per token, and that loop is half the kernel's cost. Every thread in
    // the block reads the same address, so it belongs in shared memory beside
    // `sk`, loaded once per token the same way.
    __shared__ float sq[D];

    float* __restrict__ sh = state + base + ((size_t)b * n_v_heads + hv) * D * D;
    float Sc[HW];
#pragma unroll
    for (int m = 0; m < HW; ++m) Sc[m] = sh[(size_t)(hbase + m) * D + j];

    for (int t = 0; t < T; ++t) {
        const float* __restrict__ row = qkv + (size_t)(b * T + t) * row_stride;
        const float* __restrict__ qh = row + q_off + (size_t)kh * D;
        const float* __restrict__ khp = row + k_off + (size_t)kh * D;
        const float* __restrict__ vh = row + v_off + (size_t)hv * D;

        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            sk[i] = khp[i];
            sq[i] = qh[i];
        }
        __syncthreads();

        const float dec = decay[(size_t)(b * T + t) * n_v_heads + hv];
        const float bet = beta[(size_t)(b * T + t) * n_v_heads + hv];
        const float vj = vh[j];

        // Each of these two loops carried one 128-long serial fma chain, and the
        // whole kernel runs at 0.64% of fp32 peak (round 66) -- so the chain, not
        // the throughput, is what the block waits on. Four independent partial
        // sums cut the chain to 32 and give the scheduler work to interleave.
        // Only the summation order changes; the result is equal to fp32 rounding.
        float kv0 = 0.0f, kv1 = 0.0f, kv2 = 0.0f, kv3 = 0.0f;
#pragma unroll
        for (int m = 0; m < HW; m += 4) {
            const int i = hbase + m;
            const float s0 = Sc[m] * dec;
            const float s1 = Sc[m + 1] * dec;
            const float s2 = Sc[m + 2] * dec;
            const float s3 = Sc[m + 3] * dec;
            Sc[m] = s0;
            Sc[m + 1] = s1;
            Sc[m + 2] = s2;
            Sc[m + 3] = s3;
            kv0 = fmaf(s0, sk[i], kv0);
            kv1 = fmaf(s1, sk[i + 1], kv1);
            kv2 = fmaf(s2, sk[i + 2], kv2);
            kv3 = fmaf(s3, sk[i + 3], kv3);
        }
        float kv = (kv0 + kv1) + (kv2 + kv3);
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        const float delta = (vj - kv) * bet;

        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
#pragma unroll
        for (int m = 0; m < HW; m += 4) {
            const int i = hbase + m;
            const float s0 = fmaf(sk[i], delta, Sc[m]);
            const float s1 = fmaf(sk[i + 1], delta, Sc[m + 1]);
            const float s2 = fmaf(sk[i + 2], delta, Sc[m + 2]);
            const float s3 = fmaf(sk[i + 3], delta, Sc[m + 3]);
            Sc[m] = s0;
            Sc[m + 1] = s1;
            Sc[m + 2] = s2;
            Sc[m + 3] = s3;
            o0 = fmaf(s0, sq[i], o0);
            o1 = fmaf(s1, sq[i + 1], o1);
            o2 = fmaf(s2, sq[i + 2], o2);
            o3 = fmaf(s3, sq[i + 3], o3);
        }
        float o = (o0 + o1) + (o2 + o3);
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        if (h == 0) out[(size_t)(b * T + t) * n_v_heads * D + (size_t)hv * D + j] = o;
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < HW; ++m) sh[(size_t)(hbase + m) * D + j] = Sc[m];
}

// Batched `deinterleave_heads`: src is [T, n_heads * 2 * head_dim].
extern "C" __global__ void deinterleave_heads_batched_kernel(const float* __restrict__ src,
                                                             float* __restrict__ dst,
                                                             int n_heads, int head_dim,
                                                             int src_off) {
    const int t = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_heads * head_dim;
    if (idx >= total) return;
    const int h = idx / head_dim;
    const int d = idx % head_dim;
    dst[(size_t)t * total + idx] =
        src[(size_t)t * n_heads * 2 * head_dim + h * 2 * head_dim + src_off + d];
}

// Gather T token embeddings in one launch. `tokens` is a device array of T
// int32 ids; `out` is [T, hidden] fp32 (bf16 bits widened by a shift).
extern "C" __global__ void embed_gather_batched_kernel(const uint16_t* __restrict__ table,
                                                       const int* __restrict__ tokens,
                                                       float* __restrict__ out, int hidden) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    const uint16_t bits = table[(size_t)tokens[t] * hidden + i];
    out[(size_t)t * hidden + i] = __uint_as_float((uint32_t)bits << 16);
}

// Copy row `t-1` of an `[t, n]` buffer to the front of `dst`. Prefill only
// needs logits for the final prompt token, and running lm_head (715 MB of
// weights) over every prompt row would cost a full extra pass.
extern "C" __global__ void copy_last_row_kernel(const float* __restrict__ src,
                                                float* __restrict__ dst, int t, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[(size_t)(t - 1) * n + i];
}

// Gather `rows` rows starting at `row0` of an `[row0+rows, n]` buffer into a
// dense `[rows, n]` one. The perplexity path has to score a window's rows with
// the GEMM, which takes a whole buffer rather than a row offset.
extern "C" __global__ void copy_rows_kernel(const float* __restrict__ src,
                                            float* __restrict__ dst,
                                            int row0, int rows, int n) {
    const int total = rows * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        const int r = i / n;
        const int c = i - r * n;
        dst[i] = src[(size_t)(row0 + r) * n + c];
    }
}

// Concatenate two `n`-vectors into `dst`: dst[0..n] = a, dst[n..2n] = b.
//
// The MTP head feeds `mtp.fc` with `concat(norm(embed(t+1)), norm(h_t))`, and
// the RMSNorm ops write to a whole buffer rather than an offset, so the two
// halves are staged separately and joined here.
extern "C" __global__ void concat2_kernel(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = a[i];
    dst[n + i] = b[i];
}

// ---------------------------------------------------------------------------
// Multi-sequence decode variants
//
// These are the `_multi` counterparts of the four decode kernels above. Each
// takes `gridDim.y = sequence index` and reads that sequence's position from a
// device array, so one launch serves the whole batch instead of one launch per
// sequence. The per-sequence state lives at `seq * base_stride` inside the
// shared state allocation (see `LayerState`, which is laid out sequence-major).
//
// They are deliberately additive: the single-sequence kernels are untouched,
// and nothing calls these yet.
// ---------------------------------------------------------------------------

extern "C" __global__ void conv1d_step_silu_multi_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ hist,
    float* __restrict__ y, int channels, int base_stride) {
    const int s = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const size_t xb = (size_t)s * channels;
    const size_t hb = (size_t)s * base_stride;
    const float* __restrict__ wc = w + (size_t)c * 4;
    float* __restrict__ h = hist + hb + (size_t)c * 3;
    const float xc = x[xb + c];
    const float acc = fmaf(wc[0], h[0], fmaf(wc[1], h[1], fmaf(wc[2], h[2], wc[3] * xc)));
    y[xb + c] = silu_f(acc);
    h[0] = h[1];
    h[1] = h[2];
    h[2] = xc;
}

// One token of the Gated DeltaNet recurrence for every sequence in the batch.
// `qkv` is `[n_seq, row_stride]`; the q/k/v offsets inside a row are the same
// for all sequences, so only the row base moves.
extern "C" __global__ void gated_delta_rule_step_multi_kernel(
    const float* __restrict__ qkv, int q_off, int k_off, int v_off, int row_stride,
    const float* __restrict__ decay, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ out, int n_v_heads, int n_k_heads,
    int group, int base_stride) {
    constexpr int D = 128;
    const int hv = blockIdx.x;
    const int s = blockIdx.y;
    const int j = threadIdx.x;
    const int kh = hv / group;

    __shared__ float S[D][D + 1];
    __shared__ float sk[D];

    const float* __restrict__ row = qkv + (size_t)s * row_stride;
    const float* __restrict__ qh = row + q_off + (size_t)kh * D;
    const float* __restrict__ khp = row + k_off + (size_t)kh * D;
    const float* __restrict__ vh = row + v_off + (size_t)hv * D;
    float* __restrict__ sh =
        state + (size_t)s * base_stride + (size_t)hv * D * D;
    const float dec = decay[(size_t)s * n_v_heads + hv];
    const float bet = beta[(size_t)s * n_v_heads + hv];

    for (int i = threadIdx.x; i < D * D; i += blockDim.x) S[i / D][i % D] = sh[i];
    __syncthreads();
    sk[j] = khp[j];
    __syncthreads();

    for (int i = 0; i < D; ++i) {
        const float s_ij = S[i][j] * dec;
        S[i][j] = s_ij;
    }
    __syncthreads();

    float delta = 0.0f;
    for (int i = 0; i < D; ++i) delta = fmaf(S[i][j], sk[i], delta);
    delta = (vh[j] - delta) * bet;
    for (int i = 0; i < D; ++i) S[i][j] = fmaf(sk[i], delta, S[i][j]);
    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < D; ++i) acc = fmaf(S[i][j], qh[i], acc);
    out[(size_t)s * n_v_heads * D + (size_t)hv * D + j] = acc;

    for (int i = threadIdx.x; i < D * D; i += blockDim.x) sh[i] = S[i / D][i % D];
}

// Append each sequence's k/v row at its own position.
extern "C" __global__ void kv_cache_append_multi_kernel(
    const float* __restrict__ k, const float* __restrict__ v, __half* __restrict__ k_cache,
    __half* __restrict__ v_cache, const int* __restrict__ positions, int n_kv_heads,
    int head_dim, int base_stride) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_kv_heads * head_dim;
    if (i >= n) return;
    const size_t src = (size_t)s * n;
    const size_t dst = (size_t)s * base_stride + (size_t)positions[s] * n;
    k_cache[dst + i] = __float2half(k[src + i]);
    v_cache[dst + i] = __float2half(v[src + i]);
}

// Decode attention for every sequence, warp-parallel over the keys.
//
// The previous shape gave every thread one `head_dim` lane and walked the keys
// serially, calling `block_reduce_sum` once per key. That puts two
// `__syncthreads()` inside the loop, so the entire block advances exactly one
// key per barrier and no two keys are ever in flight. Measured at 32K keys it
// was 847 ms per token -- about 1.2 tok/s, against a memory bound nearer 100 ms.
//
// Here a warp owns a strided subset of the keys: lane l holds dims
// [8l, 8l+8), a key's score is a five-step `__shfl_xor` reduction inside the
// warp, and the eight warps' online-softmax partials are merged once at the
// end. There is no barrier inside the loop, and the warps' key ranges are
// interleaved so their loads cover adjacent lines.
//
// Requires `head_dim == 256` (32 lanes x 8 dims). The host dispatches
// `attn_decode_multi_serial_kernel` for any other head width; that one is also
// the independent implementation the batch-parity gate compares against.
extern "C" __global__ void attn_decode_multi_kernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, float* __restrict__ out,
    const int* __restrict__ positions, int n_q_heads, int n_kv_heads, int head_dim,
    float scale, int base_stride) {
    constexpr int DPL = 8;   // dims per lane, for the q.k dot product
    constexpr int DV  = 4;   // dims per lane, for the v accumulation (split-D)
    // Warps per block, i.e. how many key streams are in flight per SM. The
    // grid is only (n_q_heads, n_seq) = 24 blocks for a single sequence, so at
    // NW = 8 a 256-thread block left just 4 warps resident per SM and the
    // kernel ran at 328 GB/s. Raising NW raises the resident warps without
    // changing the grid: 32 warps x 24 blocks is 4x the in-flight work for the
    // same traffic. See `NW_MAX` for the shared-memory ceiling.
    constexpr int NW = 32;   // warps per block
    const int h = blockIdx.x;
    // Split-D: blockIdx.y packs (sequence, dim half). Both halves compute the
    // same softmax weights over ALL keys -- the dot product needs every dim --
    // but each accumulates only its own half of v. No cross-block merge.
    const int s = blockIdx.y >> 1;
    const int half = blockIdx.y & 1;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d0 = lane * DPL;
    const int group = n_q_heads / n_kv_heads;
    const int kh = h / group;
    const int n_keys = positions[s];

    const size_t qb = (size_t)s * n_q_heads * head_dim;
    const size_t cb = (size_t)s * base_stride;

    float qv[DPL];
#pragma unroll
    for (int j = 0; j < DPL; ++j) qv[j] = q[qb + (size_t)h * head_dim + d0 + j];

    const int d0v = half * (head_dim / 2) + lane * DV;
    float mx = -INFINITY, sum = 0.0f;
    float acc[DV];
#pragma unroll
    for (int j = 0; j < DV; ++j) acc[j] = 0.0f;

    for (int t = warp; t < n_keys; t += NW) {
        const size_t off = cb + ((size_t)t * n_kv_heads + kh) * head_dim + d0;
        const __half* kp = k_cache + off;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < DPL; ++j) dot = fmaf(qv[j], __half2float(kp[j]), dot);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        dot *= scale;

        const float m_new = fmaxf(mx, dot);
        const float corr = __expf(mx - m_new);
        const float p = __expf(dot - m_new);
        sum = sum * corr + p;
        const __half* vp = v_cache + cb + ((size_t)t * n_kv_heads + kh) * head_dim + d0v;
#pragma unroll
        for (int j = 0; j < DV; ++j) acc[j] = fmaf(p, __half2float(vp[j]), acc[j] * corr);
        mx = m_new;
    }

    // One merge of the NW per-warp partials. `sm_acc` is sized for the 256-wide
    // head this kernel requires, so it is 8 KB, not one entry per key.
    __shared__ float sm_m[NW], sm_l[NW], sm_acc[NW][128];
    if (lane == 0) {
        sm_m[warp] = mx;
        sm_l[warp] = sum;
    }
#pragma unroll
    for (int j = 0; j < DV; ++j) sm_acc[warp][lane * DV + j] = acc[j];
    __syncthreads();

    const int d = threadIdx.x;
    if (d < head_dim / 2) {
        float m = -INFINITY, l = 0.0f, a = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            const float pm = sm_m[w];
            if (pm == -INFINITY) continue;   // that warp saw no keys
            const float m_new = fmaxf(m, pm);
            const float corr = __expf(m - m_new);
            const float wt = __expf(pm - m_new);
            l = l * corr + sm_l[w] * wt;
            a = a * corr + sm_acc[w][d] * wt;
            m = m_new;
        }
        // An empty cache leaves `l` at zero; the old form returned 0 here
        // because its accumulation loop never ran, so keep that rather than
        // emitting NaN.
        out[qb + (size_t)h * head_dim + half * (head_dim / 2) + d] = (l > 0.0f) ? a / l : 0.0f;
    }
}

// Serial reference: one thread per `head_dim` lane, one block reduction per
// key. Kept for head widths the warp kernel above does not cover, and as the
// independent implementation the batch-parity gate compares against.
//
// The score for each key used to be materialised in dynamic shared memory --
// one entry per position in the context -- so the shared-memory request grew
// with the context window and the launch failed outright once it passed the
// 48 KB a launch can request (12,288 keys). Streaming the keys through the
// online-softmax recurrence keeps the running state in two registers instead,
// so the context length no longer affects shared memory at all.
extern "C" __global__ void attn_decode_multi_serial_kernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, float* __restrict__ out,
    const int* __restrict__ positions, int n_q_heads, int n_kv_heads, int head_dim,
    float scale, int base_stride) {
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int d = threadIdx.x;
    const int group = n_q_heads / n_kv_heads;
    const int kh = h / group;
    const int n_keys = positions[s];

    const size_t qb = (size_t)s * n_q_heads * head_dim;
    const size_t cb = (size_t)s * base_stride;
    const bool active = d < head_dim;
    const float qv = active ? q[qb + h * head_dim + d] : 0.0f;

    float mx = -INFINITY, sum = 0.0f, acc = 0.0f;
    for (int i = 0; i < n_keys; ++i) {
        const float kk =
            active ? __half2float(k_cache[cb + ((size_t)i * n_kv_heads + kh) * head_dim + d]) : 0.0f;
        const float dot = block_reduce_sum(qv * kk) * scale;
        const float m_new = fmaxf(mx, dot);
        const float corr = __expf(mx - m_new);
        const float p = __expf(dot - m_new);
        sum = sum * corr + p;
        acc = fmaf(p,
                   active ? __half2float(v_cache[cb + ((size_t)i * n_kv_heads + kh) * head_dim + d]) : 0.0f,
                   acc * corr);
        mx = m_new;
    }

    // An empty cache leaves `sum` at zero; the old form returned 0 here because
    // its accumulation loop never ran, so keep that rather than emitting NaN.
    if (active) out[qb + h * head_dim + d] = (sum > 0.0f) ? acc / sum : 0.0f;
}

// Per-sequence argmax: `gridDim.y` selects the row of a `[n_seq, n]` logits
// buffer, so the batched decode needs one launch rather than one per sequence.
extern "C" __global__ void argmax_multi_kernel(const float* __restrict__ x, int n,
                                               int* __restrict__ out_idx) {
    __shared__ float best_val[256];
    __shared__ int best_idx[256];
    const int tid = threadIdx.x;
    const float* __restrict__ row = x + (size_t)blockIdx.y * n;

    float bv = -INFINITY;
    int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = row[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    best_val[tid] = bv;
    best_idx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && best_val[tid + s] > best_val[tid]) {
            best_val[tid] = best_val[tid + s];
            best_idx[tid] = best_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out_idx[blockIdx.y] = best_idx[0];
}
