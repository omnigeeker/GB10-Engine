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
extern "C" __global__ void attn_prefill_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
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
            ? k[(size_t)kv_base + ((size_t)s * n_kv_heads + kh) * head_dim + d] : 0.0f;
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
            acc = fmaf(p, v[(size_t)kv_base + ((size_t)s * n_kv_heads + kh) * head_dim + d], acc);
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
#define PREFILL_BQ 8
#define PREFILL_BK 16

extern "C" __global__ void attn_prefill_tiled_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    float scale, int start, int kv_base) {
    extern __shared__ float smem[];
    const int HD = head_dim;
    float* Qs = smem;                              // BQ * HD
    float* Ks = Qs + PREFILL_BQ * HD;              // BK * HD
    float* Vs = Ks + PREFILL_BK * HD;              // BK * HD
    float* S = Vs + PREFILL_BK * HD;               // BQ * BK  (running p, then probabilities)
    float* red = S + PREFILL_BQ * PREFILL_BK;      // 3 * BQ  (m, l, correction)

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
        Qs[idx] = (i < rows) ? q[((size_t)(t0 + i) * n_q_heads + h) * HD + d] : 0.0f;
    }
    if (tid < PREFILL_BQ) {
        red[tid] = -INFINITY;                  // m
        red[PREFILL_BQ + tid] = 0.0f;          // l
    }

    // BQ accumulators in registers: thread `tid` owns output dim `tid`.
    float acc[PREFILL_BQ];
#pragma unroll
    for (int i = 0; i < PREFILL_BQ; ++i) acc[i] = 0.0f;

    __syncthreads();

    // Highest key index any row in this block may attend to.
    const int win_max = start + t0 + rows - 1;

    for (int s0 = 0; s0 <= win_max; s0 += PREFILL_BK) {
        // Stage this key tile. Keys past `win_max` are irrelevant to every row
        // here and get zero-filled; the causal mask below excludes them anyway.
        for (int idx = tid; idx < PREFILL_BK * HD; idx += nt) {
            const int j = idx / HD, d = idx % HD;
            const int s = s0 + j;
            if (s <= win_max) {
                const size_t off = (size_t)kv_base + ((size_t)s * n_kv_heads + kh) * HD + d;
                Ks[idx] = k[off];
                Vs[idx] = v[off];
            } else {
                Ks[idx] = 0.0f;
                Vs[idx] = 0.0f;
            }
        }
        __syncthreads();

        // S[i][j] = Qs[i] . Ks[j] * scale. BQ*BK pairs over `nt` threads: two
        // threads per pair, each covering half of head_dim, combined with a
        // lane-adjacent shuffle.
        const int half = HD / 2;
        for (int p = tid >> 1; p < PREFILL_BQ * PREFILL_BK; p += nt >> 1) {
            const int i = p / PREFILL_BK, j = p % PREFILL_BK;
            const int sub = tid & 1;
            float dot = 0.0f;
            const float* qrow = Qs + i * HD + sub * half;
            const float* krow = Ks + j * HD + sub * half;
            for (int d = 0; d < half; ++d) dot = fmaf(qrow[d], krow[d], dot);
            dot += __shfl_xor_sync(0xffffffffu, dot, 1);
            if (sub == 0) {
                const int s = s0 + j;
                const bool ok = (i < rows) && (s <= start + t0 + i);
                S[i * PREFILL_BK + j] = ok ? dot * scale : -INFINITY;
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
                a = fmaf(prow[j], Vs[j * HD + tid], a);
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
                                              const float* __restrict__ k_cache,
                                              const float* __restrict__ v_cache,
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
            active ? k_cache[base + ((size_t)s * n_kv_heads + kh) * head_dim + d] : 0.0f;
        const float dot = block_reduce_sum(qv * kk) * scale;
        const float m_new = fmaxf(mx, dot);
        const float corr = __expf(mx - m_new);
        const float p = __expf(dot - m_new);
        sum = sum * corr + p;
        acc = fmaf(p,
                   active ? v_cache[base + ((size_t)s * n_kv_heads + kh) * head_dim + d] : 0.0f,
                   acc * corr);
        mx = m_new;
    }

    if (active) out[q_off + h * head_dim + d] = (sum > 0.0f) ? acc / sum : 0.0f;
}

// Append one token's k/v row into the cache at position `pos`.
extern "C" __global__ void kv_cache_append_kernel(const float* __restrict__ k,
                                                  const float* __restrict__ v,
                                                  float* __restrict__ k_cache,
                                                  float* __restrict__ v_cache, int pos,
                                                  int n_kv_heads, int head_dim, int base, int src_off) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_kv_heads * head_dim;
    if (i >= n) return;
    k_cache[base + (size_t)pos * n + i] = k[src_off + i];
    v_cache[base + (size_t)pos * n + i] = v[src_off + i];
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
                                                          float* __restrict__ k_cache,
                                                          float* __restrict__ v_cache,
                                                          int start_pos, int n_kv_heads,
                                                          int head_dim, int base) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_kv_heads * head_dim;
    if (i >= n) return;
    const int t = blockIdx.y;
    k_cache[base + (size_t)(start_pos + t) * n + i] = k[(size_t)t * n + i];
    v_cache[base + (size_t)(start_pos + t) * n + i] = v[(size_t)t * n + i];
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
    const int hv = blockIdx.x;
    const int b = blockIdx.y;
    const int j = threadIdx.x;
    const int kh = hv / group;

    __shared__ float S[D][D + 1];
    __shared__ float sk[D];

    float* __restrict__ sh = state + base + ((size_t)b * n_v_heads + hv) * D * D;
    for (int i = threadIdx.x; i < D * D; i += blockDim.x) S[i / D][i % D] = sh[i];
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const float* __restrict__ row = qkv + (size_t)(b * T + t) * row_stride;
        const float* __restrict__ qh = row + q_off + (size_t)kh * D;
        const float* __restrict__ khp = row + k_off + (size_t)kh * D;
        const float* __restrict__ vh = row + v_off + (size_t)hv * D;

        for (int i = threadIdx.x; i < D; i += blockDim.x) sk[i] = khp[i];
        __syncthreads();

        const float dec = decay[(size_t)(b * T + t) * n_v_heads + hv];
        const float bet = beta[(size_t)(b * T + t) * n_v_heads + hv];
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
        out[(size_t)(b * T + t) * n_v_heads * D + (size_t)hv * D + j] = o;
        __syncthreads();
    }

    for (int i = threadIdx.x; i < D * D; i += blockDim.x) sh[i] = S[i / D][i % D];
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
    const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ k_cache,
    float* __restrict__ v_cache, const int* __restrict__ positions, int n_kv_heads,
    int head_dim, int base_stride) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_kv_heads * head_dim;
    if (i >= n) return;
    const size_t src = (size_t)s * n;
    const size_t dst = (size_t)s * base_stride + (size_t)positions[s] * n;
    k_cache[dst + i] = k[src + i];
    v_cache[dst + i] = v[src + i];
}

// Decode attention for every sequence.
//
// The score for each key used to be materialised in dynamic shared memory --
// one entry per position in the context -- so the shared-memory request grew
// with the context window and the launch failed outright once it passed the
// 48 KB a launch can request (12,288 keys). Streaming the keys through the
// online-softmax recurrence keeps the running state in two registers instead,
// so the context length no longer affects shared memory at all.
extern "C" __global__ void attn_decode_multi_kernel(
    const float* __restrict__ q, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, float* __restrict__ out,
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
            active ? k_cache[cb + ((size_t)i * n_kv_heads + kh) * head_dim + d] : 0.0f;
        const float dot = block_reduce_sum(qv * kk) * scale;
        const float m_new = fmaxf(mx, dot);
        const float corr = __expf(mx - m_new);
        const float p = __expf(dot - m_new);
        sum = sum * corr + p;
        acc = fmaf(p,
                   active ? v_cache[cb + ((size_t)i * n_kv_heads + kh) * head_dim + d] : 0.0f,
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
