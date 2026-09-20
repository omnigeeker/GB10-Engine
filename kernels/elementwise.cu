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
    float scale) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const int group = n_q_heads / n_kv_heads;
    const int kh = h / group;

    extern __shared__ float scores[];  // n_tokens entries

    const bool active = d < head_dim;
    const float qv = active ? q[((size_t)t * n_q_heads + h) * head_dim + d] : 0.0f;

    // One block reduction per key. O(T) reductions per query: correct and
    // simple, but quadratic in T. Replaced by a tiled kernel in M6.
    for (int s = 0; s <= t; ++s) {
        const float kv_ = active ? k[((size_t)s * n_kv_heads + kh) * head_dim + d] : 0.0f;
        const float dot = block_reduce_sum(qv * kv_) * scale;
        if (d == 0) scores[s] = dot;
        __syncthreads();
    }

    // Every thread redundantly recomputes the softmax (T is small here), which
    // avoids further synchronisation.
    float mx = -INFINITY;
    for (int s = 0; s <= t; ++s) mx = fmaxf(mx, scores[s]);
    float sum = 0.0f;
    for (int s = 0; s <= t; ++s) sum += __expf(scores[s] - mx);
    const float inv = 1.0f / sum;

    if (active) {
        float acc = 0.0f;
        for (int s = 0; s <= t; ++s) {
            const float p = __expf(scores[s] - mx) * inv;
            acc = fmaf(p, v[((size_t)s * n_kv_heads + kh) * head_dim + d], acc);
        }
        out[((size_t)t * n_q_heads + h) * head_dim + d] = acc;
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
                                              float scale, int base) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int group = n_q_heads / n_kv_heads;
    const int kh = h / group;

    extern __shared__ float scores[];  // n_keys entries

    const bool active = d < head_dim;
    const float qv = active ? q[h * head_dim + d] : 0.0f;

    for (int s = 0; s < n_keys; ++s) {
        const float kk =
            active ? k_cache[base + ((size_t)s * n_kv_heads + kh) * head_dim + d] : 0.0f;
        const float dot = block_reduce_sum(qv * kk) * scale;
        if (d == 0) scores[s] = dot;
        __syncthreads();
    }

    float mx = -INFINITY;
    for (int s = 0; s < n_keys; ++s) mx = fmaxf(mx, scores[s]);
    float sum = 0.0f;
    for (int s = 0; s < n_keys; ++s) sum += __expf(scores[s] - mx);
    const float inv = 1.0f / sum;

    if (active) {
        float acc = 0.0f;
        for (int s = 0; s < n_keys; ++s) {
            const float p = __expf(scores[s] - mx) * inv;
            acc = fmaf(p, v_cache[base + ((size_t)s * n_kv_heads + kh) * head_dim + d], acc);
        }
        out[h * head_dim + d] = acc;
    }
}

// Append one token's k/v row into the cache at position `pos`.
extern "C" __global__ void kv_cache_append_kernel(const float* __restrict__ k,
                                                  const float* __restrict__ v,
                                                  float* __restrict__ k_cache,
                                                  float* __restrict__ v_cache, int pos,
                                                  int n_kv_heads, int head_dim, int base) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_kv_heads * head_dim;
    if (i >= n) return;
    k_cache[base + (size_t)pos * n + i] = k[i];
    v_cache[base + (size_t)pos * n + i] = v[i];
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
                                                      int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* __restrict__ wc = w + (size_t)c * 4;
    float* __restrict__ h = hist + (size_t)c * 3;
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
                                                          int head_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_kv_heads * head_dim;
    if (i >= n) return;
    const int t = blockIdx.y;
    k_cache[(size_t)(start_pos + t) * n + i] = k[(size_t)t * n + i];
    v_cache[(size_t)(start_pos + t) * n + i] = v[(size_t)t * n + i];
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
    int group) {
    constexpr int D = 128;
    const int hv = blockIdx.x;
    const int b = blockIdx.y;
    const int j = threadIdx.x;
    const int kh = hv / group;

    __shared__ float S[D][D + 1];
    __shared__ float sk[D];

    float* __restrict__ sh = state + ((size_t)b * n_v_heads + hv) * D * D;
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

// Decode attention for every sequence. `max_keys` sizes the shared score
// buffer; each sequence loops only to its own `positions[s]`.
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

    extern __shared__ float scores[];

    const size_t qb = (size_t)s * n_q_heads * head_dim;
    const size_t cb = (size_t)s * base_stride;
    const bool active = d < head_dim;
    const float qv = active ? q[qb + h * head_dim + d] : 0.0f;

    for (int i = 0; i < n_keys; ++i) {
        const float kk =
            active ? k_cache[cb + ((size_t)i * n_kv_heads + kh) * head_dim + d] : 0.0f;
        const float dot = block_reduce_sum(qv * kk) * scale;
        if (d == 0) scores[i] = dot;
        __syncthreads();
    }

    float mx = -INFINITY;
    for (int i = 0; i < n_keys; ++i) mx = fmaxf(mx, scores[i]);
    float sum = 0.0f;
    for (int i = 0; i < n_keys; ++i) sum += __expf(scores[i] - mx);
    const float inv = 1.0f / sum;

    if (active) {
        float acc = 0.0f;
        for (int i = 0; i < n_keys; ++i) {
            const float p = __expf(scores[i] - mx) * inv;
            acc = fmaf(p, v_cache[cb + ((size_t)i * n_kv_heads + kh) * head_dim + d], acc);
        }
        out[qb + h * head_dim + d] = acc;
    }
}
