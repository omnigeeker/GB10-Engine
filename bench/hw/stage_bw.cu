// Measures the achievable read rate for the GEMM weight-staging access pattern,
// so the 136 GB/s seen in the real kernel can be compared against what this
// *pattern* can actually reach -- not against the contiguous random-access
// figure from bw4.cu, which a warp here does not resemble.
//
// Two patterns, same bytes:
//   A  contiguous: each thread reads consecutive 16 B blocks, grid-strided
//   B  staging:    the real layout -- a warp reads 8 rows that are K/2 bytes
//                  apart, 32 consecutive bytes within each row
//
// Build: nvcc -arch sm_121 -O3 -o stage_bw stage_bw.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define TN 64        // rows of N per block, as in kernels/gemm.cu
#define KC 64        // k-chunk
#define GROUPS 4     // KC / 16
#define BLOCK 128

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// A: contiguous 16 B per thread.
__global__ void read_contig(const uint4* __restrict__ w, size_t n4,
                            unsigned long long* out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long s = 0;
    for (; i < n4; i += stride) {
        const uint4 v = __ldg(&w[i]);
        s += v.x + v.y + v.z + v.w;
    }
    if (s == 0xdeadbeefULL) *out = s;
}

// B: the staging pattern. Thread u takes row (nbase + u/GROUPS) and the 16 B at
// k-offset c*KC + (u%GROUPS)*16 -- exactly what stage_wtile does, minus the
// dequant and the shared store.
__global__ void read_staging(const uint2* __restrict__ w, int K, int N, int nchunk,
                             unsigned long long* out) {
    const int nbase = blockIdx.x * TN;
    unsigned long long s = 0;
    for (int c = 0; c < nchunk; ++c) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int u = threadIdx.x + p * BLOCK;
            const int nl = u / GROUPS, gr = u % GROUPS;
            const int n = nbase + nl;
            if (n < N) {
                const size_t off = (size_t)n * (K >> 1) + c * KC + gr * 16;
                const uint2 v = __ldg(reinterpret_cast<const uint2*>(
                    reinterpret_cast<const char*>(w) + off));
                s += v.x + v.y;
            }
        }
    }
    if (s == 0xdeadbeefULL) *out = s;
}

int main(int argc, char** argv) {
    // One layer's NVFP4 weights are ~150 MB; the model streams ~9.63 GB of them
    // across 64 layers, so a 150 MB buffer repeated 64x reproduces both the
    // working-set size and the total traffic.
    const size_t bytes = 150ull << 20;
    const int K = 5120, N = 17408, nchunk = K / KC;
    const int reps = 64;

    void* buf = nullptr;
    check(cudaMalloc(&buf, bytes), "cudaMalloc");
    check(cudaMemset(buf, 0x5a, bytes), "cudaMemset");
    unsigned long long* out = nullptr;
    check(cudaMalloc(&out, sizeof(*out)), "cudaMalloc out");

    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");

    const int gridA = (int)(bytes / 16 / BLOCK);
    const int gridB = (N + TN - 1) / TN;

    for (int which = 0; which < 2; ++which) {
        // warm
        if (which == 0)
            read_contig<<<gridA, BLOCK>>>((const uint4*)buf, bytes / 16, out);
        else
            read_staging<<<gridB, BLOCK>>>((const uint2*)buf, K, N, nchunk, out);
        check(cudaDeviceSynchronize(), "warmup");

        check(cudaEventRecord(a), "record");
        for (int r = 0; r < reps; ++r) {
            if (which == 0)
                read_contig<<<gridA, BLOCK>>>((const uint4*)buf, bytes / 16, out);
            else
                read_staging<<<gridB, BLOCK>>>((const uint2*)buf, K, N, nchunk, out);
        }
        check(cudaEventRecord(b), "record");
        check(cudaEventSynchronize(b), "sync");

        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, a, b), "elapsed");
        // The staging pattern reads the whole N x K/2 matrix per rep -- which is
        // smaller than the buffer -- so count what it actually touches.
        const double per_rep = (which == 0) ? (double)bytes
                                            : (double)N * (K / 2);
        const double gb = per_rep * reps / 1e9;
        std::printf("%-10s %8.1f ms   %7.1f GB   %7.1f GB/s\n",
                    which == 0 ? "contig" : "staging", ms, gb, gb / (ms / 1000.0));
    }

    check(cudaFree(buf), "free");
    check(cudaFree(out), "free out");
    return 0;
}
