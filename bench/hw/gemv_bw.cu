// Does the GEMV's own weight access pattern reach the 228 GB/s that bench/hw/bw4.cu
// measures with a different pattern?
//
// This is the round-128 decisive experiment for T1. The single-stream GEMV runs at
// ~190 GB/s = 83% of the probe figure, and eight candidate explanations have been
// eliminated (x traffic, bytes in flight, DRAM latency, occupancy, local memory,
// weight stream, load instruction mix, and the warp reorganisation that turned out
// to be already implemented). If this probe also returns ~190 GB/s, then 190 is the
// ceiling of the pattern and T1's 12.5 tok/s is not a kernel problem.
//
// The pattern, copied exactly from nvfp4_gemv_tmpl in kernels/gemv.cu:
//
//     e0 = i * kTile + lane * kVec          kTile = 512, kVec = 16
//     pk = *(const uint2*)(w + row*rowbytes + (e0 >> 1))
//
// so lane L touches bytes [8L, 8L+8) of a packed 4-bit row and one warp instruction
// covers 256 contiguous bytes of a single row. ROWS rows are kept in flight per
// warp, matching the "issue every row's load before any arithmetic" structure.
//
// Build: nvcc -arch sm_121 -O3 --use_fast_math -o /tmp/gemv_bw bench/hw/gemv_bw.cu
// Run:   /tmp/gemv_bw [N] [K] [ROWS] [reps]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(c)                                                                        \
    do {                                                                                \
        cudaError_t e = (c);                                                            \
        if (e != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),     \
                         __FILE__, __LINE__);                                           \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

constexpr int kTile = 512;
constexpr int kVec = 16;
constexpr int kWarp = 32;

// One warp per row group; lane L reads 8 bytes at [8L,8L+8) of each 256-byte span.
template <int ROWS>
__global__ void gemv_pattern_bw(const uint8_t* __restrict__ w, float* __restrict__ out,
                                int N, int K) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int rowbytes = K >> 1;
    const int full_tiles = K / kTile;
    const int row_stride = gridDim.x * nwarps * ROWS;

    for (int rbase = (blockIdx.x * nwarps + warp) * ROWS; rbase < N; rbase += row_stride) {
        uint2 acc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = make_uint2(0, 0);

        for (int i = 0; i < full_tiles; ++i) {
            const int e0 = i * kTile + lane * kVec;
            uint2 pk[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = rbase + r;
                if (row < N)
                    pk[r] = *reinterpret_cast<const uint2*>(w + (size_t)row * rowbytes + (e0 >> 1));
            }
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (rbase + r < N) {
                    acc[r].x ^= pk[r].x;
                    acc[r].y ^= pk[r].y;
                }
            }
        }

        // One store per warp per row group, so the measurement is load-dominated.
        uint32_t s = 0;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) s ^= acc[r].x ^ acc[r].y;
        if (lane == 0 && rbase < N) out[rbase] = (float)s;
    }
}

template <int ROWS>
static double run(const uint8_t* d_w, float* d_out, int N, int K, int blocks, int threads,
                  int reps, size_t bytes) {
    gemv_pattern_bw<ROWS><<<blocks, threads>>>(d_w, d_out, N, K);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) gemv_pattern_bw<ROWS><<<blocks, threads>>>(d_w, d_out, N, K);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return (double)bytes * reps / (ms / 1e3) / 1e9;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 17408;
    const int K = argc > 2 ? std::atoi(argv[2]) : 5120;
    const int ROWS = argc > 3 ? std::atoi(argv[3]) : 1;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 20;

    const size_t bytes = (size_t)N * (K / 2);
    std::printf("pattern: one row per warp, 256 B per warp instruction\n");
    std::printf("matrix : N=%d K=%d -> %.1f MB, ROWS=%d, reps=%d\n", N, K,
                bytes / 1e6, ROWS, reps);

    uint8_t* d_w;
    float* d_out;
    CHECK(cudaMalloc(&d_w, bytes));
    CHECK(cudaMalloc(&d_out, sizeof(float) * N));
    CHECK(cudaMemset(d_w, 0xA5, bytes));

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    const int threads = 128;
    const int blocks = p.multiProcessorCount * 8;

    const size_t total = bytes * (size_t)reps;
    double gbs = 0;
    if (ROWS == 1) gbs = run<1>(d_w, d_out, N, K, blocks, threads, reps, bytes);
    else if (ROWS == 2) gbs = run<2>(d_w, d_out, N, K, blocks, threads, reps, bytes);
    else if (ROWS == 4) gbs = run<4>(d_w, d_out, N, K, blocks, threads, reps, bytes);
    else { std::fprintf(stderr, "ROWS must be 1, 2 or 4\n"); return 1; }

    std::printf("result : %.1f GB/s  (%.1f MB in total, blocks=%d threads=%d)\n", gbs,
                total / 1e6, blocks, threads);
    std::printf("verdict: vs the 228 GB/s bw4.cu figure -> %.1f%%\n", 100.0 * gbs / 228.0);
    if (gbs < 200.0)
        std::printf("         ~190 means the pattern is the ceiling and T1 is not a kernel problem\n");
    else
        std::printf("         ~228 means the kernel has bandwidth to find\n");

    CHECK(cudaFree(d_w));
    CHECK(cudaFree(d_out));
    return 0;
}
