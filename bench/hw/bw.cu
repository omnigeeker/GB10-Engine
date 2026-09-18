// GB10 memory bandwidth + basic compute characterization
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1;} } while(0)

__global__ void read_kernel(const float4* __restrict__ in, float* __restrict__ out, long n4) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n4; i += stride) {
        float4 v = __ldcs(&in[i]);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == -1.0f) out[0] = acc;   // never true; prevents DCE
}

__global__ void write_kernel(float4* __restrict__ out, long n4) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float4 v = make_float4(1.f,2.f,3.f,4.f);
    for (; i < n4; i += stride) __stcs(&out[i], v);
}

__global__ void triad_kernel(float4* __restrict__ a, const float4* __restrict__ b, long n4) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) {
        float4 x = __ldcs(&b[i]);
        float4 y = a[i];
        y.x += x.x; y.y += x.y; y.z += x.z; y.w += x.w;
        a[i] = y;
    }
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("Device            : %s\n", p.name);
    printf("SM count          : %d\n", p.multiProcessorCount);
    printf("Compute capability: %d.%d\n", p.major, p.minor);
    int smClock=0, memClock=0, busWidth=0;
    cudaDeviceGetAttribute(&smClock,  cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&memClock, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("Clock rate        : %.0f MHz\n", smClock/1000.0);
    printf("Mem clock         : %.0f MHz  bus %d-bit\n", memClock/1000.0, busWidth);
    printf("Theoretical BW    : %.1f GB/s\n", 2.0*memClock*(busWidth/8)/1.0e6);

    size_t bytes = 8ull<<30;              // 8 GiB buffer
    long n4 = bytes/16;
    float4 *a, *b; float* o;
    CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMalloc(&o, 4));
    CK(cudaMemset(a, 1, bytes)); CK(cudaMemset(b, 2, bytes));

    int blocks = p.multiProcessorCount * 32;
    int threads = 256;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // warmup
    read_kernel<<<blocks,threads>>>(b, o, n4);
    CK(cudaDeviceSynchronize());

    auto bench = [&](const char* name, double moved_bytes, auto fn) {
        int iters = 10;
        CK(cudaEventRecord(e0));
        for (int i=0;i<iters;i++) fn();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= iters;
        printf("%-22s %8.2f ms   %8.1f GB/s\n", name, ms, moved_bytes/(ms*1e-3)/1e9);
        return 0;
    };

    bench("read (8GiB)",        (double)bytes, [&]{ read_kernel<<<blocks,threads>>>(b,o,n4); });
    bench("write (8GiB)",       (double)bytes, [&]{ write_kernel<<<blocks,threads>>>(a,n4); });
    bench("triad (r+w 16GiB)",  2.0*bytes,     [&]{ triad_kernel<<<blocks,threads>>>(a,b,n4); });

    // D2D copy
    bench("D2D memcpy (16GiB)", 2.0*bytes,     [&]{ cudaMemcpy(a,b,bytes,cudaMemcpyDeviceToDevice); });

    cudaFree(a); cudaFree(b); cudaFree(o);
    return 0;
}
