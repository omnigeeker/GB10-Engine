// Definitive GB10 read-bandwidth probe: incompressible (pseudo-random) data,
// high occupancy, several access shapes.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("ERR %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

__global__ void fill_rand(uint4* __restrict__ p, long n4, unsigned seed){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    const long stride=(long)gridDim.x*blockDim.x;
    unsigned s = seed + (unsigned)i*2654435761u;
    for(; i<n4; i+=stride){
        uint4 v;
        v.x = (s = s*1664525u+1013904223u);
        v.y = (s = s*1664525u+1013904223u);
        v.z = (s = s*1664525u+1013904223u);
        v.w = (s = s*1664525u+1013904223u);
        p[i]=v;
    }
}

__global__ void read_k(const uint4* __restrict__ in, unsigned* __restrict__ out, long n4){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    const long stride=(long)gridDim.x*blockDim.x;
    unsigned acc=0;
    for(; i<n4; i+=stride){
        uint4 v = in[i];
        acc += v.x^v.y^v.z^v.w;
    }
    if(acc==0xDEADBEEFu) out[0]=acc;
}

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s  SMs=%d\n\n", p.name, p.multiProcessorCount);
    size_t bytes = 16ull<<30; long n4=bytes/16;
    uint4* b; unsigned* o;
    CK(cudaMalloc(&b,bytes)); CK(cudaMalloc(&o,4));
    fill_rand<<<p.multiProcessorCount*64,256>>>(b,n4,12345u);
    CK(cudaDeviceSynchronize());
    printf("filled 16 GiB with incompressible pseudo-random data\n\n");
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int cfgs[][2] = {{48*32,256},{48*64,256},{48*128,256},{48*64,512},{48*32,512},{48*256,256}};
    for(auto&c : cfgs){
        int blocks=c[0], threads=c[1], iters=5;
        read_k<<<blocks,threads>>>(b,o,n4); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for(int i=0;i<iters;i++) read_k<<<blocks,threads>>>(b,o,n4);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
        printf("  blocks=%-6d threads=%-4d  %8.2f ms  %7.1f GB/s\n", blocks, threads, ms, bytes/(ms*1e-3)/1e9);
    }
    cudaFree(b); cudaFree(o); return 0;
}
