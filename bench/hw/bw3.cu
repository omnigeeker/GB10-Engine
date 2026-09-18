// Compare load policies to find the true achievable read bandwidth on GB10.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("ERR %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

template<int POLICY>
__global__ void read_k(const float4* __restrict__ in, float* __restrict__ out, long n4){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    const long stride=(long)gridDim.x*blockDim.x;
    float acc=0.f;
    for(; i<n4; i+=stride){
        float4 v;
        if      (POLICY==0) v = in[i];              // default (cached)
        else if (POLICY==1) v = __ldg(&in[i]);      // read-only cache
        else if (POLICY==2) v = __ldcs(&in[i]);     // streaming / evict-first
        else                v = __ldcg(&in[i]);     // cache-global, skip L1
        acc += v.x+v.y+v.z+v.w;
    }
    if(acc==-1.f) out[0]=acc;
}

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s  SMs=%d\n\n", p.name, p.multiProcessorCount);
    for (size_t gib : {2ull, 4ull, 8ull}) {
        size_t bytes = gib<<30; long n4=bytes/16;
        float4* b; float* o;
        CK(cudaMalloc(&b,bytes)); CK(cudaMalloc(&o,4)); CK(cudaMemset(b,1,bytes));
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        const char* names[4]={"default","__ldg","__ldcs","__ldcg"};
        printf("--- buffer %zu GiB ---\n", gib);
        for(int pol=0; pol<4; ++pol){
            int blocks=p.multiProcessorCount*64, threads=256, iters=5;
            // warm
            if(pol==0) read_k<0><<<blocks,threads>>>(b,o,n4);
            if(pol==1) read_k<1><<<blocks,threads>>>(b,o,n4);
            if(pol==2) read_k<2><<<blocks,threads>>>(b,o,n4);
            if(pol==3) read_k<3><<<blocks,threads>>>(b,o,n4);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for(int i=0;i<iters;i++){
                if(pol==0) read_k<0><<<blocks,threads>>>(b,o,n4);
                if(pol==1) read_k<1><<<blocks,threads>>>(b,o,n4);
                if(pol==2) read_k<2><<<blocks,threads>>>(b,o,n4);
                if(pol==3) read_k<3><<<blocks,threads>>>(b,o,n4);
            }
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
            printf("  %-9s %8.2f ms  %7.1f GB/s\n", names[pol], ms, bytes/(ms*1e-3)/1e9);
        }
        cudaFree(b); cudaFree(o);
    }
    return 0;
}
