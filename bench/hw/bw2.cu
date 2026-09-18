// Aggressive GB10 streaming-bandwidth probe: find the true achievable read BW.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("ERR %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

template<int VPT>
__global__ void read_v(const float4* __restrict__ in, float* __restrict__ out, long n4){
    long i=((long)blockIdx.x*blockDim.x+threadIdx.x);
    long stride=(long)gridDim.x*blockDim.x;
    float acc=0.f;
    for(; i<n4; i+=stride*VPT){
        #pragma unroll
        for(int k=0;k<VPT;k++){ long j=i+(long)k*stride; if(j<n4){ float4 v=__ldcs(&in[j]); acc+=v.x+v.y+v.z+v.w; } }
    }
    if(acc==-1.f) out[0]=acc;
}

int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    size_t bytes = 24ull<<30;
    long n4=bytes/16;
    float4* b; float* o;
    CK(cudaMalloc(&b,bytes)); CK(cudaMalloc(&o,4)); CK(cudaMemset(b,1,bytes));
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    read_v<8><<<p.multiProcessorCount*32,256>>>(b,o,n4);
    CK(cudaDeviceSynchronize());
    struct Cfg{int blocks,threads,vpt;} cfgs[]={{48*32,256,8},{48*64,256,8},{48*32,512,8},{48*64,512,16},{48*128,256,8}};
    for(auto c:cfgs){
        int iters=5;
        CK(cudaEventRecord(e0));
        for(int i=0;i<iters;i++){
            if(c.vpt==8) read_v<8><<<c.blocks,c.threads>>>(b,o,n4);
            else if(c.vpt==16) read_v<16><<<c.blocks,c.threads>>>(b,o,n4);
        }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
        printf("blocks=%-6d threads=%-4d vpt=%-3d  %7.2f ms  %7.1f GB/s\n",c.blocks,c.threads,c.vpt,ms,bytes/(ms*1e-3)/1e9);
    }
    cudaFree(b); cudaFree(o); return 0;
}
