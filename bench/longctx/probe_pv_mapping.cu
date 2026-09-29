// Validate the P.V fragment mapping in the ORIGINAL orientation:
//   O[16 rows][16 dims] = P[16][16 keys] . V[16 keys][16 dims]
//   A = P  via ldmatrix.x4        (row-major [row][key], stride 16)
//   B = V^T via ldmatrix.x2.trans (from V's natural [key][dim], stride 24)
// Two n-tiles of 8 dims, so 2 mma per warp.
#include <cstdio>
#include <cuda_fp16.h>
#define VS 24          // halfs; multiple of 8 => rows are 16-byte aligned

__device__ __forceinline__ unsigned sa(const void* p){ return (unsigned)__cvta_generic_to_shared(p); }

__global__ void pv(const __half* P, const __half* V, float* O) {
    __shared__ __half Ps[16*16], Vs[16*VS];
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2, t4 = lane & 3, c = t4*2;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) Ps[(i/16)*16 + i%16] = P[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) Vs[(i/16)*VS + i%16] = V[i];
    __syncwarp();
    const int arow = (lane < 16) ? lane : (lane - 16);
    const int acol = (lane < 16) ? 0 : 8;
    unsigned a[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3])
        : "r"(sa(&Ps[arow*16 + acol])));
    const int krow = lane & 15;          // x2 reads lanes 0-15; keep the rest in range
    for (int nt = 0; nt < 2; ++nt) {
        const int n0 = nt*8;
        unsigned b[2];
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(b[0]),"=r"(b[1])
            : "r"(sa(&Vs[krow*VS + n0])));
        float d[4] = {0,0,0,0};
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
        O[gid*16 + n0 + t4*2]         = d[0];
        O[gid*16 + n0 + t4*2 + 1]     = d[1];
        O[(gid+8)*16 + n0 + t4*2]     = d[2];
        O[(gid+8)*16 + n0 + t4*2 + 1] = d[3];
    }
}
int main(){
    __half P[256], V[256]; float want[256], got[256];
    for (int i=0;i<256;++i) P[i]=__float2half((float)((i*3+1)%7-3)*0.25f);
    for (int i=0;i<256;++i) V[i]=__float2half((float)((i*5+2)%9-4)*0.5f);
    for (int m=0;m<16;++m) for(int n=0;n<16;++n){ float s=0;
        for(int k=0;k<16;++k) s+=__half2float(P[m*16+k])*__half2float(V[k*16+n]);
        want[m*16+n]=s; }
    __half *dP,*dV; float *dO;
    cudaMalloc(&dP,sizeof P); cudaMalloc(&dV,sizeof V); cudaMalloc(&dO,sizeof got);
    cudaMemcpy(dP,P,sizeof P,cudaMemcpyHostToDevice);
    cudaMemcpy(dV,V,sizeof V,cudaMemcpyHostToDevice);
    pv<<<1,32>>>(dP,dV,dO);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){printf("FAILED: %s\n",cudaGetErrorString(e));return 1;}
    cudaMemcpy(got,dO,sizeof got,cudaMemcpyDeviceToHost);
    int bad=0; for(int i=0;i<256;++i) if(got[i]!=want[i]) ++bad;
    printf("P.V mapping: A=ldmatrix.x4(P[row][key])  B=ldmatrix.x2.trans(V[key][dim] stride %d)\n",VS);
    if(!bad) printf("  EXACT: 256/256 match the CPU reference\n");
    else { printf("  MISMATCH %d/256\n",bad);
      for(int m=0;m<3;++m){for(int n=0;n<6;++n)printf("   O[%d][%d] want %7.3f got %7.3f\n",m,n,want[m*16+n],got[m*16+n]);}}
    return bad!=0;
}
