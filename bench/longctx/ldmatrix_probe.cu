// ldmatrix lane->element map probe for sm_121, used to settle the fragment
// layouts of `attn_prefill_fa2_kernel` (kernels/elementwise.cu) empirically
// rather than by reasoning about the PTX docs.
//
// Why this exists: the FA2 prefill design rests on two claims that are easy to
// get subtly wrong and impossible to debug from the kernel's output alone --
//   (1) QK^T with M=qcols and B=K loads as a PLAIN row-major
//       `ldmatrix.m8n8.x4` on the `[key][head_dim]` tile (so B is the mma's
//       "column-major" operand because a key's head_dim is contiguous), and
//   (2) P*V loads B=V^T with `ldmatrix.m8n8.x4.trans` straight off the
//       `[key][dv]` tile, so V is never stored transposed.
// A wrong lane->element map here produces a plausible-looking but wrong kernel
// (it was the difference between a 4x scale error and a correct one in an
// earlier iteration of this file), so it is worth a committed artifact: any
// change of tile shape or of the smem swizzle invalidates the mapping and this
// probe is how you re-derive it in one command.
//
// It fills a 16x16 fp16 smem tile with s[row][col] = row*100 + col and prints,
// for each of the 32 lanes, the (value,value) pair held in every destination
// register, for four loads:
//   * x4 non-trans with the QK^T B-operand lane addressing
//   * x4.trans with the P*V lane addressing, and
//   * x4.trans with the alternate (llama.cpp-style) addressing, for comparison
//   * x2 non-trans
// Read the output as "r_j of lane L holds matrix_j[L/4][2*(L%%4), +1]" for
// non-trans, and "matrix_j[2*(L%%4)][L/4], matrix_j[2*(L%%4)+1][L/4]" for .trans.
// The thing that is NOT obvious from the PTX doc and that this probe pins down:
// for x4.trans the destination registers come out
//   {b0(dv_lo), b0(dv_hi), b1(dv_lo), b1(dv_hi)}
// not {b0(dv_lo), b1(dv_lo), b0(dv_hi), b1(dv_hi)} -- the 1<->2 swap that
// ggml's `mma.cuh:884-894` bakes into its asm output list
// (`"=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])`).
//
// Build and run:  nvcc -arch=sm_121 -O2 -o ldmatrix_probe ldmatrix_probe.cu && ./ldmatrix_probe
// Tile is 16 rows x 16 halves, s[row][col] = row*100 + col.
#include <cstdio>
#include <cuda_fp16.h>

__device__ __forceinline__ unsigned sa(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__global__ void probe(unsigned* out) {
    __shared__ __half s[16][16];
    const int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 256; i += 32) {
        s[i / 16][i % 16] = __float2half((float)((i / 16) * 100 + (i % 16)));
    }
    __syncwarp();

    // ---- non-trans x4: lanes 0-7 -> rows 0-7 col 0; 8-15 -> rows 8-15 col 0;
    //      16-23 -> rows 0-7 col 8; 24-31 -> rows 8-15 col 8.
    {
        const int r = (lane & 7) + ((lane & 16) ? 8 : 0);
        const int c = ((lane & 8) ? 8 : 0);
        unsigned x[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
                     : "r"(sa(&s[r][c])));
        for (int j = 0; j < 4; ++j) out[lane * 4 + j] = x[j];
    }
    __syncwarp();
    // ---- trans x4 with the same lane addressing.
    {
        const int r = (lane & 7) + ((lane & 16) ? 8 : 0);
        const int c = ((lane & 8) ? 8 : 0);
        unsigned y[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(y[0]), "=r"(y[1]), "=r"(y[2]), "=r"(y[3])
                     : "r"(sa(&s[r][c])));
        for (int j = 0; j < 4; ++j) out[128 + lane * 4 + j] = y[j];
    }
    __syncwarp();
    // ---- trans x4 with the llama.cpp-style lane addressing (rows use lane&8,
    //      cols use lane&16) for comparison.
    {
        const int r = (lane & 7) + ((lane & 8) ? 8 : 0);
        const int c = ((lane & 16) ? 8 : 0);
        unsigned y[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(y[0]), "=r"(y[1]), "=r"(y[2]), "=r"(y[3])
                     : "r"(sa(&s[r][c])));
        for (int j = 0; j < 4; ++j) out[256 + lane * 4 + j] = y[j];
    }
    __syncwarp();
    // ---- x2 non-trans: lanes 0-7 -> rows 0-7 col 0; lanes 8-15 -> rows 0-7 col 8.
    {
        const int r = lane & 7;
        const int c = ((lane & 8) ? 8 : 0);
        unsigned x[2];
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(x[0]), "=r"(x[1])
                     : "r"(sa(&s[r][c])));
        out[384 + lane * 2 + 0] = x[0];
        out[384 + lane * 2 + 1] = x[1];
    }
}

static void unpack(unsigned u, int& a, int& b) {
    __half2 h = *reinterpret_cast<__half2*>(&u);
    a = (int)__half2float(__low2half(h));
    b = (int)__half2float(__high2half(h));
}

int main() {
    unsigned* d;
    cudaMalloc(&d, 512 * sizeof(unsigned));
    probe<<<1, 32>>>(d);
    cudaDeviceSynchronize();
    unsigned h[512];
    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);

    printf("=== x4 non-trans, QK^T B-operand addressing: lanes 0-7 -> (rows 0-7, col 0),\n"
           "    8-15 -> (rows 0-7, col 8), 16-23 -> (rows 8-15, col 0), 24-31 -> (rows 8-15, col 8).\n"
           "    Expect r0=b0(nt), r1=b1(nt), r2=b0(nt+1), r3=b1(nt+1), i.e. lane L r_j = M_j[L/4][2*(L%%4), +1].\n");
    for (int lane = 0; lane < 32; ++lane) {
        printf("lane %2d:", lane);
        for (int j = 0; j < 4; ++j) {
            int a, b; unpack(h[lane * 4 + j], a, b);
            printf("  r%d=(%d,%d)", j, a, b);
        }
        printf("\n");
    }
    printf("=== x4.trans, P*V addressing A (rows use lane&16, cols use lane&8).\n"
           "    Expect r0=b0(dv_lo) r1=b0(dv_hi) r2=b1(dv_lo) r3=b1(dv_hi): lane L r_j = M_j[2*(L%%4), +1][L/4].\n");
    for (int lane = 0; lane < 32; ++lane) {
        printf("lane %2d:", lane);
        for (int j = 0; j < 4; ++j) {
            int a, b; unpack(h[128 + lane * 4 + j], a, b);
            printf("  r%d=(%d,%d)", j, a, b);
        }
        printf("\n");
    }
    printf("=== x4.trans, P*V addressing B (rows use lane&8, cols use lane&16) -- same fragments, different lanes.\n");
    for (int lane = 0; lane < 32; ++lane) {
        printf("lane %2d:", lane);
        for (int j = 0; j < 4; ++j) {
            int a, b; unpack(h[256 + lane * 4 + j], a, b);
            printf("  r%d=(%d,%d)", j, a, b);
        }
        printf("\n");
    }
    printf("=== x2 non-trans: lanes 0-7 -> (rows 0-7, col 0), lanes 8-15 -> (rows 0-7, col 8).\n"
           "    Expect r0=b0, r1=b1.\n");
    for (int lane = 0; lane < 32; ++lane) {
        int a0, b0, a1, b1;
        unpack(h[384 + lane * 2 + 0], a0, b0);
        unpack(h[384 + lane * 2 + 1], a1, b1);
        printf("lane %2d:  r0=(%d,%d)  r1=(%d,%d)\n", lane, a0, b0, a1, b1);
    }
    return 0;
}
