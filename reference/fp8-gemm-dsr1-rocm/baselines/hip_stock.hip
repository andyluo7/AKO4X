// === Identity ===
// hip_stock — naive FP8 GEMM, one output per thread, fp32 accum, scalar loads.
// Correctness anchor for variants. Expected to be very slow; the only goal is
// "matches torch_naive on a 32x32 probe so we trust the variant comparison."
//
// Shape:
//   a       [M, K] fp8_e4m3fnuz (row-major, K-contig)
//   b       [N, K] fp8_e4m3fnuz (row-major, K-contig — weight layout)
//   scale_a [M] fp32
//   scale_b [N] fp32
//   c       [M, N] bf16
//
// Grid: (ceil(M/BM), ceil(N/BN)), block: (BM*BN) — one thread per output.
// No LDS, no MFMA, no vectorization.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#define BM 16
#define BN 16

__global__ void fp8_gemm_stock_kernel(
    const __hip_fp8_e4m3_fnuz* __restrict__ a,
    const __hip_fp8_e4m3_fnuz* __restrict__ b,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    __hip_bfloat16* __restrict__ c,
    int M, int N, int K)
{
    int m = blockIdx.x * BM + threadIdx.x;
    int n = blockIdx.y * BN + threadIdx.y;
    if (m >= M || n >= N) return;

    float acc = 0.f;
    const __hip_fp8_e4m3_fnuz* a_row = a + m * K;
    const __hip_fp8_e4m3_fnuz* b_row = b + n * K;
    for (int k = 0; k < K; ++k) {
        float av = static_cast<float>(a_row[k]);
        float bv = static_cast<float>(b_row[k]);
        acc += av * bv;
    }
    acc *= scale_a[m] * scale_b[n];
    c[m * N + n] = static_cast<__hip_bfloat16>(acc);
}

extern "C" void fp8_gemm_launch(
    const void* a, const void* b,
    const void* scale_a, const void* scale_b,
    void* c, int M, int N, int K, hipStream_t stream)
{
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BM, BN);
    fp8_gemm_stock_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __hip_fp8_e4m3_fnuz*>(a),
        static_cast<const __hip_fp8_e4m3_fnuz*>(b),
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b),
        static_cast<__hip_bfloat16*>(c),
        M, N, K);
}
