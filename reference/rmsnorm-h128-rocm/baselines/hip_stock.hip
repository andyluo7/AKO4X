// hip_stock.hip — naive HIP RMSNorm baseline for AKO4X-rocm Phase 1.
//
// Two kernels, dispatched by the wrapper based on `hidden`:
//   1) rmsnorm_h128_kernel — fixed BLOCK=128 = HIDDEN, single LDS reduction.
//      Wins decisively at h=128 (small per-row work, single-pass).
//   2) rmsnorm_dynamic_kernel — BLOCK=256 chunked reduction for arbitrary hidden.
//      Each thread handles ceil(hidden/256) elements; single LDS reduction.
//      Intentionally naive — no LDS bank-conflict padding, no vector loads,
//      no wave64 cross-lane reduction, no persistent kernel.
//
// Both kernels accumulate sum-of-squares in fp32 (bf16 has too much error in
// the variance) and write back in bf16.
//
// Expected to be CLEARLY beatable by the agent at h>=4096 via:
//   - 128-bit (8×bf16) vector loads — 8× memory throughput per thread
//   - wave64 cross-lane reduction via __builtin_amdgcn_ds_bpermute (saves LDS)
//   - LDS bank-conflict padding (BLOCK + 1 stride)
//   - persistent kernel with grid = CU count (256 on gfx950), each block
//     processes multiple rows — wins on the small-n_rows regime

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define HIDDEN_128 128
#define BLOCK_128 128
#define BLOCK_DYN 256

__global__ void rmsnorm_h128_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ w,
          __hip_bfloat16* __restrict__ y,
    int n_rows,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n_rows) return;

    __shared__ float ssq[BLOCK_128];
    float val = static_cast<float>(x[row * HIDDEN_128 + tid]);
    ssq[tid] = val * val;
    __syncthreads();

    for (int s = BLOCK_128 / 2; s > 0; s >>= 1) {
        if (tid < s) ssq[tid] += ssq[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(ssq[0] / static_cast<float>(HIDDEN_128) + eps);
    float w_val = static_cast<float>(w[tid]);
    y[row * HIDDEN_128 + tid] = static_cast<__hip_bfloat16>(val * inv_rms * w_val);
}

__global__ void rmsnorm_dynamic_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ w,
          __hip_bfloat16* __restrict__ y,
    int n_rows,
    int hidden,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n_rows) return;

    __shared__ float ssq[BLOCK_DYN];
    const __hip_bfloat16* x_row = x + row * hidden;

    // Pass 1: per-thread sum-of-squares over strided chunks.
    float my_ssq = 0.0f;
    for (int i = tid; i < hidden; i += BLOCK_DYN) {
        float v = static_cast<float>(x_row[i]);
        my_ssq += v * v;
    }
    ssq[tid] = my_ssq;
    __syncthreads();

    // Pass 2: block-wide tree reduction.
    for (int s = BLOCK_DYN / 2; s > 0; s >>= 1) {
        if (tid < s) ssq[tid] += ssq[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(ssq[0] / static_cast<float>(hidden) + eps);

    // Pass 3: scale + store.
    __hip_bfloat16* y_row = y + row * hidden;
    for (int i = tid; i < hidden; i += BLOCK_DYN) {
        float v = static_cast<float>(x_row[i]);
        float wv = static_cast<float>(w[i]);
        y_row[i] = static_cast<__hip_bfloat16>(v * inv_rms * wv);
    }
}

extern "C" void rmsnorm_launch(
    const void* x, const void* w, void* y,
    int n_rows, int hidden, float eps, hipStream_t stream)
{
    if (hidden == HIDDEN_128) {
        dim3 grid(n_rows);
        dim3 block(BLOCK_128);
        rmsnorm_h128_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __hip_bfloat16*>(x),
            static_cast<const __hip_bfloat16*>(w),
            static_cast<__hip_bfloat16*>(y),
            n_rows, eps);
    } else {
        dim3 grid(n_rows);
        dim3 block(BLOCK_DYN);
        rmsnorm_dynamic_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __hip_bfloat16*>(x),
            static_cast<const __hip_bfloat16*>(w),
            static_cast<__hip_bfloat16*>(y),
            n_rows, hidden, eps);
    }
}
