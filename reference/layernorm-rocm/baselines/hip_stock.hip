// hip_stock.hip — naive HIP LayerNorm baseline.
//
// One block per row, BLOCK=256 threads, chunked sum + sum-of-squares in fp32,
// single LDS reduction for each. Block-sized to handle any hidden dim that's a
// multiple of BLOCK (true for 128, 4096, etc.). No vector loads, no MFMA, no
// persistent kernels — same shape as the RMSNorm baseline.
//
// y = (x - mean) / sqrt(var + eps) * w + b
//   where mean = sum(x)/N, var = sum(x^2)/N - mean^2.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BLOCK 256

__global__ void layernorm_kernel(
    const __hip_bfloat16* __restrict__ x_,
    const __hip_bfloat16* __restrict__ w_,
    const __hip_bfloat16* __restrict__ b_,
          __hip_bfloat16* __restrict__ y_,
    int n_rows,
    int hidden,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n_rows) return;

    __shared__ float ssum[BLOCK];
    __shared__ float ssq[BLOCK];

    const __hip_bfloat16* x_row = x_ + row * hidden;

    // Pass 1: sum and sum-of-squares in one loop.
    float my_sum = 0.0f, my_ssq = 0.0f;
    for (int i = tid; i < hidden; i += BLOCK) {
        float v = static_cast<float>(x_row[i]);
        my_sum += v;
        my_ssq += v * v;
    }
    ssum[tid] = my_sum;
    ssq[tid]  = my_ssq;
    __syncthreads();

    // Pass 2: two parallel tree reductions.
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssq[tid]  += ssq[tid + s];
        }
        __syncthreads();
    }

    float N    = static_cast<float>(hidden);
    float mean = ssum[0] / N;
    float var  = ssq[0]  / N - mean * mean;
    float inv  = rsqrtf(var + eps);

    // Pass 3: normalize + scale + bias.
    __hip_bfloat16* y_row = y_ + row * hidden;
    for (int i = tid; i < hidden; i += BLOCK) {
        float v  = static_cast<float>(x_row[i]);
        float wv = static_cast<float>(w_[i]);
        float bv = static_cast<float>(b_[i]);
        y_row[i] = static_cast<__hip_bfloat16>((v - mean) * inv * wv + bv);
    }
}

extern "C" void layernorm_launch(
    const void* x, const void* w, const void* b, void* y,
    int n_rows, int hidden, float eps, hipStream_t stream)
{
    dim3 grid(n_rows);
    dim3 block(BLOCK);
    layernorm_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __hip_bfloat16*>(x),
        static_cast<const __hip_bfloat16*>(w),
        static_cast<const __hip_bfloat16*>(b),
        static_cast<__hip_bfloat16*>(y),
        n_rows, hidden, eps);
}
