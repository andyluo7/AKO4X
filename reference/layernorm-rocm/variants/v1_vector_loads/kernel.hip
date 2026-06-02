// === Identity ===
// v1_vector_loads — LayerNorm with 128-bit (uint4 = 8xbf16) vector loads/writes.
// Same win pattern as RMSNorm v1: vectorize global memory access, keep scalar
// LDS reduction.
//
// === Delta from parent (hip_stock) ===
// 1) Loads/stores 8 bf16 elements per memory transaction via uint4 + union.
// 2) Reduces both sum and sum-of-squares in the same vector-load loop (single
//    pass over x, two parallel LDS reductions).
// 3) Vectorized scale + bias + store loop.
//
// === Lessons (carry-over from RMSNorm v1) ===
// WHEN hidden % 8 == 0 AND tensors 16-byte aligned (true for torch tensors)
//   → uint4 reinterpret loads always win, no scalar tail needed.
//
// === Dead-ends (inherited from RMSNorm v2) ===
// * Persistent kernel — hurts at n_rows >> CU_COUNT on CDNA.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define BLOCK 256
#define VEC 8

typedef union {
    uint4 u;
    __hip_bfloat16 bf[VEC];
} bf16x8_t;

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

    const uint4* x_row = reinterpret_cast<const uint4*>(x_ + row * hidden);
    const uint4* w_vec = reinterpret_cast<const uint4*>(w_);
    const uint4* b_vec = reinterpret_cast<const uint4*>(b_);
    uint4*       y_row = reinterpret_cast<uint4*>(y_ + row * hidden);
    const int n_vec = hidden / VEC;

    // Pass 1: vectorized sum + sum-of-squares.
    float my_sum = 0.0f, my_ssq = 0.0f;
    for (int i = tid; i < n_vec; i += BLOCK) {
        bf16x8_t v; v.u = x_row[i];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float f = static_cast<float>(v.bf[k]);
            my_sum += f;
            my_ssq += f * f;
        }
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

    // Pass 3: vectorized normalize + scale + bias + store.
    for (int i = tid; i < n_vec; i += BLOCK) {
        bf16x8_t v, wv, bv, out;
        v.u  = x_row[i];
        wv.u = w_vec[i];
        bv.u = b_vec[i];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float f  = static_cast<float>(v.bf[k]);
            float g  = static_cast<float>(wv.bf[k]);
            float bb = static_cast<float>(bv.bf[k]);
            out.bf[k] = static_cast<__hip_bfloat16>((f - mean) * inv * g + bb);
        }
        y_row[i] = out.u;
    }
}

extern "C" void layernorm_launch(
    const void* x, const void* w, const void* b, void* y,
    int n_rows, int hidden, float eps, hipStream_t stream)
{
    layernorm_kernel<<<dim3(n_rows), dim3(BLOCK), 0, stream>>>(
        static_cast<const __hip_bfloat16*>(x),
        static_cast<const __hip_bfloat16*>(w),
        static_cast<const __hip_bfloat16*>(b),
        static_cast<__hip_bfloat16*>(y),
        n_rows, hidden, eps);
}
